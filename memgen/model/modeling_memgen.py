import logging
from typing import Union, Optional, List

import random
import torch
import torch.nn as nn
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    GenerationConfig,
    DynamicCache
)
from transformers.modeling_utils import PreTrainedModel

from memgen.model.configuration_memgen import MemGenConfig
from memgen.model.modeling_utils import (
    MemGenOutputWithPast,
    MemGenLoraSwitchMixin,
    MemGenGenerationMixin,
)
from memgen.model.trigger import MemGenTrigger
from memgen.model.weaver import MemGenWeaver
from memgen.utils import (
    CONVERSATION_TEMPLATE,
    fix_model_parameters,
)

class MemGenModel(PreTrainedModel, MemGenLoraSwitchMixin, MemGenGenerationMixin):
    config_class = MemGenConfig

    def __init__(
        self, 
        config: MemGenConfig, 
        base_tokenizer,
        base_model: PreTrainedModel, 
        weaver_model: Optional[PreTrainedModel] = None,
        trigger_model: Optional[PreTrainedModel] = None,
    ):   
        super().__init__(config)
        
        self.config = config
        
        if weaver_model is None:
            weaver_model = base_model
        if trigger_model is None:
            trigger_model = base_model

        fix_model_parameters(base_model)
        fix_model_parameters(weaver_model)
        fix_model_parameters(trigger_model)

        # Get the target dtype from base model before LoRA insertion
        target_dtype = next(base_model.parameters()).dtype

        weaver_model, trigger_model = self._insert_lora_adapters(
            weaver_model, config.weaver_lora_config, trigger_model, config.trigger_lora_config,
        )

        # Ensure weaver and trigger models use the same dtype as base model
        weaver_model = weaver_model.to(target_dtype)
        trigger_model = trigger_model.to(target_dtype)

        self.weaver = MemGenWeaver(
            weaver_model, config.prompt_latents_len, config.inference_latents_len,
        )
        self.trigger = MemGenTrigger(
            trigger_model, config.trigger_active,
        )
        self.reasoner = base_model
        self.tokenizer = base_tokenizer
        
        # projection layers for mapping embeddings between reasoner and weaver
        # map reasoner input embeddings to weaver input embeddings
        reasoner_hidden_size = self.config.hidden_size
        weaver_hidden_size = weaver_model.base_model.config.hidden_size

        # Get the dtype from the base model
        model_dtype = next(base_model.parameters()).dtype

        self.reasoner_to_weaver = nn.Linear(
            reasoner_hidden_size, weaver_hidden_size, dtype=model_dtype
        )
        # Map weaver hidden states to reasoner input embeddings
        self.weaver_to_reasoner = nn.Linear(
            weaver_hidden_size, reasoner_hidden_size, dtype=model_dtype
        )
        
        self.delimiters: list[str] = [",", ".", "\n"]  # delimiters for detecting augmentation points

        # postprocess
        self._postprocess_models()

    def _postprocess_models(self):
        # Ensure tokenizer has a pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            self.tokenizer.padding_side = "left"
            logging.info(
                f"Tokenizer has no pad token. Using EOS token ({self.tokenizer.eos_token}) as pad token."
            )

        # Normalize the tokenizer's chat template
        self.tokenizer.chat_template = CONVERSATION_TEMPLATE
    

    @property
    def device(self):
        return self.reasoner.device
    
    def _forward(
        self, 
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,   
        **kwargs
    ) -> torch.Tensor:
        # preprocess inputs
        assert input_ids.shape == attention_mask.shape == labels.shape
        
        tokenizer = self.tokenizer
        reasoner = self.reasoner
        weaver = self.weaver
        delimiters = self.delimiters
        max_augment_num = self.config.max_inference_aug_num  # Limit the number of inference augmentation points to avoid excessive augmentation
        device = self.device
        embeds_dtype = reasoner.get_input_embeddings().weight.dtype
        B, _ = input_ids.shape
        hidden_size = self.config.hidden_size

        # select augment idx
        augmentation_indices = self._select_augment_points_after_delimiter(
            input_ids, labels, delimiters, tokenizer, max_augment_num
        )
        
        # origin inputs embeds
        inputs_embeds = reasoner.get_input_embeddings()(input_ids)
                
        # Initialize the start index and empty tensors for accumulating processed segments
        current_start_idx = 0
        current_inputs_embeds = torch.empty((B, 0, hidden_size), device=device, dtype=embeds_dtype)
        current_attention_mask = torch.empty((B, 0), device=device, dtype=attention_mask.dtype)
        current_latents_mask = torch.empty((B, 0), device=device, dtype=torch.bool)

        # Iterate over the selected augmentation points
        for aug_point_idx in augmentation_indices:
            # Slice the current segment of original embeddings and attention mask
            segment_inputs_embeds = inputs_embeds[:, current_start_idx:aug_point_idx]
            segment_attention_mask = attention_mask[:, current_start_idx:aug_point_idx]
            segment_latents_mask = torch.zeros((B, segment_inputs_embeds.size(1)), device=device, dtype=torch.bool)

            # Concatenate the current segment to the accumulated embeddings and masks
            current_inputs_embeds = torch.cat([current_inputs_embeds, segment_inputs_embeds], dim=1)
            current_attention_mask = torch.cat([current_attention_mask, segment_attention_mask], dim=1)
            current_position_ids = self._generate_position_ids(current_attention_mask)
            current_latents_mask = torch.cat([current_latents_mask, segment_latents_mask], dim=1)

            # Map reasoner embeddings to weaver embeddings for augmentation
            weaver_inputs_embeds = self.reasoner_to_weaver(current_inputs_embeds)

            # Determine whether this point is the end of the prompt (prompt augmentation)
            is_prompt_end_aug = (labels[:, aug_point_idx] != -100).all() and (labels[:, aug_point_idx-1] == -100).all().item()
            
            # Depending on type, use weaver to augment prompt or inference
            if is_prompt_end_aug:
                weaver_hidden_states, attn_mask, pos_ids = weaver.augment_prompt(
                    weaver_inputs_embeds, current_attention_mask, current_position_ids
                )
            else:
                weaver_hidden_states, attn_mask, pos_ids = weaver.augment_inference(
                    weaver_inputs_embeds, current_attention_mask, current_position_ids
                ) 

            # Map weaver hidden states back to reasoner embeddings
            latent_inputs_embeds = self.weaver_to_reasoner(weaver_hidden_states)

            # Update accumulated embeddings and masks with the newly augmented segment
            current_inputs_embeds = torch.cat([current_inputs_embeds, latent_inputs_embeds], dim=1)
            current_attention_mask = torch.cat([current_attention_mask, attn_mask], dim=1)
            current_start_idx = aug_point_idx
            
            # Update latent mask for the newly added latent embeddings
            latent_mask = torch.ones((B, latent_inputs_embeds.size(1)), device=device, dtype=torch.bool)
            current_latents_mask = torch.cat([current_latents_mask, latent_mask], dim=1)
            
        # Process the remaining segment after the last augmentation point
        remaining_inputs_embeds = inputs_embeds[:, current_start_idx:]
        remaining_attention_mask = attention_mask[:, current_start_idx:]
        latent_mask = torch.zeros((B, remaining_attention_mask.size(1)), device=device, dtype=torch.bool)
        
        current_inputs_embeds = torch.cat([current_inputs_embeds, remaining_inputs_embeds], dim=1)
        current_attention_mask = torch.cat([current_attention_mask, remaining_attention_mask], dim=1)
        current_position_ids = self._generate_position_ids(current_attention_mask)
        current_latents_mask = torch.cat([current_latents_mask, latent_mask], dim=1)

        reasoner_outputs = reasoner(
            inputs_embeds=current_inputs_embeds,
            attention_mask=current_attention_mask,
            position_ids=current_position_ids
        )
        logits = reasoner_outputs.logits
        
        # Identify valid positions in logits (positions that should contribute to loss)
        shifted = torch.zeros_like(current_latents_mask)
        shifted[:, :-1] = current_latents_mask[:, 1:]
        valid_mask = ~shifted
        
        valid_logits = logits[valid_mask].view(logits.size(0), -1, logits.size(2))  
        # assert shifted.sum() == current_latents_mask.sum()
        # assert valid_logits.shape[:2] == input_ids.shape
        return valid_logits
    
    def _instructional_forward(
        self, 
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,   
        **kwargs
    ) -> tuple[torch.FloatTensor, torch.LongTensor]:
        """
        Forward pass for single-turn instructional data (no multi-turn conversation required).

        This method is used for instruction-following tasks (SFT), where the input
        consists of a single instruction and the corresponding labels. It directly
        delegates to the single-turn forward method `_forward`.

        Args:
            input_ids (torch.Tensor): Tensor of shape (batch_size, seq_len) containing input token IDs.
            attention_mask (torch.Tensor): Tensor indicating padding positions.
            labels (torch.Tensor): Tensor containing the target labels for supervised fine-tuning.
            **kwargs: Additional keyword arguments passed to `_forward`.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: 
                - logits: The output logits from the model for each input token.
                - labels: The same as input labels, used for loss computation.
        """
        logits = self._forward(input_ids, attention_mask, labels, **kwargs)
        # For Instruction SFT, labels remain the same as input
        return logits, labels

    def _conversational_forward(
        self, 
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,   
        **kwargs
    ) -> tuple[torch.FloatTensor, torch.LongTensor]:
        """
        Forward pass for conversational (multi-turn) data.

        Multi-turn forward is constructed by sequentially calling the single-turn forward
        for each conversation turn. Latents inserted in turn i-1 are not visible to turn i.

        Args:
            input_ids (torch.Tensor): Input token IDs, shape (1, seq_len). Batch size must be 1.
            attention_mask (torch.Tensor): Attention mask for input tokens.
            labels (torch.Tensor): Target labels for supervised fine-tuning (-100 for ignore positions).
            **kwargs: Additional arguments passed to `_forward`.

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                - all_logits: Logits for the entire sequence, with zeros for unsupervised positions.
                - all_labels: Labels for the entire sequence, with -100 for unsupervised positions.
        """
        assert input_ids.shape[0] == 1, "Conversational SFT currently only supports batch_size = 1"
        seq_len = input_ids.shape[1]
        vocab_size = self.config.vocab_size
        device = input_ids.device

        # Identify single-turn segments within the conversation based on labels
        label_row = labels[0]
        should_supervise = label_row != -100
        if not should_supervise.any():
            raise ValueError("At least one completion segment is required")

        # Compute the start and end indices of valid supervised segments
        valid_mask = should_supervise.int()
        diff = torch.diff(torch.cat([torch.tensor([0], device=device), valid_mask]))
        valid_starts = (diff == 1).nonzero(as_tuple=True)[0].tolist()  # Transition 0 -> 1
        ends = (diff == -1).nonzero(as_tuple=True)[0].tolist()          # Transition 1 -> 0
        if len(ends) < len(valid_starts):
            ends.append(seq_len)
        assert len(valid_starts) == len(ends)
        
        # Build triplets (start of previous segment, start of supervised segment, end of supervised segment)
        triplets = []
        start = 0
        for s, e in zip(valid_starts, ends):
            triplets.append((start, s, e))
            start = e
        
        # If there are more segments than allowed, randomly select self.max_prompt_aug_num segments
        if len(triplets) <= self.config.max_prompt_aug_num:
            select_turns = [1] * len(triplets)
        else:
            triplets_num = len(triplets)
            selected_indices = set(random.sample(range(triplets_num), self.config.max_prompt_aug_num))
            select_turns = [1 if i in selected_indices else 0 for i in range(triplets_num)]

        # Initialize tensors to store logits and labels for the entire sequence
        all_logits = torch.zeros(1, seq_len, vocab_size, device=device)
        all_labels = torch.full((1, seq_len), -100, device=device)

        # Loop over each conversation turn and perform single-turn forward if supervised
        for triplet, should_supervise in zip(triplets, select_turns):
            start, valid_start, end = triplet
            if should_supervise:
                cur_input_ids = input_ids[0, :end].unsqueeze(0)
                cur_attention = attention_mask[0, :end].unsqueeze(0)
                # cur_labels only used for _forward, does not represent the true supervision range
                cur_labels = labels[0, :end].clone().unsqueeze(0)
                cur_labels[0, :valid_start] = -100  # Mask tokens before supervision start

                # Single-turn forward for the current conversation segment
                logits = self._forward(cur_input_ids, cur_attention, cur_labels, **kwargs)
                
                # Update overall logits and labels with the results of this segment
                all_logits[0, start:end, :] = logits[0, start:end, :]
                all_labels[0, start:end] = labels[0, start:end]

        # Return logits and labels:
        # - supervised positions retain computed logits and original labels
        # - unsupervised positions have logits = 0 and labels = -100
        return all_logits, all_labels

    def forward(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        **kwargs
    ) -> MemGenOutputWithPast:  
        tokenizer = self.tokenizer

        # Ensure labels are provided, required for training the reasoning processor
        assert labels is not None, "Reasoning Processor requires input labels for training"
        
        # Determine whether the input is single-turn (instruction) or multi-turn (conversation)
        forward_func = self._instructional_forward
        # DISABLED: Force use of instructional forward for GRPO training
        # The conversational forward expects specific label patterns that GRPO doesn't provide
        # if self._is_conversation(input_ids, tokenizer):
        #     # For conversational data, mask assistant tokens in labels
        #     labels = self._postprocess_assistant_labels(input_ids, labels, tokenizer)
        #     forward_func = self._conversational_forward
        
        batch_size = 1  # Currently process one sequence per batch
        iter_num = input_ids.size(0) // batch_size

        # Forward pass per batch
        logits, supervised_labels = [], []
        for i in range(iter_num):
            batch_input_ids = input_ids[i * batch_size: (i + 1) * batch_size]
            batch_attention_mask = attention_mask[i * batch_size: (i + 1) * batch_size]
            batch_labels = labels[i * batch_size: (i + 1) * batch_size]

            # Call the appropriate forward function (instruction or conversation)
            batch_logits, batch_supervised_labels = forward_func(
                input_ids=batch_input_ids,
                attention_mask=batch_attention_mask,
                labels=batch_labels,
                **kwargs
            )
            logits.append(batch_logits)
            supervised_labels.append(batch_supervised_labels)
        
        # Concatenate results from all batches
        all_logits = torch.concat(logits, dim=0)
        all_labels = torch.concat(supervised_labels, dim=0)

        # Compute causal language modeling loss (shifted by one)
        shift_logits = all_logits[..., :-1, :].contiguous()
        shift_labels = all_labels[..., 1:].contiguous()
        # assert shift_logits.shape[:-1] == shift_labels.shape
        loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        # Return model outputs
        outputs = MemGenOutputWithPast(loss=loss, logits=all_logits)
        outputs.supervised_labels = all_labels  # Positions in input_ids that are supervised
        return outputs

    @torch.no_grad()
    def generate(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: torch.Tensor,
        generation_config: GenerationConfig = None, 
        return_augmentation_mask: bool = False,
        memory_texts: Optional[List[Optional[str]]] = None,
        **kwargs
    ) -> Union[torch.LongTensor, tuple[torch.LongTensor, torch.LongTensor]]: 
        
        tokenizer = self.tokenizer
        reasoner = self.reasoner
        weaver = self.weaver
        max_augment_num = self.config.max_inference_aug_num
        invalid_token_id = -100

        # preproecess inputs
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        max_new_tokens = generation_config.max_new_tokens
        pad_token_id = tokenizer.pad_token_id
        eos_token_id = tokenizer.eos_token_id
        prompt_len = input_ids.size(1)

        inputs_embeds = reasoner.get_input_embeddings()(input_ids)
        B, _, hidden_size = inputs_embeds.shape
        device = inputs_embeds.device

        memory_available = None
        memory_weaver_embeds: Optional[List[Optional[torch.Tensor]]] = None
        if memory_texts is not None:
            if len(memory_texts) != B:
                raise ValueError(
                    f"memory_texts length ({len(memory_texts)}) does not match batch size ({B})"
                )
            memory_available = torch.zeros(B, dtype=torch.bool, device=device)
            memory_weaver_embeds = [None] * B
            max_memory_tokens = getattr(self.config, "memory_max_length", 128)
            for idx, text in enumerate(memory_texts):
                if not text:
                    continue
                mem_inputs = tokenizer(
                    text,
                    return_tensors="pt",
                    add_special_tokens=False,
                    truncation=True,
                    max_length=max_memory_tokens,
                )
                mem_ids = mem_inputs["input_ids"].to(device)
                mem_embeds = reasoner.get_input_embeddings()(mem_ids)
                mem_embeds = self.reasoner_to_weaver(mem_embeds).squeeze(0)
                if mem_embeds.numel() == 0:
                    continue
                memory_weaver_embeds[idx] = mem_embeds
                memory_available[idx] = True
        
        # --- generation loop ---
        current_inputs_embeds = inputs_embeds
        current_attention_mask = attention_mask
        current_position_ids = self._generate_position_ids(current_attention_mask)
        current_input_ids = input_ids
        current_cache: DynamicCache = None

        # Generation Loop Initialization
        sentence_augment_count = torch.zeros(B, dtype=torch.int, device=device)
        
        # NOTE - Whether to call the trigger and insert latent memory before generating the token at this position
        # - augmentation_pos[b][i] == -100: For the b-th sequence, no augmentation was sampled before generating the i-th token
        # - augmentation_pos[b][i] == 0: For the b-th sequence, augmentation was sampled before generating the i-th token, but the trigger decided NOT to insert latent memory
        # - augmentation_pos[b][i] == 1: For the b-th sequence, augmentation was sampled before generating the i-th token, and the trigger decided to insert latent memory
        augmentation_pos = torch.full((B, max_new_tokens), fill_value=invalid_token_id, device=device) 

        for i in range(max_new_tokens):
            
            assert current_inputs_embeds.shape[:2] == current_attention_mask.shape == current_position_ids.shape
            augment_decision = self._should_augment(
                current_input_ids, 
                sentence_augment_count=sentence_augment_count, 
                do_sample=generation_config.do_sample,
                temperature=generation_config.temperature,
                is_prompt=(i==0)  
            )
            if memory_available is not None:
                no_memory_mask = ~memory_available
                if no_memory_mask.any():
                    augment_decision = augment_decision.clone()
                    aug_mask = no_memory_mask & (augment_decision == 1)
                    augment_decision[aug_mask] = 0
            augmentation_pos[:, i] = augment_decision
            augment_indices = torch.where(augment_decision == 1)[0]

            # If there are sentences to augment, apply augmentation; others remain with left padding
            if len(augment_indices) > 0:
                # Increment the augmentation count for sentences that are being augmented
                if i != 0:  
                    sentence_augment_count[augment_indices] += 1

                # Select embeddings, attention masks, and position IDs for sentences to be augmented
                candidate_inputs_embeds = current_inputs_embeds[augment_indices]
                candidate_attention_mask = current_attention_mask[augment_indices]

                # Perform inference augmentation using the weaver
                weaver_inputs_embeds = self.reasoner_to_weaver(candidate_inputs_embeds)
                weaver_attention_mask = candidate_attention_mask
                if memory_weaver_embeds is not None:
                    mem_lengths = []
                    for idx in augment_indices.tolist():
                        mem = memory_weaver_embeds[idx]
                        mem_lengths.append(mem.size(0) if mem is not None else 0)
                    max_mem_len = max(mem_lengths) if mem_lengths else 0
                    if max_mem_len > 0:
                        mem_tensor = torch.zeros(
                            (len(augment_indices), max_mem_len, weaver_inputs_embeds.size(-1)),
                            device=device,
                            dtype=weaver_inputs_embeds.dtype,
                        )
                        mem_mask = torch.zeros(
                            (len(augment_indices), max_mem_len),
                            device=device,
                            dtype=weaver_attention_mask.dtype,
                        )
                        for row, idx in enumerate(augment_indices.tolist()):
                            mem = memory_weaver_embeds[idx]
                            if mem is None:
                                continue
                            mem_len = mem.size(0)
                            mem_tensor[row, :mem_len, :] = mem
                            mem_mask[row, :mem_len] = 1
                        weaver_inputs_embeds = torch.cat([weaver_inputs_embeds, mem_tensor], dim=1)
                        weaver_attention_mask = torch.cat([weaver_attention_mask, mem_mask], dim=1)

                weaver_position_ids = self._generate_position_ids(weaver_attention_mask)
                if i == 0:
                    weaver_hidden_states, attn_mask, _ = weaver.augment_prompt(
                        weaver_inputs_embeds, weaver_attention_mask, weaver_position_ids
                    )                    
                else:
                    weaver_hidden_states, attn_mask, _ = weaver.augment_inference(
                        weaver_inputs_embeds, weaver_attention_mask, weaver_position_ids
                    )
                latent_inputs_embeds = self.weaver_to_reasoner(weaver_hidden_states)

                candidate_inputs_embeds = torch.cat([candidate_inputs_embeds, latent_inputs_embeds], dim=1)
                candidate_attention_mask = torch.cat([candidate_attention_mask, attn_mask], dim=1)
                
                # Create a single merged tensor for all sequences
                new_len = candidate_inputs_embeds.size(1)
                merged_inputs_embeds = torch.zeros((B, new_len, hidden_size), device=device, dtype=current_inputs_embeds.dtype)
                merged_attention_mask = torch.zeros((B, new_len), device=device, dtype=current_attention_mask.dtype)
                   
                # Directly place augmented and non-augmented sequences
                merged_inputs_embeds[augment_indices] = candidate_inputs_embeds
                merged_attention_mask[augment_indices] = candidate_attention_mask
                
                # Non-augmented sequences now include both -100 and 0
                non_augment_indices = torch.where(augment_decision != 1)[0]
                if len(non_augment_indices) > 0:
                    # dynamic left padding
                    non_aug_inputs_embeds = current_inputs_embeds[non_augment_indices]
                    non_aug_attention_mask = current_attention_mask[non_augment_indices]
                    pad_len = weaver.prompt_latents_num if i == 0 else weaver.inference_latents_num
                    non_aug_inputs_embeds, non_aug_attention_mask, _ = self._left_pad(
                        non_aug_inputs_embeds, non_aug_attention_mask, None, pad_len
                    )
                    
                    merged_inputs_embeds[non_augment_indices] = non_aug_inputs_embeds
                    merged_attention_mask[non_augment_indices] = non_aug_attention_mask
                
                current_inputs_embeds = merged_inputs_embeds
                current_attention_mask = merged_attention_mask
                current_position_ids = self._generate_position_ids(current_attention_mask)
                current_cache = None  

            # Check if all sequences have reached the maximum number of augmentations
            if (sentence_augment_count >= max_augment_num).all():
                # Adjust the remaining generation length
                generation_config = GenerationConfig(
                    do_sample=False,
                    pad_token_id=pad_token_id,
                    eos_token_id=eos_token_id,
                    use_cache=False,
                    max_new_tokens=max_new_tokens-i
                )
                # Perform generation for the remaining tokens using the reasoner
                generated = reasoner.generate(
                    inputs_embeds=current_inputs_embeds,
                    attention_mask=current_attention_mask,
                    generation_config=generation_config
                )
                current_input_ids = torch.cat([current_input_ids, generated], dim=1)
                break            

            if current_cache is not None:
                assert current_inputs_embeds.size(1) == current_cache.get_seq_length() + 1
                reasoner_inputs_embeds = current_inputs_embeds[:, -1:]
                reasoner_position_ids = current_position_ids[:, -1:]
            else:
                reasoner_inputs_embeds = current_inputs_embeds
                reasoner_position_ids = current_position_ids

            outputs = reasoner(
                inputs_embeds=reasoner_inputs_embeds,
                attention_mask=current_attention_mask,
                position_ids=reasoner_position_ids,
                output_hidden_states=False,
                use_cache=True,
                past_key_values=current_cache
            )
            current_inputs_embeds, current_attention_mask, current_position_ids, current_input_ids = self._append_one_step(
                outputs, 
                current_inputs_embeds, 
                current_attention_mask, 
                current_position_ids, 
                current_input_ids, 
                do_sample=False, 
                temperature=0.0
            )
            current_cache = outputs.past_key_values

            # If all sequences in the batch have already generated an EOS token, stop early
            if (current_input_ids[:, -1] == eos_token_id).all():
                break  

            # This is needed to properly delete outputs.logits which may be very large for first iteration
            # Otherwise a reference to outputs is kept which keeps the logits alive in the next iteration
            del outputs

        # postprocess
        new_generated_len = current_input_ids.size(1) - prompt_len
        augmentation_pos = augmentation_pos[:, :new_generated_len]
        
        self._check_generate(
            current_input_ids[:, prompt_len:],
            augmentation_pos
        )
        
        if return_augmentation_mask:
            return (current_input_ids, augmentation_pos)
        else:
            return current_input_ids

    def save_pretrained(
        self,
        save_directory: str,
        is_main_process: bool = True,
        state_dict: Optional[dict] = None,
        save_function = None,
        push_to_hub: bool = False,
        max_shard_size: Union[int, str] = "5GB",
        safe_serialization: bool = True,
        variant: Optional[str] = None,
        token: Optional[Union[str, bool]] = None,
        save_peft_format: bool = True,
        **kwargs,
    ):
        """
        Custom save_pretrained to handle shared tensors between reasoner, weaver, and trigger.

        Only saves trainable parameters (LoRA weights and projection layers) to avoid
        the shared tensor issue.
        """
        import os
        from safetensors.torch import save_file

        if os.path.isfile(save_directory):
            logging.error(f"Provided path ({save_directory}) should be a directory, not a file")
            return

        os.makedirs(save_directory, exist_ok=True)

        # Save config
        self.config.save_pretrained(save_directory)

        # Collect only trainable parameters (LoRA weights and projection layers)
        state_dict_to_save = {}

        # Save projection layers (always trainable)
        state_dict_to_save['reasoner_to_weaver.weight'] = self.reasoner_to_weaver.weight.detach().clone().cpu()
        state_dict_to_save['reasoner_to_weaver.bias'] = self.reasoner_to_weaver.bias.detach().clone().cpu()
        state_dict_to_save['weaver_to_reasoner.weight'] = self.weaver_to_reasoner.weight.detach().clone().cpu()
        state_dict_to_save['weaver_to_reasoner.bias'] = self.weaver_to_reasoner.bias.detach().clone().cpu()

        # Save weaver LoRA weights (if trainable)
        # Use a set to track saved parameter IDs to avoid duplicates
        saved_param_ids = set()

        for name, param in self.weaver.named_parameters():
            if param.requires_grad and 'lora' in name.lower():
                param_id = id(param)
                if param_id not in saved_param_ids:
                    state_dict_to_save[f'weaver.{name}'] = param.detach().clone().cpu()
                    saved_param_ids.add(param_id)

        # Save trigger LoRA weights (if trainable and not already saved)
        for name, param in self.trigger.named_parameters():
            if param.requires_grad and 'lora' in name.lower():
                param_id = id(param)
                if param_id not in saved_param_ids:
                    state_dict_to_save[f'trigger.{name}'] = param.detach().clone().cpu()
                    saved_param_ids.add(param_id)

        # Save using safetensors
        if safe_serialization:
            save_file(state_dict_to_save, os.path.join(save_directory, "model.safetensors"))
            logging.info(f"Model weights saved to {os.path.join(save_directory, 'model.safetensors')}")
        else:
            torch.save(state_dict_to_save, os.path.join(save_directory, "pytorch_model.bin"))
            logging.info(f"Model weights saved to {os.path.join(save_directory, 'pytorch_model.bin')}")

        # Save tokenizer
        if hasattr(self, 'tokenizer') and self.tokenizer is not None:
            self.tokenizer.save_pretrained(save_directory)
            logging.info(f"Tokenizer saved to {save_directory}")

        logging.info(f"Saved {len(state_dict_to_save)} trainable parameters to {save_directory}")

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        config: MemGenConfig = None,
        base_model: PreTrainedModel = None,
        base_tokenizer = None,
        weaver_model: PreTrainedModel = None,
        trigger_model: PreTrainedModel = None,
        **kwargs,
    ):
        """
        Load a MemGenModel from a pretrained checkpoint.

        This method loads only the trainable parameters (LoRA weights and projection layers)
        that were saved by save_pretrained.
        """
        import os
        from safetensors.torch import load_file

        # Initialize the model with base models
        model = cls(
            config=config,
            base_model=base_model,
            base_tokenizer=base_tokenizer,
            weaver_model=weaver_model,
            trigger_model=trigger_model
        )

        # Load saved weights
        safetensors_path = os.path.join(pretrained_model_name_or_path, "model.safetensors")
        pytorch_path = os.path.join(pretrained_model_name_or_path, "pytorch_model.bin")

        if os.path.exists(safetensors_path):
            state_dict = load_file(safetensors_path)
            logging.info(f"Loading weights from {safetensors_path}")
        elif os.path.exists(pytorch_path):
            state_dict = torch.load(pytorch_path, map_location='cpu')
            logging.info(f"Loading weights from {pytorch_path}")
        else:
            logging.warning(f"No saved weights found in {pretrained_model_name_or_path}, using initialized weights")
            return model

        # Load projection layers
        if 'reasoner_to_weaver.weight' in state_dict:
            model.reasoner_to_weaver.weight.data = state_dict['reasoner_to_weaver.weight'].to(model.device)
            model.reasoner_to_weaver.bias.data = state_dict['reasoner_to_weaver.bias'].to(model.device)
            model.weaver_to_reasoner.weight.data = state_dict['weaver_to_reasoner.weight'].to(model.device)
            model.weaver_to_reasoner.bias.data = state_dict['weaver_to_reasoner.bias'].to(model.device)

        # Load weaver LoRA weights
        weaver_params = {k.replace('weaver.', ''): v for k, v in state_dict.items() if k.startswith('weaver.')}
        if weaver_params:
            model.weaver.load_state_dict(weaver_params, strict=False)
            logging.info(f"Loaded {len(weaver_params)} weaver parameters")

        # Load trigger LoRA weights
        trigger_params = {k.replace('trigger.', ''): v for k, v in state_dict.items() if k.startswith('trigger.')}
        if trigger_params:
            model.trigger.load_state_dict(trigger_params, strict=False)
            logging.info(f"Loaded {len(trigger_params)} trigger parameters")

        logging.info(f"Successfully loaded model from {pretrained_model_name_or_path}")
        return model

    @classmethod
    def from_config(cls, config_dict: dict):
        # base LLM
        model_name = config_dict.get("model_name")

        # max augment numbers
        max_prompt_aug_num = config_dict.get("max_prompt_aug_num", 1)
        max_inference_aug_num = config_dict.get("max_inference_aug_num", 5)

        # Weaver configs
        weaver_config = config_dict.get("weaver", {})
        prompt_latents_len = weaver_config.get("prompt_latents_len", 8)
        inference_latents_len = weaver_config.get("inference_latents_len", 8)
        weaver_lora_config_dict = weaver_config.get("lora_config", None)
        weaver_model_name = weaver_config.get("model_name", None)

        # Trigger configs
        trigger_config = config_dict.get("trigger", {})
        trigger_active = trigger_config.get("active", False)
        trigger_lora_config_dict = trigger_config.get("lora_config", None)
        trigger_model_name = trigger_config.get("model_name", None)

        # 构造 MemGenConfig
        from transformers import AutoConfig
        base_config = AutoConfig.from_pretrained(model_name)
        memgen_config = MemGenConfig.from_pretrained(
            model_name,
            max_prompt_aug_num=max_prompt_aug_num,
            max_inference_aug_num=max_inference_aug_num,
            # weaver
            prompt_latents_len=prompt_latents_len,
            inference_latents_len=inference_latents_len,
            weaver_lora_config=weaver_lora_config_dict,
            # trigger
            trigger_active=trigger_active,
            trigger_lora_config=trigger_lora_config_dict
        )

        # Ensure _name_or_path is set for GRPO compatibility
        if not memgen_config._name_or_path:
            memgen_config._name_or_path = model_name
        
        base_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32, attn_implementation="eager")
        base_tokenizer = AutoTokenizer.from_pretrained(model_name)

        if weaver_model_name != model_name:
            weaver_model = AutoModelForCausalLM.from_pretrained(weaver_model_name, torch_dtype=torch.float32, attn_implementation="eager")
        else:
            weaver_model = base_model

        if trigger_model_name != model_name:
            trigger_model = AutoModelForCausalLM.from_pretrained(trigger_model_name, torch_dtype=torch.float32, attn_implementation="eager")
        else:
            trigger_model = base_model
        
        load_model_path = config_dict.get("load_model_path", None)
        if not load_model_path:
            model = cls(
                config=memgen_config, 
                base_model=base_model, 
                base_tokenizer=base_tokenizer,
                weaver_model=weaver_model,
                trigger_model=trigger_model
            )
        else:
            model = cls.from_pretrained(
                load_model_path, 
                config=memgen_config,
                base_model=base_model,
                base_tokenizer=base_tokenizer,
                weaver_model=weaver_model,
                trigger_model=trigger_model
            )
        
        return model
