#!/usr/bin/env python
"""
Simple evaluation script for Self-Evolving RAG
Compares baseline (no memory) vs memory-enhanced performance
"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm

# Setup environment
os.environ['HF_HOME'] = '/root/autodl-tmp/models'
os.environ['TRANSFORMERS_CACHE'] = '/root/autodl-tmp/models'
os.environ['HF_DATASETS_CACHE'] = '/root/autodl-tmp/datasets'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import torch
from datasets import load_dataset
from omegaconf import OmegaConf
from memgen.model import MemGenModel
from memgen.memory.store import ExperienceStore
from transformers import GenerationConfig


def normalize_answer(s: str) -> str:
    """Normalize answer for comparison"""
    import re
    import string

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def compute_exact_match(prediction: str, ground_truths: List[str]) -> float:
    """Compute Exact Match score"""
    normalized_pred = normalize_answer(prediction)
    for gt in ground_truths:
        if normalize_answer(gt) == normalized_pred:
            return 1.0
    return 0.0


def compute_f1(prediction: str, ground_truths: List[str]) -> float:
    """Compute F1 score"""
    normalized_pred = normalize_answer(prediction)
    pred_tokens = normalized_pred.split()

    max_f1 = 0.0
    for gt in ground_truths:
        normalized_gt = normalize_answer(gt)
        gt_tokens = normalized_gt.split()

        common = set(pred_tokens) & set(gt_tokens)
        num_common = len(common)

        if num_common == 0:
            continue

        precision = num_common / len(pred_tokens) if pred_tokens else 0
        recall = num_common / len(gt_tokens) if gt_tokens else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        max_f1 = max(max_f1, f1)

    return max_f1


def evaluate(
    model: MemGenModel,
    dataset,
    experience_store: ExperienceStore = None,
    num_samples: int = 50,
    batch_size: int = 4,
    device: str = "cuda"
):
    """
    Evaluate model performance

    Args:
        model: MemGenModel instance
        dataset: TriviaQA dataset
        experience_store: Optional experience store for memory retrieval
        num_samples: Number of samples to evaluate
        batch_size: Batch size
        device: Device
    """

    use_memory = experience_store is not None
    mode_name = "Memory-Enhanced" if use_memory else "Baseline"

    print(f"\n{'='*80}")
    print(f"{mode_name} Evaluation")
    print(f"{'='*80}")
    print(f"  - Samples: {num_samples}")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Memory: {'ENABLED' if use_memory else 'DISABLED'}")
    print(f"{'='*80}\n")

    # Generation config
    gen_config = GenerationConfig(
        max_new_tokens=64,
        do_sample=False,
        pad_token_id=model.tokenizer.pad_token_id,
        eos_token_id=model.tokenizer.eos_token_id
    )

    model.eval()

    # Statistics
    total_em = 0
    total_f1 = 0.0
    total_processed = 0

    # Process in batches
    num_batches = (num_samples + batch_size - 1) // batch_size

    with torch.no_grad():
        for batch_idx in tqdm(range(num_batches), desc=f"Evaluating {mode_name}"):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, num_samples)

            if start_idx >= len(dataset):
                break

            batch_data = dataset[start_idx:end_idx]

            # Prepare prompts
            prompts = []
            memory_texts_batch = []

            for question in batch_data['question']:
                prompt = f"Question: {question}\\nAnswer:"
                prompts.append(prompt)

                # Retrieve memory if enabled
                if use_memory:
                    memories = experience_store.search(question, topk=1)
                    if memories:
                        memory_text = memories[0]['entry']['answer']  # Use top-1 memory
                        memory_texts_batch.append(memory_text)
                    else:
                        memory_texts_batch.append(None)
                else:
                    memory_texts_batch.append(None)

            # Tokenize
            inputs = model.tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # Generate
            outputs = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                generation_config=gen_config,
                memory_texts=memory_texts_batch if use_memory else None
            )

            # Decode responses
            responses = model.tokenizer.batch_decode(
                outputs[:, inputs["input_ids"].size(1):],
                skip_special_tokens=True
            )

            # Evaluate
            for i, (response, answer_dict) in enumerate(zip(responses, batch_data['answer'])):
                answers = answer_dict['normalized_aliases']

                em = compute_exact_match(response, answers)
                f1 = compute_f1(response, answers)

                total_em += em
                total_f1 += f1
                total_processed += 1

    # Compute averages
    avg_em = total_em / total_processed if total_processed > 0 else 0
    avg_f1 = total_f1 / total_processed if total_processed > 0 else 0

    # Print results
    print(f"\n{'='*80}")
    print(f"{mode_name} Results")
    print(f"{'='*80}")
    print(f"  - Total samples: {total_processed}")
    print(f"  - Average EM: {avg_em:.4f}")
    print(f"  - Average F1: {avg_f1:.4f}")
    print(f"{'='*80}\n")

    return {
        "mode": mode_name,
        "total_samples": total_processed,
        "avg_em": avg_em,
        "avg_f1": avg_f1
    }


def main():
    parser = argparse.ArgumentParser(description="Simple evaluation for Self-Evolving RAG")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/latent_memory/phase0_coldstart.yaml",
        help="Config file path"
    )
    parser.add_argument(
        "--experience-store",
        type=str,
        default="/root/autodl-tmp/phase0_experience.jsonl",
        help="Experience store path"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=50,
        help="Number of samples to evaluate"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size"
    )

    args = parser.parse_args()

    # Load config
    print(f"Loading config from: {args.config}")
    config = OmegaConf.load(args.config)
    config_dict = OmegaConf.to_container(config, resolve=True)

    # Load model
    print(f"Loading model...")
    model = MemGenModel.from_config(config_dict.get("model"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    print(f"✓ Model loaded on {device}")

    # Load dataset
    print(f"Loading TriviaQA dataset...")
    dataset = load_dataset(
        'trivia_qa',
        'rc.nocontext',
        split='validation',  # Use validation set for evaluation
        cache_dir='/root/autodl-tmp/datasets'
    )
    print(f"✓ Dataset loaded: {len(dataset)} samples")

    # Evaluate baseline (no memory)
    baseline_results = evaluate(
        model=model,
        dataset=dataset,
        experience_store=None,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        device=device
    )

    # Load experience store
    print(f"Loading experience store from: {args.experience_store}")
    experience_store = ExperienceStore(
        store_path=args.experience_store,
        index_type="simple",
        min_score=0.2
    )
    print(f"✓ Experience store loaded: {len(experience_store.entries)} experiences")

    # Evaluate with memory
    memory_results = evaluate(
        model=model,
        dataset=dataset,
        experience_store=experience_store,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        device=device
    )

    # Compare results
    print(f"\n{'='*80}")
    print(f"COMPARISON SUMMARY")
    print(f"{'='*80}")
    print(f"Metric          Baseline    Memory      Improvement")
    print(f"{'-'*80}")

    em_improvement = ((memory_results['avg_em'] - baseline_results['avg_em']) / baseline_results['avg_em'] * 100) if baseline_results['avg_em'] > 0 else 0
    f1_improvement = ((memory_results['avg_f1'] - baseline_results['avg_f1']) / baseline_results['avg_f1'] * 100) if baseline_results['avg_f1'] > 0 else 0

    print(f"EM              {baseline_results['avg_em']:.4f}      {memory_results['avg_em']:.4f}      {em_improvement:+.2f}%")
    print(f"F1              {baseline_results['avg_f1']:.4f}      {memory_results['avg_f1']:.4f}      {f1_improvement:+.2f}%")
    print(f"{'='*80}\n")

    # Save results
    results = {
        "baseline": baseline_results,
        "memory": memory_results,
        "improvement": {
            "em": em_improvement,
            "f1": f1_improvement
        }
    }

    output_path = "/root/autodl-tmp/evaluation_results.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✓ Results saved to: {output_path}")


if __name__ == "__main__":
    main()
