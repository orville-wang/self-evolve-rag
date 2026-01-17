#!/usr/bin/env python
"""
Trigger 训练数据标注脚本

实现三种标注策略:
1. Phase A: 启发式监督（基于检索得分和任务类型）
2. Phase B: 对比学习（对比有无 memory 的效果）
3. Phase C: GRPO 在线学习（直接优化任务目标）

本脚本实现 Phase A 和 Phase B
"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import List, Dict, Tuple
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
    """标准化答案用于比较"""
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


def compute_f1(prediction: str, ground_truths: List[str]) -> float:
    """计算 F1 分数"""
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


def label_trigger_heuristic(
    query: str,
    memory_store: ExperienceStore,
    task_type: str = "open_qa",
    high_score_threshold: float = 0.7,
    low_score_threshold: float = 0.3
) -> Tuple[int, Dict]:
    """
    Phase A: 启发式标注

    标注规则:
    - 检索得分 >= 0.7: INVOKE (1)
    - 检索得分 < 0.3: SKIP (0)
    - 0.3 <= 检索得分 < 0.7: 根据任务类型判断

    Returns:
        label: 0 (SKIP) or 1 (INVOKE)
        metadata: 标注元数据
    """

    # 检索 memory
    memories = memory_store.search(query, topk=1)

    if len(memories) == 0:
        return 0, {"reason": "no_memory", "score": 0.0}

    score = memories[0]['score']
    memory_task_type = memories[0].get('entry', {}).get('task_type', 'unknown')

    # 高相似度，直接 INVOKE
    if score >= high_score_threshold:
        return 1, {"reason": "high_score", "score": score}

    # 低相似度，直接 SKIP
    if score < low_score_threshold:
        return 0, {"reason": "low_score", "score": score}

    # 中等相似度，根据任务类型判断
    if task_type == memory_task_type:
        return 1, {"reason": "same_task_type", "score": score, "task_type": task_type}
    else:
        return 0, {"reason": "different_task_type", "score": score, "task_type": task_type}


def label_trigger_contrastive(
    model: MemGenModel,
    query: str,
    ground_truths: List[str],
    memory_text: str,
    gen_config: GenerationConfig,
    device: str,
    improvement_threshold: float = 0.1
) -> Tuple[int, Dict]:
    """
    Phase B: 对比学习标注

    对比有无 memory 的生成效果，如果 memory 带来显著提升则标注为 INVOKE

    Returns:
        label: 0 (SKIP) or 1 (INVOKE)
        metadata: 标注元数据
    """

    prompt = f"Question: {query}\nAnswer:"
    inputs = model.tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        # Generate without memory
        outputs_no_mem = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            generation_config=gen_config,
            memory_texts=None
        )
        response_no_mem = model.tokenizer.decode(
            outputs_no_mem[0, inputs["input_ids"].size(1):],
            skip_special_tokens=True
        )
        reward_no_mem = compute_f1(response_no_mem, ground_truths)

        # Generate with memory
        outputs_with_mem = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            generation_config=gen_config,
            memory_texts=[memory_text]
        )
        response_with_mem = model.tokenizer.decode(
            outputs_with_mem[0, inputs["input_ids"].size(1):],
            skip_special_tokens=True
        )
        reward_with_mem = compute_f1(response_with_mem, ground_truths)

    # Compute improvement
    delta_reward = reward_with_mem - reward_no_mem

    # Label based on improvement
    if delta_reward > improvement_threshold:
        label = 1  # INVOKE (memory helps)
    elif delta_reward < -0.05:
        label = 0  # SKIP (memory hurts)
    else:
        label = 0  # SKIP (memory has no effect)

    metadata = {
        "reward_no_mem": reward_no_mem,
        "reward_with_mem": reward_with_mem,
        "delta_reward": delta_reward,
        "response_no_mem": response_no_mem,
        "response_with_mem": response_with_mem
    }

    return label, metadata


def generate_trigger_labels_heuristic(
    memory_store: ExperienceStore,
    dataset,
    output_path: str,
    num_samples: int = 1000,
    task_type: str = "open_qa"
):
    """
    使用启发式方法生成 Trigger 标注数据
    """

    print(f"\n{'='*80}")
    print(f"Phase A: 启发式 Trigger 标注")
    print(f"{'='*80}")
    print(f"  - 样本数: {num_samples}")
    print(f"  - 输出路径: {output_path}")
    print(f"{'='*80}\n")

    labeled_data = []
    label_stats = {"INVOKE": 0, "SKIP": 0}
    reason_stats = {}

    for i in tqdm(range(min(num_samples, len(dataset))), desc="Labeling"):
        sample = dataset[i]
        query = sample['question']

        # Label
        label, metadata = label_trigger_heuristic(
            query=query,
            memory_store=memory_store,
            task_type=task_type
        )

        # Statistics
        label_name = "INVOKE" if label == 1 else "SKIP"
        label_stats[label_name] += 1
        reason = metadata.get('reason', 'unknown')
        reason_stats[reason] = reason_stats.get(reason, 0) + 1

        # Save
        labeled_data.append({
            "query": query,
            "label": label,
            "metadata": metadata
        })

    # Save to file
    with open(output_path, 'w') as f:
        for item in labeled_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    # Print statistics
    print(f"\n{'='*80}")
    print(f"标注完成")
    print(f"{'='*80}")
    print(f"  - 总样本: {len(labeled_data)}")
    print(f"  - INVOKE: {label_stats['INVOKE']} ({label_stats['INVOKE']/len(labeled_data)*100:.1f}%)")
    print(f"  - SKIP: {label_stats['SKIP']} ({label_stats['SKIP']/len(labeled_data)*100:.1f}%)")
    print(f"\n  标注原因分布:")
    for reason, count in sorted(reason_stats.items(), key=lambda x: -x[1]):
        print(f"    - {reason}: {count} ({count/len(labeled_data)*100:.1f}%)")
    print(f"{'='*80}\n")


def generate_trigger_labels_contrastive(
    model: MemGenModel,
    memory_store: ExperienceStore,
    dataset,
    output_path: str,
    num_samples: int = 100,  # 对比学习成本高，样本数少
    batch_size: int = 1,
    device: str = "cuda"
):
    """
    使用对比学习方法生成 Trigger 标注数据
    """

    print(f"\n{'='*80}")
    print(f"Phase B: 对比学习 Trigger 标注")
    print(f"{'='*80}")
    print(f"  - 样本数: {num_samples}")
    print(f"  - 输出路径: {output_path}")
    print(f"  - 警告: 对比学习需要生成两次，成本较高")
    print(f"{'='*80}\n")

    gen_config = GenerationConfig(
        max_new_tokens=64,
        do_sample=False,
        pad_token_id=model.tokenizer.pad_token_id,
        eos_token_id=model.tokenizer.eos_token_id
    )

    model.eval()

    labeled_data = []
    label_stats = {"INVOKE": 0, "SKIP": 0}

    for i in tqdm(range(min(num_samples, len(dataset))), desc="Labeling (contrastive)"):
        sample = dataset[i]
        query = sample['question']
        ground_truths = sample['answer']['normalized_aliases']

        # Retrieve memory
        memories = memory_store.search(query, topk=1)
        if len(memories) == 0:
            # No memory, skip
            labeled_data.append({
                "query": query,
                "label": 0,
                "metadata": {"reason": "no_memory"}
            })
            label_stats["SKIP"] += 1
            continue

        memory_text = memories[0]['text']

        # Label using contrastive method
        label, metadata = label_trigger_contrastive(
            model=model,
            query=query,
            ground_truths=ground_truths,
            memory_text=memory_text,
            gen_config=gen_config,
            device=device
        )

        # Statistics
        label_name = "INVOKE" if label == 1 else "SKIP"
        label_stats[label_name] += 1

        # Save
        labeled_data.append({
            "query": query,
            "label": label,
            "metadata": metadata
        })

    # Save to file
    with open(output_path, 'w') as f:
        for item in labeled_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    # Print statistics
    print(f"\n{'='*80}")
    print(f"标注完成")
    print(f"{'='*80}")
    print(f"  - 总样本: {len(labeled_data)}")
    print(f"  - INVOKE: {label_stats['INVOKE']} ({label_stats['INVOKE']/len(labeled_data)*100:.1f}%)")
    print(f"  - SKIP: {label_stats['SKIP']} ({label_stats['SKIP']/len(labeled_data)*100:.1f}%)")
    print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(description="Generate Trigger training labels")
    parser.add_argument(
        "--method",
        type=str,
        choices=["heuristic", "contrastive"],
        default="heuristic",
        help="Labeling method"
    )
    parser.add_argument(
        "--memory-store",
        type=str,
        default="/root/autodl-tmp/initial_experience.jsonl",
        help="Memory store path"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="/root/autodl-tmp/trigger_labels.jsonl",
        help="Output labels path"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1000,
        help="Number of samples to label"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/latent_memory/phase0_coldstart.yaml",
        help="Config file (only for contrastive method)"
    )

    args = parser.parse_args()

    # Load memory store
    print(f"Loading memory store from: {args.memory_store}")
    memory_store = ExperienceStore(
        store_path=args.memory_store,
        index_type="simple",
        min_score=0.2
    )
    print(f"✓ Memory store loaded: {len(memory_store.entries)} entries")

    # Load dataset
    print(f"Loading TriviaQA dataset...")
    dataset = load_dataset(
        'trivia_qa',
        'rc.nocontext',
        split='train',
        cache_dir='/root/autodl-tmp/datasets'
    )
    print(f"✓ Dataset loaded: {len(dataset)} samples")

    if args.method == "heuristic":
        # Phase A: Heuristic labeling
        generate_trigger_labels_heuristic(
            memory_store=memory_store,
            dataset=dataset,
            output_path=args.output,
            num_samples=args.num_samples
        )

    elif args.method == "contrastive":
        # Phase B: Contrastive labeling
        # Load model
        print(f"Loading model from config: {args.config}")
        config = OmegaConf.load(args.config)
        config_dict = OmegaConf.to_container(config, resolve=True)

        model = MemGenModel.from_config(config_dict.get("model"))
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        print(f"✓ Model loaded on {device}")

        generate_trigger_labels_contrastive(
            model=model,
            memory_store=memory_store,
            dataset=dataset,
            output_path=args.output,
            num_samples=args.num_samples,
            device=device
        )


if __name__ == "__main__":
    main()
