#!/usr/bin/env python
"""
Phase 0: 冷启动 - 生成初始经验库

使用 baseline 模型（不使用 memory）在 TriviaQA 数据集上生成答案，
收集高质量的 query-answer 对作为初始经验库。

目标: 生成 100-1000 条高质量经验
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
from memgen.memory.writer import MemoryWriter
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


def compute_exact_match(prediction: str, ground_truths: List[str]) -> float:
    """计算 Exact Match 分数"""
    normalized_pred = normalize_answer(prediction)
    for gt in ground_truths:
        if normalize_answer(gt) == normalized_pred:
            return 1.0
    return 0.0


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


def generate_initial_experiences(
    model: MemGenModel,
    dataset,
    output_path: str,
    num_samples: int = 1000,
    batch_size: int = 4,
    min_reward: float = 0.7,
    device: str = "cuda"
):
    """
    生成初始经验库

    Args:
        model: MemGenModel 实例
        dataset: TriviaQA 数据集
        output_path: 输出路径
        num_samples: 生成样本数
        batch_size: batch 大小
        min_reward: 最小 reward 阈值
        device: 设备
    """

    print(f"\n{'='*80}")
    print(f"Phase 0: 生成初始经验库")
    print(f"{'='*80}")
    print(f"  - 数据集: TriviaQA")
    print(f"  - 样本数: {num_samples}")
    print(f"  - Batch size: {batch_size}")
    print(f"  - 最小 reward: {min_reward}")
    print(f"  - 输出路径: {output_path}")
    print(f"{'='*80}\n")

    # Initialize memory writer
    writer = MemoryWriter(min_reward=min_reward, require_grounding=False)

    # Initialize experience store
    store = ExperienceStore(
        store_path=output_path,
        index_type="simple",
        min_score=0.2
    )

    # Generation config
    gen_config = GenerationConfig(
        max_new_tokens=64,
        do_sample=False,  # Use greedy decoding for baseline
        pad_token_id=model.tokenizer.pad_token_id,
        eos_token_id=model.tokenizer.eos_token_id
    )

    model.eval()

    # Statistics
    total_processed = 0
    total_high_quality = 0
    total_em = 0
    total_f1 = 0.0

    # Process in batches
    num_batches = (num_samples + batch_size - 1) // batch_size

    with torch.no_grad():
        for batch_idx in tqdm(range(num_batches), desc="Generating experiences"):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, num_samples)

            if start_idx >= len(dataset):
                break

            batch_data = dataset[start_idx:end_idx]

            # Prepare prompts
            prompts = []
            for question in batch_data['question']:
                prompt = f"Question: {question}\nAnswer:"
                prompts.append(prompt)

            # Tokenize
            inputs = model.tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # Generate (without memory for baseline)
            outputs = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                generation_config=gen_config,
                memory_texts=None  # No memory for baseline
            )

            # Decode responses
            responses = model.tokenizer.batch_decode(
                outputs[:, inputs["input_ids"].size(1):],
                skip_special_tokens=True
            )

            # Evaluate and collect experiences
            for i, (question, response, answer_dict) in enumerate(zip(
                batch_data['question'],
                responses,
                batch_data['answer']
            )):
                # Get normalized aliases from answer dict
                answers = answer_dict['normalized_aliases']

                # Compute reward
                em = compute_exact_match(response, answers)
                f1 = compute_f1(response, answers)
                reward = max(em, f1)  # Use max of EM and F1 as reward

                total_processed += 1
                total_em += em
                total_f1 += f1

                # Write back if high quality
                if writer.should_write(reward=reward, has_grounding=True):
                    entry = writer.create_entry(
                        query=question,
                        answer=response,
                        reward=reward,
                        task_type="open_qa",
                        context_snippets=answers[:3]  # Store ground truth as context
                    )
                    store.add(entry)
                    total_high_quality += 1

            # Save periodically
            if (batch_idx + 1) % 10 == 0:
                store.save()
                print(f"\n  [Checkpoint] Processed: {total_processed}, High-quality: {total_high_quality}")

    # Final save
    store.save()

    # Print statistics
    print(f"\n{'='*80}")
    print(f"Phase 0 完成")
    print(f"{'='*80}")
    print(f"  - 总处理样本: {total_processed}")
    print(f"  - 高质量经验: {total_high_quality} ({total_high_quality/total_processed*100:.1f}%)")
    print(f"  - 平均 EM: {total_em/total_processed:.3f}")
    print(f"  - 平均 F1: {total_f1/total_processed:.3f}")
    print(f"  - 经验库路径: {output_path}")
    print(f"{'='*80}\n")

    return {
        "total_processed": total_processed,
        "total_high_quality": total_high_quality,
        "avg_em": total_em / total_processed,
        "avg_f1": total_f1 / total_processed,
        "output_path": output_path
    }


def main():
    parser = argparse.ArgumentParser(description="Generate initial experience store")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/latent_memory/phase0_coldstart.yaml",
        help="Config file path"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="/root/autodl-tmp/initial_experience.jsonl",
        help="Output experience store path"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=500,
        help="Number of samples to generate"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for generation"
    )
    parser.add_argument(
        "--min-reward",
        type=float,
        default=0.7,
        help="Minimum reward threshold for writeback"
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
        split='train',
        cache_dir='/root/autodl-tmp/datasets'
    )
    print(f"✓ Dataset loaded: {len(dataset)} samples")

    # Generate experiences
    results = generate_initial_experiences(
        model=model,
        dataset=dataset,
        output_path=args.output,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        min_reward=args.min_reward,
        device=device
    )

    # Save results summary
    summary_path = args.output.replace('.jsonl', '_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✓ Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
