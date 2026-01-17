"""
评估指标和监控统计工具

提供以下功能:
1. Memory 命中率统计
2. Trigger 触发率统计
3. 经验库质量评估
4. 训练/评估指标收集
"""

from typing import Dict, List, Optional
from dataclasses import dataclass, field
from collections import defaultdict
import json
import time


@dataclass
class MemoryMetrics:
    """Memory 系统指标"""

    # Memory 检索统计
    total_queries: int = 0
    memory_hits: int = 0
    memory_misses: int = 0
    avg_retrieval_score: float = 0.0

    # Trigger 统计
    total_augmentation_points: int = 0
    trigger_invoke_count: int = 0
    trigger_skip_count: int = 0

    # 经验库统计
    total_experiences: int = 0
    writeback_attempts: int = 0
    writeback_success: int = 0
    avg_writeback_reward: float = 0.0

    # 性能统计
    avg_retrieval_time_ms: float = 0.0
    avg_generation_time_ms: float = 0.0

    def __post_init__(self):
        self._retrieval_scores: List[float] = []
        self._writeback_rewards: List[float] = []
        self._retrieval_times: List[float] = []
        self._generation_times: List[float] = []

    @property
    def memory_hit_rate(self) -> float:
        """Memory 命中率"""
        if self.total_queries == 0:
            return 0.0
        return self.memory_hits / self.total_queries

    @property
    def trigger_invoke_rate(self) -> float:
        """Trigger 触发率"""
        if self.total_augmentation_points == 0:
            return 0.0
        return self.trigger_invoke_count / self.total_augmentation_points

    @property
    def writeback_success_rate(self) -> float:
        """写回成功率"""
        if self.writeback_attempts == 0:
            return 0.0
        return self.writeback_success / self.writeback_attempts

    def record_retrieval(self, hit: bool, score: Optional[float] = None, time_ms: Optional[float] = None):
        """记录一次检索"""
        self.total_queries += 1
        if hit:
            self.memory_hits += 1
            if score is not None:
                self._retrieval_scores.append(score)
        else:
            self.memory_misses += 1

        if time_ms is not None:
            self._retrieval_times.append(time_ms)

    def record_trigger(self, invoked: bool):
        """记录一次 Trigger 决策"""
        self.total_augmentation_points += 1
        if invoked:
            self.trigger_invoke_count += 1
        else:
            self.trigger_skip_count += 1

    def record_writeback(self, success: bool, reward: Optional[float] = None):
        """记录一次写回尝试"""
        self.writeback_attempts += 1
        if success:
            self.writeback_success += 1
            if reward is not None:
                self._writeback_rewards.append(reward)

    def record_generation_time(self, time_ms: float):
        """记录生成时间"""
        self._generation_times.append(time_ms)

    def compute_averages(self):
        """计算平均值"""
        if self._retrieval_scores:
            self.avg_retrieval_score = sum(self._retrieval_scores) / len(self._retrieval_scores)

        if self._writeback_rewards:
            self.avg_writeback_reward = sum(self._writeback_rewards) / len(self._writeback_rewards)

        if self._retrieval_times:
            self.avg_retrieval_time_ms = sum(self._retrieval_times) / len(self._retrieval_times)

        if self._generation_times:
            self.avg_generation_time_ms = sum(self._generation_times) / len(self._generation_times)

    def to_dict(self) -> Dict:
        """转换为字典"""
        self.compute_averages()
        return {
            "memory": {
                "total_queries": self.total_queries,
                "hits": self.memory_hits,
                "misses": self.memory_misses,
                "hit_rate": self.memory_hit_rate,
                "avg_retrieval_score": self.avg_retrieval_score,
                "avg_retrieval_time_ms": self.avg_retrieval_time_ms,
            },
            "trigger": {
                "total_augmentation_points": self.total_augmentation_points,
                "invoke_count": self.trigger_invoke_count,
                "skip_count": self.trigger_skip_count,
                "invoke_rate": self.trigger_invoke_rate,
            },
            "experience_store": {
                "total_experiences": self.total_experiences,
                "writeback_attempts": self.writeback_attempts,
                "writeback_success": self.writeback_success,
                "writeback_success_rate": self.writeback_success_rate,
                "avg_writeback_reward": self.avg_writeback_reward,
            },
            "performance": {
                "avg_generation_time_ms": self.avg_generation_time_ms,
            }
        }

    def print_summary(self):
        """打印统计摘要"""
        print("\n" + "="*80)
        print("Memory System Metrics")
        print("="*80)

        print("\n[Memory Retrieval]")
        print(f"  Total queries: {self.total_queries}")
        print(f"  Hits: {self.memory_hits} ({self.memory_hit_rate*100:.1f}%)")
        print(f"  Misses: {self.memory_misses}")
        print(f"  Avg retrieval score: {self.avg_retrieval_score:.3f}")
        print(f"  Avg retrieval time: {self.avg_retrieval_time_ms:.2f} ms")

        print("\n[Trigger]")
        print(f"  Total augmentation points: {self.total_augmentation_points}")
        print(f"  Invoke count: {self.trigger_invoke_count} ({self.trigger_invoke_rate*100:.1f}%)")
        print(f"  Skip count: {self.trigger_skip_count}")

        print("\n[Experience Store]")
        print(f"  Total experiences: {self.total_experiences}")
        print(f"  Writeback attempts: {self.writeback_attempts}")
        print(f"  Writeback success: {self.writeback_success} ({self.writeback_success_rate*100:.1f}%)")
        print(f"  Avg writeback reward: {self.avg_writeback_reward:.3f}")

        print("\n[Performance]")
        print(f"  Avg generation time: {self.avg_generation_time_ms:.2f} ms")

        print("="*80 + "\n")


@dataclass
class TaskMetrics:
    """任务性能指标"""

    # 准确率指标
    total_samples: int = 0
    correct_samples: int = 0
    total_em: float = 0.0
    total_f1: float = 0.0

    # 成本指标
    total_tokens_generated: int = 0
    total_retrieval_calls: int = 0

    def __post_init__(self):
        self._em_scores: List[float] = []
        self._f1_scores: List[float] = []

    @property
    def accuracy(self) -> float:
        """准确率"""
        if self.total_samples == 0:
            return 0.0
        return self.correct_samples / self.total_samples

    @property
    def avg_em(self) -> float:
        """平均 Exact Match"""
        if self.total_samples == 0:
            return 0.0
        return self.total_em / self.total_samples

    @property
    def avg_f1(self) -> float:
        """平均 F1"""
        if self.total_samples == 0:
            return 0.0
        return self.total_f1 / self.total_samples

    @property
    def avg_tokens_per_sample(self) -> float:
        """平均每样本生成 token 数"""
        if self.total_samples == 0:
            return 0.0
        return self.total_tokens_generated / self.total_samples

    @property
    def avg_retrieval_per_sample(self) -> float:
        """平均每样本检索次数"""
        if self.total_samples == 0:
            return 0.0
        return self.total_retrieval_calls / self.total_samples

    def record_sample(self, em: float, f1: float, num_tokens: int, num_retrievals: int = 0):
        """记录一个样本的结果"""
        self.total_samples += 1
        self.total_em += em
        self.total_f1 += f1
        self.total_tokens_generated += num_tokens
        self.total_retrieval_calls += num_retrievals

        if em == 1.0:
            self.correct_samples += 1

        self._em_scores.append(em)
        self._f1_scores.append(f1)

    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            "accuracy": self.accuracy,
            "exact_match": self.avg_em,
            "f1": self.avg_f1,
            "total_samples": self.total_samples,
            "correct_samples": self.correct_samples,
            "avg_tokens_per_sample": self.avg_tokens_per_sample,
            "avg_retrieval_per_sample": self.avg_retrieval_per_sample,
        }

    def print_summary(self):
        """打印统计摘要"""
        print("\n" + "="*80)
        print("Task Performance Metrics")
        print("="*80)

        print(f"\n  Total samples: {self.total_samples}")
        print(f"  Correct samples: {self.correct_samples}")
        print(f"  Accuracy: {self.accuracy*100:.2f}%")
        print(f"  Exact Match: {self.avg_em:.3f}")
        print(f"  F1 Score: {self.avg_f1:.3f}")
        print(f"  Avg tokens/sample: {self.avg_tokens_per_sample:.1f}")
        print(f"  Avg retrievals/sample: {self.avg_retrieval_per_sample:.2f}")

        print("="*80 + "\n")


class MetricsCollector:
    """指标收集器"""

    def __init__(self):
        self.memory_metrics = MemoryMetrics()
        self.task_metrics = TaskMetrics()
        self.start_time = time.time()

    def save(self, output_path: str):
        """保存指标到文件"""
        metrics = {
            "memory": self.memory_metrics.to_dict(),
            "task": self.task_metrics.to_dict(),
            "elapsed_time_seconds": time.time() - self.start_time,
        }

        with open(output_path, 'w') as f:
            json.dump(metrics, f, indent=2)

        print(f"✓ Metrics saved to: {output_path}")

    def print_summary(self):
        """打印所有指标摘要"""
        self.memory_metrics.print_summary()
        self.task_metrics.print_summary()

        elapsed = time.time() - self.start_time
        print(f"Total elapsed time: {elapsed:.2f} seconds")


def compare_metrics(baseline_path: str, treatment_path: str):
    """对比两个实验的指标"""

    with open(baseline_path, 'r') as f:
        baseline = json.load(f)

    with open(treatment_path, 'r') as f:
        treatment = json.load(f)

    print("\n" + "="*80)
    print("Metrics Comparison: Baseline vs Treatment")
    print("="*80)

    # Task metrics comparison
    print("\n[Task Performance]")
    baseline_task = baseline['task']
    treatment_task = treatment['task']

    metrics_to_compare = [
        ('Accuracy', 'accuracy', '%'),
        ('Exact Match', 'exact_match', ''),
        ('F1 Score', 'f1', ''),
        ('Avg tokens/sample', 'avg_tokens_per_sample', ''),
        ('Avg retrievals/sample', 'avg_retrieval_per_sample', ''),
    ]

    for name, key, unit in metrics_to_compare:
        baseline_val = baseline_task[key]
        treatment_val = treatment_task[key]
        diff = treatment_val - baseline_val
        diff_pct = (diff / baseline_val * 100) if baseline_val != 0 else 0

        if unit == '%':
            baseline_val *= 100
            treatment_val *= 100
            diff *= 100

        print(f"  {name}:")
        print(f"    Baseline:  {baseline_val:.3f}{unit}")
        print(f"    Treatment: {treatment_val:.3f}{unit}")
        print(f"    Diff:      {diff:+.3f}{unit} ({diff_pct:+.1f}%)")

    # Memory metrics (only for treatment)
    if 'memory' in treatment:
        print("\n[Memory System (Treatment only)]")
        memory = treatment['memory']
        print(f"  Memory hit rate: {memory['memory']['hit_rate']*100:.1f}%")
        print(f"  Trigger invoke rate: {memory['trigger']['invoke_rate']*100:.1f}%")
        print(f"  Writeback success rate: {memory['experience_store']['writeback_success_rate']*100:.1f}%")

    print("="*80 + "\n")
