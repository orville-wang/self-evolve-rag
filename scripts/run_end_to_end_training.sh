#!/bin/bash
#
# 端到端 Self-Evolving RAG 训练流程
#
# 包含三个阶段:
# Phase 0: 冷启动 - 生成初始经验库
# Phase 1: 自进化训练 - 使用 memory 训练 Weaver
# Phase 2: 评估对比 - 对比 baseline vs memory-enhanced
#

set -e  # Exit on error

echo "================================================================================"
echo "Self-Evolving RAG 端到端训练流程"
echo "================================================================================"
echo ""

# Configuration
NUM_PHASE0_SAMPLES=500
NUM_PHASE1_EPOCHS=3
NUM_EVAL_SAMPLES=200
MIN_REWARD=0.7

EXPERIENCE_PATH="/root/autodl-tmp/initial_experience.jsonl"
TRIGGER_LABELS_PATH="/root/autodl-tmp/trigger_labels.jsonl"
RESULTS_DIR="/root/autodl-tmp/results"

mkdir -p $RESULTS_DIR

# ============================================================================
# Phase 0: 冷启动 - 生成初始经验库
# ============================================================================
echo "================================================================================"
echo "Phase 0: 冷启动 - 生成初始经验库"
echo "================================================================================"
echo "  - 样本数: $NUM_PHASE0_SAMPLES"
echo "  - 最小 reward: $MIN_REWARD"
echo "  - 输出路径: $EXPERIENCE_PATH"
echo "================================================================================"
echo ""

if [ -f "$EXPERIENCE_PATH" ]; then
    echo "⚠ 经验库已存在: $EXPERIENCE_PATH"
    read -p "是否重新生成? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -f "$EXPERIENCE_PATH"
        echo "✓ 已删除旧经验库"
    else
        echo "✓ 使用现有经验库"
    fi
fi

if [ ! -f "$EXPERIENCE_PATH" ]; then
    echo "开始生成初始经验库..."
    python scripts/training/generate_initial_experience.py \
        --config configs/latent_memory/phase0_coldstart.yaml \
        --output "$EXPERIENCE_PATH" \
        --num-samples $NUM_PHASE0_SAMPLES \
        --batch-size 4 \
        --min-reward $MIN_REWARD

    if [ $? -eq 0 ]; then
        echo "✓ Phase 0 完成"
    else
        echo "✗ Phase 0 失败"
        exit 1
    fi
else
    echo "✓ 使用现有经验库: $EXPERIENCE_PATH"
fi

# 统计经验库
NUM_EXPERIENCES=$(wc -l < "$EXPERIENCE_PATH")
echo ""
echo "经验库统计:"
echo "  - 总条目数: $NUM_EXPERIENCES"
echo ""

# ============================================================================
# Phase 0.5: 生成 Trigger 训练标注（可选）
# ============================================================================
echo "================================================================================"
echo "Phase 0.5: 生成 Trigger 训练标注（可选）"
echo "================================================================================"
echo ""

read -p "是否生成 Trigger 训练标注? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "开始生成 Trigger 标注..."
    python scripts/training/generate_trigger_labels.py \
        --method heuristic \
        --memory-store "$EXPERIENCE_PATH" \
        --output "$TRIGGER_LABELS_PATH" \
        --num-samples 1000

    if [ $? -eq 0 ]; then
        echo "✓ Trigger 标注完成"
    else
        echo "✗ Trigger 标注失败"
        exit 1
    fi
else
    echo "✓ 跳过 Trigger 标注"
fi

# ============================================================================
# Phase 1: 自进化训练
# ============================================================================
echo ""
echo "================================================================================"
echo "Phase 1: 自进化训练"
echo "================================================================================"
echo "  - 训练轮数: $NUM_PHASE1_EPOCHS"
echo "  - 经验库: $EXPERIENCE_PATH"
echo "  - 配置: configs/latent_memory/triviaqa_self_evolving_rag.yaml"
echo "================================================================================"
echo ""

read -p "是否开始训练? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "开始自进化训练..."
    python scripts/training/train_self_evolving_rag.py \
        --phase 1 \
        --config configs/latent_memory/triviaqa_self_evolving_rag.yaml \
        --experience-store "$EXPERIENCE_PATH" \
        --num-epochs $NUM_PHASE1_EPOCHS

    if [ $? -eq 0 ]; then
        echo "✓ Phase 1 训练完成"
    else
        echo "✗ Phase 1 训练失败"
        exit 1
    fi
else
    echo "✓ 跳过训练"
fi

# ============================================================================
# Phase 2: 评估对比
# ============================================================================
echo ""
echo "================================================================================"
echo "Phase 2: 评估对比"
echo "================================================================================"
echo "  - 评估样本数: $NUM_EVAL_SAMPLES"
echo "  - Baseline 配置: configs/latent_memory/eval_baseline_small.yaml"
echo "  - Memory 配置: configs/latent_memory/eval_memory_small.yaml"
echo "================================================================================"
echo ""

read -p "是否开始评估? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    # Baseline evaluation
    echo "评估 Baseline 模型..."
    python scripts/evaluation/run_small_evaluation.py \
        --config configs/latent_memory/eval_baseline_small.yaml \
        --output "$RESULTS_DIR/baseline_results.json" \
        --num-samples $NUM_EVAL_SAMPLES

    if [ $? -ne 0 ]; then
        echo "✗ Baseline 评估失败"
        exit 1
    fi

    # Memory-enhanced evaluation
    echo "评估 Memory-Enhanced 模型..."
    python scripts/evaluation/run_small_evaluation.py \
        --config configs/latent_memory/eval_memory_small.yaml \
        --output "$RESULTS_DIR/memory_results.json" \
        --num-samples $NUM_EVAL_SAMPLES \
        --experience-store "$EXPERIENCE_PATH"

    if [ $? -ne 0 ]; then
        echo "✗ Memory 评估失败"
        exit 1
    fi

    echo "✓ Phase 2 评估完成"

    # Compare results
    echo ""
    echo "================================================================================"
    echo "结果对比"
    echo "================================================================================"
    python -c "
from memgen.utils.metrics import compare_metrics
compare_metrics(
    '$RESULTS_DIR/baseline_results.json',
    '$RESULTS_DIR/memory_results.json'
)
"
else
    echo "✓ 跳过评估"
fi

# ============================================================================
# Summary
# ============================================================================
echo ""
echo "================================================================================"
echo "训练流程完成"
echo "================================================================================"
echo ""
echo "生成的文件:"
echo "  - 经验库: $EXPERIENCE_PATH"
if [ -f "$TRIGGER_LABELS_PATH" ]; then
    echo "  - Trigger 标注: $TRIGGER_LABELS_PATH"
fi
echo "  - 结果目录: $RESULTS_DIR"
echo ""
echo "下一步:"
echo "  1. 查看训练日志: tail -f /root/autodl-tmp/self_evolving_rag_training.log"
echo "  2. 查看评估结果: cat $RESULTS_DIR/baseline_results.json"
echo "  3. 查看经验库: head -5 $EXPERIENCE_PATH"
echo ""
echo "================================================================================"
