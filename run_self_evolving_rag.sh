#!/bin/bash
# 自进化 RAG 训练快速启动脚本

set -e  # Exit on error

echo "=========================================="
echo "自进化 RAG 训练快速启动"
echo "=========================================="

# 激活环境
echo "1. 激活 conda 环境..."
source /root/miniconda3/etc/profile.d/conda.sh
conda activate memgen
source setup_env.sh

# 配置参数
CONFIG="configs/latent_memory/triviaqa_self_evolving_rag.yaml"
COLD_START_SAMPLES=1000
NUM_EPOCHS=3
LOG_FILE="/root/autodl-tmp/self_evolving_rag_training.log"

echo ""
echo "配置信息："
echo "  - 配置文件: $CONFIG"
echo "  - 冷启动样本数: $COLD_START_SAMPLES"
echo "  - 训练轮数: $NUM_EPOCHS"
echo "  - 日志文件: $LOG_FILE"
echo ""

# 询问用户选择
echo "请选择运行模式："
echo "  1) 完整流程（Phase 0 + 1 + 2）"
echo "  2) 仅冷启动（Phase 0）"
echo "  3) 仅训练（Phase 1）"
echo "  4) 仅评估（Phase 2）"
echo "  5) 测试组件"
echo "  6) 退出"
echo ""
read -p "请输入选项 (1-6): " choice

case $choice in
    1)
        echo ""
        echo "=========================================="
        echo "运行完整流程"
        echo "=========================================="
        python train_self_evolving_rag.py \
            --config $CONFIG \
            --phase all \
            --cold-start-samples $COLD_START_SAMPLES \
            --num-epochs $NUM_EPOCHS
        ;;
    2)
        echo ""
        echo "=========================================="
        echo "Phase 0: 冷启动"
        echo "=========================================="
        python train_self_evolving_rag.py \
            --config $CONFIG \
            --phase 0 \
            --cold-start-samples $COLD_START_SAMPLES
        ;;
    3)
        echo ""
        echo "=========================================="
        echo "Phase 1: 自进化训练"
        echo "=========================================="
        python train_self_evolving_rag.py \
            --config $CONFIG \
            --phase 1 \
            --num-epochs $NUM_EPOCHS
        ;;
    4)
        echo ""
        echo "=========================================="
        echo "Phase 2: 评估"
        echo "=========================================="
        read -p "请输入模型路径（留空使用 baseline）: " model_path
        if [ -z "$model_path" ]; then
            python train_self_evolving_rag.py \
                --config $CONFIG \
                --phase 2
        else
            python train_self_evolving_rag.py \
                --config $CONFIG \
                --phase 2 \
                --model-path "$model_path"
        fi
        ;;
    5)
        echo ""
        echo "=========================================="
        echo "测试组件"
        echo "=========================================="
        python test_self_evolving_components.py
        ;;
    6)
        echo "退出"
        exit 0
        ;;
    *)
        echo "无效选项"
        exit 1
        ;;
esac

echo ""
echo "=========================================="
echo "完成！"
echo "=========================================="
echo ""
echo "查看日志："
echo "  tail -f $LOG_FILE"
echo ""
echo "查看 TensorBoard："
echo "  tensorboard --logdir results/train/triviaqa/"
echo ""
