## 🚀 快速开始

### 1. 环境配置

#### 方式一：使用 requirements.txt
```bash
conda create -n memgen python=3.10
conda activate memgen
pip install -r requirements.txt
```

#### 方式二：使用 memgen.yml
```bash
conda env create -f memgen.yml
conda activate memgen
```

### 2. 配置检索环境（可选）

如需使用检索功能，请参考 [Search-R1](https://github.com/PeterGriffinJin/Search-R1?tab=readme-ov-file#retriever-environment-optional) 配置检索环境。

### 3. 运行训练

#### 🎯 完整的自进化 RAG 训练流程

**方式一：交互式菜单**
```bash
bash scripts/run_self_evolving_rag.sh
```

**方式二：端到端自动化流程**
```bash
bash scripts/run_end_to_end_training.sh
```

这个脚本会自动执行三个阶段：
- **Phase 0**：冷启动 - 生成初始经验库
- **Phase 1**：自进化训练 - 使用 memory 训练 Weaver
- **Phase 2**：评估对比 - 对比 baseline vs memory-enhanced

#### 🔧 分阶段运行

**Phase 0: 生成初始经验库**
```bash
python scripts/training/generate_initial_experience.py \
    --num-samples 500 \
    --output /root/autodl-tmp/experience.jsonl \
    --min-reward 0.7
```

**Phase 1: 自进化训练**
```bash
python scripts/training/train_self_evolving_rag.py \
    --config configs/latent_memory/triviaqa_self_evolving_rag.yaml \
    --phase 1 \
    --experience-store /root/autodl-tmp/experience.jsonl
```

**Phase 2: 评估对比**
```bash
# Baseline 评估
python scripts/evaluation/run_small_evaluation.py \
    --config configs/latent_memory/eval_baseline_small.yaml \
    --num-samples 50

# Memory-Enhanced 评估
python scripts/evaluation/run_small_evaluation.py \
    --config configs/latent_memory/eval_memory_small.yaml \
    --num-samples 50 \
    --experience-store /root/autodl-tmp/experience.jsonl
```

### 4. 原始 MemGen 模型训练

#### Weaver 模型
```bash
# 训练
bash scripts/weaver_train.sh

# 评估（需先修改 scripts/eval.sh 中的 LOAD_MODEL_PATH）
bash scripts/eval.sh
```

#### Trigger 模型
```bash
# 训练
bash scripts/trigger_train.sh

# 评估（需先修改 scripts/eval.sh 中的 LOAD_MODEL_PATH）
bash scripts/eval.sh
```

---

## 📂 项目结构

```
MemGen/
├── memgen/              # 核心库
│   ├── model/           # 模型实现（Trigger, Weaver, MemGen）
│   ├── memory/          # 记忆管理
│   ├── trainer/         # 训练逻辑（SFT, GRPO）
│   └── utils/           # 工具函数
├── scripts/             # 可执行脚本
│   ├── training/        # 训练脚本
│   ├── evaluation/      # 评估脚本
│   ├── debug/           # 调试工具
│   └── *.sh             # Shell 自动化脚本
├── tests/               # 单元测试和集成测试
├── configs/             # 配置文件
├── docs/                # 文档
│   ├── guides/          # 用户指南
│   ├── plans/           # 设计文档
│   ├── reports/         # 技术报告
│   └── archive/         # 历史文档
├── data/                # 数据处理
├── interactions/        # 交互处理器
├── common/              # 通用工具
└── main.py              # 主入口
```

---


## 📄 致谢


```bibtex
@article{zhang2025memgen,
  title={MemGen: Weaving Generative Latent Memory for Self-Evolving Agents},
  author={Zhang, Guibin and Fu, Muxin and Yan, Shuicheng},
  journal={arXiv preprint arXiv:2509.24704},
  year={2025}
}
```

---