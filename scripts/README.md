# Scripts Directory

This directory contains all executable scripts for training, evaluation, and debugging.

## 📁 Directory Structure

```
scripts/
├── training/           # Training-related scripts
├── evaluation/         # Evaluation scripts
├── debug/              # Debug utilities
├── *.sh                # Shell scripts for automation
└── README.md           # This file
```

## 🚀 Training Scripts (`training/`)

### Core Training
- **train_self_evolving_rag.py** - Main self-evolving RAG training script
  - Supports 3-phase training pipeline
  - Usage: `python scripts/training/train_self_evolving_rag.py --config <config_file>`

### Data Generation
- **generate_initial_experience.py** - Phase 0: Generate initial experience pool
  - Cold-start experience generation
  - Usage: `python scripts/training/generate_initial_experience.py --num-samples 500`

- **generate_trigger_labels.py** - Generate trigger training labels
  - Creates heuristic labels for trigger training
  - Usage: `python scripts/training/generate_trigger_labels.py --method heuristic`

- **run_phase0_coldstart.py** - Simplified Phase 0 runner
  - Quick cold-start script
  - Usage: `python scripts/training/run_phase0_coldstart.py`

## 📊 Evaluation Scripts (`evaluation/`)

- **run_small_evaluation.py** - Small-scale evaluation
  - Quick evaluation on subset of data
  - Usage: `python scripts/evaluation/run_small_evaluation.py --config <config>`

- **simple_evaluation.py** - Simple baseline evaluation
  - Basic evaluation without memory
  - Usage: `python scripts/evaluation/simple_evaluation.py`

- **verify_memory_integration.py** - Memory integration verification
  - Tests memory retrieval and integration
  - Usage: `python scripts/evaluation/verify_memory_integration.py`

## 🐛 Debug Scripts (`debug/`)

- **debug_conversation_markers.py** - Debug conversation format markers
- **debug_dtype.py** - Debug dtype issues
- **debug_info_mask.py** - Debug info_mask generation
- **debug_info_mask_with_markers.py** - Debug info_mask with conversation markers

## 🔧 Shell Scripts

### Main Workflows
- **run_end_to_end_training.sh** - Complete end-to-end training pipeline
  - Phase 0: Cold-start
  - Phase 1: Self-evolving training
  - Phase 2: Evaluation
  - Usage: `bash scripts/run_end_to_end_training.sh`

- **run_self_evolving_rag.sh** - Interactive training launcher
  - Menu-driven interface
  - Supports individual phase execution
  - Usage: `bash scripts/run_self_evolving_rag.sh`

### Model Training
- **weaver_train.sh** - Train Weaver model
- **trigger_train.sh** - Train Trigger model
- **eval.sh** - Evaluate trained models

### Setup
- **setup_env.sh** - Environment setup
- **setup_wandb.sh** - Weights & Biases setup
- **start_training_with_wandb.sh** - Start training with W&B logging

## 📝 Usage Examples

### Complete Training Pipeline
```bash
# Run full end-to-end training
bash scripts/run_end_to_end_training.sh
```

### Individual Phases
```bash
# Phase 0: Generate experience pool
python scripts/training/generate_initial_experience.py \
    --num-samples 500 \
    --output /root/autodl-tmp/experience.jsonl

# Phase 1: Train with self-evolution
python scripts/training/train_self_evolving_rag.py \
    --config configs/latent_memory/triviaqa_self_evolving_rag.yaml \
    --phase 1

# Phase 2: Evaluate
python scripts/evaluation/run_small_evaluation.py \
    --config configs/latent_memory/eval_memory_small.yaml \
    --num-samples 50
```

### Debug
```bash
# Debug info_mask issues
python scripts/debug/debug_info_mask.py

# Verify memory integration
python scripts/evaluation/verify_memory_integration.py
```

## 🔗 Related Directories

- [../tests/](../tests/) - Unit tests and integration tests
- [../configs/](../configs/) - Configuration files
- [../docs/](../docs/) - Documentation

## 📞 For More Information

See the main [README](../README.md) for project overview and setup instructions.
