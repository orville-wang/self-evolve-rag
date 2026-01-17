#!/usr/bin/env python
"""
Self-Evolving RAG Training Script

This script implements a three-phase training pipeline:
1. Phase 0: Cold-start - Build initial experience store from baseline model
2. Phase 1: Self-evolving training - Train with memory retrieval and writeback
3. Phase 2: Evaluation - Compare baseline vs memory-enhanced performance
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Setup environment
os.environ['HF_HOME'] = '/root/autodl-tmp/models'
os.environ['TRANSFORMERS_CACHE'] = '/root/autodl-tmp/models'
os.environ['HF_DATASETS_CACHE'] = '/root/autodl-tmp/datasets'
os.environ['TORCH_HOME'] = '/root/autodl-tmp/models'

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('/root/autodl-tmp/self_evolving_rag_training.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def phase_0_cold_start(config_path: str, experience_store_path: str, num_samples: int = 1000):
    """
    Phase 0: Cold-start - Build initial experience store

    Use baseline model to generate initial high-quality experiences.
    """
    logger.info("=" * 80)
    logger.info("PHASE 0: Cold-start - Building Initial Experience Store")
    logger.info("=" * 80)

    from omegaconf import OmegaConf
    from memgen.runner import MemGenRunner

    # Load config
    config = OmegaConf.load(config_path)

    # Modify config for cold-start
    config.run.mode = "eval"  # Use eval mode to generate experiences
    config.run.train_weaver = False
    config.memory.enable = False  # Disable memory retrieval for baseline
    config.memory.writeback.enable = True  # Enable writeback to collect experiences

    # Limit to num_samples for cold-start
    logger.info(f"Generating initial experiences from {num_samples} samples...")

    # Initialize runner
    runner = MemGenRunner(config)

    # Run evaluation to collect experiences
    # The runner will automatically write back high-quality experiences
    results = runner.run()

    logger.info(f"Cold-start completed. Experience store saved to: {experience_store_path}")
    logger.info(f"Initial experiences collected: {results.get('num_experiences', 0)}")

    return results


def phase_1_self_evolving_training(config_path: str, num_epochs: int = 3):
    """
    Phase 1: Self-evolving training with memory retrieval and writeback

    Train the model with:
    - Memory retrieval: Inject relevant experiences into prompts
    - Memory writeback: Continuously add new high-quality experiences
    """
    logger.info("=" * 80)
    logger.info("PHASE 1: Self-Evolving Training")
    logger.info("=" * 80)

    from omegaconf import OmegaConf
    from memgen.runner import MemGenRunner

    # Load config
    config = OmegaConf.load(config_path)

    # Ensure training mode with memory enabled
    config.run.mode = "train"
    config.run.train_weaver = True
    config.run.train_weaver_method = "grpo"
    config.memory.enable = True
    config.memory.writeback.enable = True
    config.run.weaver.grpo.num_train_epochs = num_epochs

    logger.info(f"Starting self-evolving training for {num_epochs} epochs...")
    logger.info(f"Memory retrieval: ENABLED (topk={config.memory.topk})")
    logger.info(f"Memory writeback: ENABLED (min_reward={config.memory.writeback.min_reward})")

    # Initialize runner
    runner = MemGenRunner(config)

    # Run training
    results = runner.run()

    logger.info("Self-evolving training completed!")
    logger.info(f"Final model saved to: {results.get('output_dir', 'N/A')}")

    return results


def phase_2_evaluation(baseline_config_path: str, memory_config_path: str, model_path: str = None):
    """
    Phase 2: Evaluation - Compare baseline vs memory-enhanced performance
    """
    logger.info("=" * 80)
    logger.info("PHASE 2: Evaluation")
    logger.info("=" * 80)

    from omegaconf import OmegaConf
    from memgen.runner import MemGenRunner

    # Evaluate baseline (no memory)
    logger.info("\n--- Evaluating Baseline (No Memory) ---")
    baseline_config = OmegaConf.load(baseline_config_path)
    baseline_config.run.mode = "eval"
    baseline_config.memory.enable = False

    if model_path:
        baseline_config.model.load_model_path = model_path

    baseline_runner = MemGenRunner(baseline_config)
    baseline_results = baseline_runner.run()

    logger.info(f"Baseline Results: {baseline_results}")

    # Evaluate with memory
    logger.info("\n--- Evaluating Memory-Enhanced Model ---")
    memory_config = OmegaConf.load(memory_config_path)
    memory_config.run.mode = "eval"
    memory_config.memory.enable = True
    memory_config.memory.writeback.enable = False  # No writeback during eval

    if model_path:
        memory_config.model.load_model_path = model_path

    memory_runner = MemGenRunner(memory_config)
    memory_results = memory_runner.run()

    logger.info(f"Memory-Enhanced Results: {memory_results}")

    # Compare results
    logger.info("\n" + "=" * 80)
    logger.info("COMPARISON SUMMARY")
    logger.info("=" * 80)

    for metric in ['accuracy', 'f1', 'exact_match']:
        if metric in baseline_results and metric in memory_results:
            baseline_val = baseline_results[metric]
            memory_val = memory_results[metric]
            improvement = ((memory_val - baseline_val) / baseline_val) * 100
            logger.info(f"{metric.upper()}:")
            logger.info(f"  Baseline: {baseline_val:.4f}")
            logger.info(f"  Memory:   {memory_val:.4f}")
            logger.info(f"  Improvement: {improvement:+.2f}%")

    return baseline_results, memory_results


def main():
    parser = argparse.ArgumentParser(description="Self-Evolving RAG Training Pipeline")
    parser.add_argument("--config", type=str, required=True, help="Path to training config file")
    parser.add_argument("--phase", type=str, choices=["all", "0", "1", "2"], default="all",
                        help="Which phase to run: 0=cold-start, 1=training, 2=eval, all=all phases")
    parser.add_argument("--cold-start-samples", type=int, default=1000,
                        help="Number of samples for cold-start (Phase 0)")
    parser.add_argument("--num-epochs", type=int, default=3,
                        help="Number of training epochs (Phase 1)")
    parser.add_argument("--model-path", type=str, default=None,
                        help="Path to trained model for evaluation (Phase 2)")

    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("Self-Evolving RAG Training Pipeline")
    logger.info("=" * 80)
    logger.info(f"Config: {args.config}")
    logger.info(f"Phase: {args.phase}")

    # Load config to get experience store path
    from omegaconf import OmegaConf
    config = OmegaConf.load(args.config)
    experience_store_path = config.memory.store_path

    try:
        if args.phase in ["all", "0"]:
            phase_0_cold_start(args.config, experience_store_path, args.cold_start_samples)

        if args.phase in ["all", "1"]:
            phase_1_self_evolving_training(args.config, args.num_epochs)

        if args.phase in ["all", "2"]:
            baseline_config = args.config.replace("self_evolving_rag", "baseline")
            phase_2_evaluation(baseline_config, args.config, args.model_path)

        logger.info("\n" + "=" * 80)
        logger.info("Pipeline completed successfully!")
        logger.info("=" * 80)

    except Exception as e:
        logger.error(f"Pipeline failed with error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
