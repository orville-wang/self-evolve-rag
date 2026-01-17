#!/usr/bin/env python
"""
Phase 0: Cold-start - Build Initial Experience Store

This script runs the baseline model on a subset of training data
to collect high-quality experiences for the initial experience store.
"""

import os
import sys

# Setup environment
os.environ['HF_HOME'] = '/root/autodl-tmp/models'
os.environ['TRANSFORMERS_CACHE'] = '/root/autodl-tmp/models'
os.environ['HF_DATASETS_CACHE'] = '/root/autodl-tmp/datasets'
os.environ['TORCH_HOME'] = '/root/autodl-tmp/models'

import logging
from omegaconf import OmegaConf

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    logger.info("=" * 80)
    logger.info("Phase 0: Cold-start - Building Initial Experience Store")
    logger.info("=" * 80)

    config_path = "configs/latent_memory/phase0_coldstart.yaml"
    config = OmegaConf.load(config_path)

    # Ensure writeback is enabled but retrieval is disabled
    config.memory.enable = False  # No retrieval
    config.memory.writeback.enable = True  # Enable writeback
    config.run.mode = "eval"  # Evaluation mode

    logger.info("\nConfiguration:")
    logger.info(f"  Mode: {config.run.mode}")
    logger.info(f"  Memory retrieval: {config.memory.enable}")
    logger.info(f"  Memory writeback: {config.memory.writeback.enable}")
    logger.info(f"  Experience store: {config.memory.store_path}")
    logger.info(f"  Min reward: {config.memory.writeback.min_reward}")

    # Import here to avoid early loading
    from memgen.runner import MemGenRunner

    logger.info("\nInitializing runner...")
    runner = MemGenRunner(config)

    # Check if experience store exists
    if os.path.exists(config.memory.store_path):
        logger.warning(f"Experience store already exists: {config.memory.store_path}")
        response = input("Do you want to overwrite it? (yes/no): ")
        if response.lower() != 'yes':
            logger.info("Aborted.")
            return
        os.remove(config.memory.store_path)

    logger.info("\n" + "=" * 80)
    logger.info("Starting evaluation to collect experiences...")
    logger.info("=" * 80)

    try:
        # Run evaluation
        results = runner.run()

        logger.info("\n" + "=" * 80)
        logger.info("Phase 0 Completed!")
        logger.info("=" * 80)

        # Check experience store
        if os.path.exists(config.memory.store_path):
            with open(config.memory.store_path, 'r') as f:
                num_experiences = sum(1 for _ in f)
            logger.info(f"\n✓ Experience store created: {config.memory.store_path}")
            logger.info(f"✓ Number of experiences: {num_experiences}")

            # Show sample experiences
            logger.info("\nSample experiences:")
            with open(config.memory.store_path, 'r') as f:
                import json
                for i, line in enumerate(f):
                    if i >= 3:  # Show first 3
                        break
                    exp = json.loads(line)
                    logger.info(f"  {i+1}. Query: {exp.get('query', '')[:50]}...")
                    logger.info(f"     Reward: {exp.get('reward', 0):.3f}")
        else:
            logger.warning("Experience store was not created!")

        logger.info("\nNext step: Run Phase 1 training with memory retrieval")

    except Exception as e:
        logger.error(f"\nPhase 0 failed: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
