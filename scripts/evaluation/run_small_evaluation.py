#!/usr/bin/env python
"""
Small-scale evaluation to compare Baseline vs Memory-Enhanced
Uses only 20 samples to avoid OOM
"""

import os
import sys
import json
import subprocess

os.environ['HF_HOME'] = '/root/autodl-tmp/models'
os.environ['TRANSFORMERS_CACHE'] = '/root/autodl-tmp/models'
os.environ['HF_DATASETS_CACHE'] = '/root/autodl-tmp/datasets'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

print("=" * 80)
print("Small-Scale Evaluation: Baseline vs Memory-Enhanced")
print("=" * 80)

# Step 1: Run Baseline Evaluation
print("\n" + "=" * 80)
print("Step 1: Running Baseline Evaluation (No Memory)")
print("=" * 80)

baseline_cmd = [
    "python", "main.py",
    "--cfg-path", "configs/latent_memory/eval_baseline_small.yaml"
]

print(f"Command: {' '.join(baseline_cmd)}")
print("This may take 5-10 minutes...")

try:
    result = subprocess.run(
        baseline_cmd,
        capture_output=True,
        text=True,
        timeout=600  # 10 minutes timeout
    )

    if result.returncode == 0:
        print("✓ Baseline evaluation completed")
    else:
        print(f"✗ Baseline evaluation failed with code {result.returncode}")
        print("Error output:")
        print(result.stderr[-1000:])  # Last 1000 chars
        sys.exit(1)

except subprocess.TimeoutExpired:
    print("✗ Baseline evaluation timed out")
    sys.exit(1)
except Exception as e:
    print(f"✗ Baseline evaluation error: {e}")
    sys.exit(1)

# Step 2: Run Memory-Enhanced Evaluation
print("\n" + "=" * 80)
print("Step 2: Running Memory-Enhanced Evaluation")
print("=" * 80)

memory_cmd = [
    "python", "main.py",
    "--cfg-path", "configs/latent_memory/eval_memory_small.yaml"
]

print(f"Command: {' '.join(memory_cmd)}")
print("This may take 5-10 minutes...")

try:
    result = subprocess.run(
        memory_cmd,
        capture_output=True,
        text=True,
        timeout=600
    )

    if result.returncode == 0:
        print("✓ Memory-enhanced evaluation completed")
    else:
        print(f"✗ Memory-enhanced evaluation failed with code {result.returncode}")
        print("Error output:")
        print(result.stderr[-1000:])
        sys.exit(1)

except subprocess.TimeoutExpired:
    print("✗ Memory-enhanced evaluation timed out")
    sys.exit(1)
except Exception as e:
    print(f"✗ Memory-enhanced evaluation error: {e}")
    sys.exit(1)

# Step 3: Compare Results
print("\n" + "=" * 80)
print("Step 3: Comparing Results")
print("=" * 80)

# Find result files
import glob

baseline_files = glob.glob("results/eval/triviaqa/**/answer.json", recursive=True)
baseline_files = [f for f in baseline_files if "eval_baseline_small" in f or (
    os.path.getmtime(f) < os.path.getmtime("configs/latent_memory/eval_memory_small.yaml")
)]

memory_files = glob.glob("results/eval/triviaqa/**/answer.json", recursive=True)
memory_files = [f for f in memory_files if "eval_memory_small" in f or (
    os.path.getmtime(f) > os.path.getmtime("configs/latent_memory/eval_baseline_small.yaml")
)]

if baseline_files and memory_files:
    baseline_file = sorted(baseline_files, key=os.path.getmtime)[-1]
    memory_file = sorted(memory_files, key=os.path.getmtime)[-1]

    print(f"Baseline results: {baseline_file}")
    print(f"Memory results: {memory_file}")

    with open(baseline_file, 'r') as f:
        baseline_data = json.load(f)

    with open(memory_file, 'r') as f:
        memory_data = json.load(f)

    print("\n" + "=" * 80)
    print("PERFORMANCE COMPARISON")
    print("=" * 80)

    metrics = ['accuracy', 'f1', 'exact_match', 'reward']
    for metric in metrics:
        if metric in baseline_data and metric in memory_data:
            baseline_val = baseline_data[metric]
            memory_val = memory_data[metric]

            if baseline_val > 0:
                improvement = ((memory_val - baseline_val) / baseline_val) * 100
            else:
                improvement = 0

            print(f"\n{metric.upper()}:")
            print(f"  Baseline:    {baseline_val:.4f}")
            print(f"  Memory:      {memory_val:.4f}")
            print(f"  Improvement: {improvement:+.2f}%")

    print("\n" + "=" * 80)
    print("Evaluation Completed Successfully!")
    print("=" * 80)

else:
    print("✗ Could not find result files")
    print(f"Baseline files found: {len(baseline_files)}")
    print(f"Memory files found: {len(memory_files)}")
