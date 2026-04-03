#!/bin/bash
# This script executes the re-experiments for A matrix fix.
# It delegates the queueing and GPU management to the python script.

echo "Starting re-experiments with fixed A matrix for beta=0.5, 0.3, 0.1 across 2 GPUs..."
python3 run_fixed_A_experiments.py
echo "All experiments completed!"
