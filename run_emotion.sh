#!/bin/bash
cd /root/r2ec
/root/miniconda3/envs/RREC/bin/python train.py \
    --run_name emotion-v1 \
    --dataset_dir data/ED_hard_a_processed \
    --vllm_gpu_memory_utilization 0.3\
    --num_train_epochs 1 \
    --use_vllm
