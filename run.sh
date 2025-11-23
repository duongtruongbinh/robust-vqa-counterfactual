#!/bin/bash

# Example training script
# Modify arguments as needed for your experiment

CUDA_VISIBLE_DEVICES=0 python main.py \
  --dataset cpv2 \
  --output exp_name \
  --epochs 30 \
  --batch_size 512 \
  --use_wandb \
  --wandb_project CL-VQA \
  --phase1_epochs 5 \
  --phase2_epochs 17 \
  --contras_loss_weight 1.0 \
  --contras_loss_type new \
  --contras_temperature 0.05 \
  --use_ac_loss \
  --use_gd_loss \
  --ac_loss_weight 3.5 \
  --gd_loss_weight 8.0 \
  --initial_lr 1e-3 \
  --min_lr 5e-6 \
  --lr_decay_start 10 \
  --lr_decay_end 20 \
  --gamma 3.0 \
  --schedule_method warm_up