#!/bin/bash

mkdir -p ./logs/final

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path ETTm2.csv \
  --model_id ETTm2_96_ema \
  --model DTSN \
  --data ETTm2 \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --enc_in 7 \
  --des Exp \
  --itr 1 \
  --batch_size 2048 \
  --learning_rate 0.0001 \
  --lradj sigmoid \
  --ma_type ema \
  --alpha 0.3 \
  --beta 0.3 \
  --seed 2021 \
  --use_gated_fusion 1 \
  --fusion_hidden 64 \
  --fusion_bias -0.5 \
  --loss_mode d_loss \
  --dloss_alpha 0.3 \
  --dloss_beta 0.2 > logs/final/DTSN_ETTm2_96_96.log

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path ETTm2.csv \
  --model_id ETTm2_192_ema \
  --model DTSN \
  --data ETTm2 \
  --features M \
  --seq_len 96 \
  --pred_len 192 \
  --enc_in 7 \
  --des Exp \
  --itr 1 \
  --batch_size 2048 \
  --learning_rate 0.0001 \
  --lradj sigmoid \
  --ma_type ema \
  --alpha 0.3 \
  --beta 0.3 \
  --seed 2021 \
  --use_gated_fusion 1 \
  --fusion_hidden 64 \
  --fusion_bias -0.5 \
  --loss_mode d_loss \
  --dloss_alpha 0.2 \
  --dloss_beta 0.5 > logs/final/DTSN_ETTm2_96_192.log

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path ETTm2.csv \
  --model_id ETTm2_336_ema \
  --model DTSN \
  --data ETTm2 \
  --features M \
  --seq_len 96 \
  --pred_len 336 \
  --enc_in 7 \
  --des Exp \
  --itr 1 \
  --batch_size 2048 \
  --learning_rate 0.0001 \
  --lradj sigmoid \
  --ma_type ema \
  --alpha 0.3 \
  --beta 0.3 \
  --seed 2021 \
  --use_gated_fusion 1 \
  --fusion_hidden 64 \
  --fusion_bias -0.5 \
  --loss_mode d_loss \
  --dloss_alpha 0.2 \
  --dloss_beta 0.5 > logs/final/DTSN_ETTm2_96_336.log

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path ETTm2.csv \
  --model_id ETTm2_720_ema \
  --model DTSN \
  --data ETTm2 \
  --features M \
  --seq_len 96 \
  --pred_len 720 \
  --enc_in 7 \
  --des Exp \
  --itr 1 \
  --batch_size 2048 \
  --learning_rate 0.0001 \
  --lradj sigmoid \
  --ma_type ema \
  --alpha 0.3 \
  --beta 0.3 \
  --seed 2021 \
  --use_gated_fusion 1 \
  --fusion_hidden 64 \
  --fusion_bias -0.5 \
  --loss_mode d_loss \
  --dloss_alpha 0.2 \
  --dloss_beta 0.6 > logs/final/DTSN_ETTm2_96_720.log


