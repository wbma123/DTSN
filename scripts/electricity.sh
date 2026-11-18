#!/bin/bash

mkdir -p ./logs/final

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path electricity.csv \
  --model_id electricity_96_ema \
  --model DTSN \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --enc_in 321 \
  --des Exp \
  --itr 1 \
  --batch_size 256 \
  --learning_rate 0.005 \
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
  --dloss_beta 0.3 > logs/final/DTSN_electricity_96_96.log

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path electricity.csv \
  --model_id electricity_192_ema \
  --model DTSN \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 192 \
  --enc_in 321 \
  --des Exp \
  --itr 1 \
  --batch_size 256 \
  --learning_rate 0.005 \
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
  --dloss_beta 0.5 > logs/final/DTSN_electricity_96_192.log

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path electricity.csv \
  --model_id electricity_336_ema \
  --model DTSN \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 336 \
  --enc_in 321 \
  --des Exp \
  --itr 1 \
  --batch_size 256 \
  --learning_rate 0.005 \
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
  --dloss_beta 0.5 > logs/final/DTSN_electricity_96_336.log

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path electricity.csv \
  --model_id electricity_720_ema \
  --model DTSN \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 720 \
  --enc_in 321 \
  --des Exp \
  --itr 1 \
  --batch_size 256 \
  --learning_rate 0.005 \
  --lradj sigmoid \
  --ma_type ema \
  --alpha 0.3 \
  --beta 0.3 \
  --seed 2021 \
  --use_gated_fusion 1 \
  --fusion_hidden 64 \
  --fusion_bias -0.5 \
  --loss_mode d_loss \
  --dloss_alpha 0.5 \
  --dloss_beta 0.3 > logs/final/DTSN_electricity_96_720.log


