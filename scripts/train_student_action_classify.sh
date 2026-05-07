#!/bin/bash
source .venv/bin/activate
export HF_ENDPOINT=https://hf-mirror.com
python train.py \
--model mobilenetv3_large_100 \
--data-dir /DATA/yujiannan/Datasets/process_20260422_limited \
--train-split train --val-split val \
--pretrained --num-classes 10 \
--epochs 80 \
--opt adamw --lr 3e-4 --weight-decay 0.02 \
--sched cosine --warmup-epochs 5 --min-lr 1e-6 \
--smoothing 0.1 \
--mixup 0 --cutmix 0.0 --mixup-off-epoch 60 \
--aa rand-m7-mstd0.5-inc1 --reprob 0.2 \
--model-ema --model-ema-decay 0.998 \
--amp