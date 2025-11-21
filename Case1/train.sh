#!/usr/bin/env bash
python Train_Case1.py \
  --total-timesteps 200000 \
  --n-parallel 20 \
  --learning-rate 3e-4 \
  --ent-coef 0.001 \
  --n-steps 10 \
  --batch-size 200 \
  --device cpu \
  --num-neurons 256 \
  --log-dir /tmp/log_training \
  --save-path ./saved_models/ \
  --checkpoint-freq 20000 \
  --eval-freq 10000 \
  --random-ode-param-variance 0.5 \
  --random-initial-state-variance 0.1 \
  --observation-horizon 2 \
  --time-step 1.

