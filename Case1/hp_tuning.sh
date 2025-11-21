#!/usr/bin/env bash

# Run Optuna hyperparameter tuning with all arguments explicitly provided

python hp_tuning.py \
  --env-id ObservationEcoli-v0 \
  --n-eval-episodes 10 \
  --n-parallel-envs 40 \
  --n-timesteps 500000 \
  --device cpu \
  --max-observation-horizon 9 \
  --max-random-ode-param-variance 1. \
  --max-random-initial-state-variance 1. \
  --time-step 1.0 \
  --log-dir /tmp/optuna_logs/ \
  --eval-freq 10000 \
  --n-trials 1000
