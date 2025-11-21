#!/usr/bin/env bash

# Run Optuna hyperparameter tuning with all arguments explicitly provided

python hp_tuning.py \
  --env-id ObservationEcoli-v0 \
  --n-eval-episodes 5 \
  --n-parallel-envs 10 \
  --n-timesteps 500 \
  --device cpu \
  --max-observation-horizon 1 \
  --max-random-ode-param-variance 0.5 \
  --max-random-initial-state-variance 0.5 \
  --time-step 1.0 \
  --log-dir /tmp/optuna_logs/ \
  --eval-freq 500 \
  --n-trials 5
