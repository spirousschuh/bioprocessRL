#!/usr/bin/env bash


# Increase the open file limit for this script
ulimit -n 8196

# Run Optuna hyperparameter tuning with all arguments explicitly provided

python hp_tuning.py \
  --env-id ObservationEcoli-v0 \
  --n-eval-episodes 10 \
  --n-parallel-envs 40 \
  --n-timesteps 1000000 \
  --device cpu \
  --max-observation-horizon 12 \
  --max-random-ode-param-variance 2. \
  --max-random-initial-state-variance 2. \
  --time-step 1.0 \
  --log-dir /tmp/optuna_logs/ \
  --eval-freq 20000 \
  --n-trials 500 \
  --evaluation-ode-parameter-variance 0.3 \
  --evaluation-initial-state-variance 0.1 \
  --num-evaluation-envs 30