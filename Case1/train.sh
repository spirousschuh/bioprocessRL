#!/usr/bin/env bash
#SBATCH --job-name=hp_tuning_bioprocessRL                                    # Job name
#SBATCH --mail-type=ALL                                                 # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=christoph.lange@tu-berlin.de                        # Where to send mail.  Set this to your email address
#SBATCH --ntasks=1                                                      # Number of MPI tasks (i.e. processes)
#SBATCH --cpus-per-task=20                                               # Number of cores per MPI task 
#SBATCH --gpus-per-task=0
#SBATCH --nodes=1                                                       # Maximum number of nodes to be allocated
#SBATCH --mem-per-cpu=3GB                                               # Memory (i.e. RAM) per processor
#SBATCH --time=14-00:00:00                                              # Wall time limit (days-hrs:min:sec)
#SBATCH --partition=standard                                                # standard cpu partition
#SBATCH -D /scratch/ch.lange/git_projects/bioprocessRL/Case1		# Working Directory
#SBATCH --account=kiwi

set -x


python Train_Case1.py \
  --total-timesteps 1000000 \
  --n-parallel 40 \
  --learning-rate 3e-4 \
  --ent-coef 0.001 \
  --n-steps 10 \
  --batch-size 160 \
  --device cpu \
  --num-neurons 128 \
  --log-dir ./log_training \
  --save-path ./saved_models/ \
  --checkpoint-freq 100000 \
  --eval-freq 100000 \
  --random-ode-param-variance 0.1 \
  --ode-param-perturbation-type uniform \
  --random-initial-state-variance 0.01 \
  --feed-std 5.0 \
  --observation-horizon 10 \
  --time-step 1. \
  --model-name kiwiGym-CS1

