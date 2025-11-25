import os
import time
import argparse
from typing import Any, Dict
from copy import deepcopy

# Stable-Baselines3 imports
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize  # <--- Added VecNormalize
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement, BaseCallback

# Optuna imports
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

# Environment imports
import Case1.observation_env

# --- Default Global Configuration ---
DEFAULT_ENV_ID = 'ObservationEcoli-v0'
DEFAULT_N_EVAL_EPISODES = 5
DEFAULT_N_PARALLEL_ENVS = 10
DEFAULT_N_TIMESTEPS = 100000

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class SyncEvalCallback(BaseCallback):
    """
    Custom Callback to synchronize normalization stats from Train to Eval environment.
    This ensures the Eval env normalizes observations exactly like the Agent is used to.
    """

    def __init__(self, train_env, eval_env):
        super().__init__()
        self.train_env = train_env
        self.eval_env = eval_env

    def _on_step(self) -> bool:
        # Sync the Moving Average statistics here
        if hasattr(self.train_env, 'obs_rms') and hasattr(self.eval_env, 'obs_rms'):
            self.eval_env.obs_rms = deepcopy(self.train_env.obs_rms)
        return True


class TrialEvalCallback(EvalCallback):
    """Callback that reports the mean reward to Optuna for pruning."""
    def __init__(self, eval_env, trial, train_env_to_sync=None, n_eval_episodes=5,
                 eval_freq=10000, deterministic=True, verbose=0):
        early_stop_cb = StopTrainingOnNoModelImprovement(
            max_no_improvement_evals=5,
            min_evals=10,
            verbose=0,
        )

        if train_env_to_sync is not None:
            self.sync_cb = SyncEvalCallback(train_env_to_sync, eval_env)
        else:
            self.sync_cb = None

        super().__init__(
            eval_env,
            best_model_save_path=None,
            log_path=None,
            eval_freq=eval_freq,
            deterministic=deterministic,
            render=False,
            verbose=verbose,
            n_eval_episodes=n_eval_episodes,
            callback_after_eval=early_stop_cb,
        )

        self.trial = trial
        self.is_pruned = False

    def _on_step(self) -> bool:
        if self.sync_cb is not None and self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            self.sync_cb._on_step()

        continue_training = super()._on_step()
        if not continue_training:
            return False

        # after super()._on_step(), self.last_mean_reward has been updated
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            self.trial.report(self.last_mean_reward, self.num_timesteps)
            if self.trial.should_prune():
                self.is_pruned = True
                print(f"Trial {self.trial.number} pruned at step {self.num_timesteps}.")
                return False
        return True



def sample_ppo_hyperparameters(trial: optuna.Trial, env_args: Dict[str, Any], fixed_gamma: float) -> Dict[str, Any]:
    """Samples hyperparameters. NOTE: Gamma is now fixed/passed in."""

    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
    batch_size = trial.suggest_int('batch_size', 50, 500, step=50)
    ent_coef = trial.suggest_float('ent_coef', 1e-6, 5e-2, log=True)
    num_neurons = trial.suggest_int('num_neurons', 64, 1024, step=64)

    policy_kwargs = dict(net_arch=[num_neurons, num_neurons])

    return {
        "learning_rate": learning_rate,
        "n_steps": 10,
        "batch_size": batch_size,
        "gamma": fixed_gamma,  # <--- Use the pre-sampled gamma
        "ent_coef": ent_coef,
        "policy_kwargs": policy_kwargs,
        "device": env_args['device'],
        "verbose": 0,
    }


def objective(trial: optuna.Trial, args: argparse.Namespace) -> float:
    env_id = args.env_id
    n_parallel_envs = args.n_parallel_envs
    n_eval_episodes = args.n_eval_episodes
    n_timesteps = args.n_timesteps
    eval_freq_n_envs = max(args.eval_freq // n_parallel_envs, 1)

    # 1. SAMPLE GAMMA FIRST
    # VecNormalize needs gamma to normalize rewards correctly (Ret = R + gamma * Ret_old)
    # gamma = trial.suggest_float('gamma', 0.9, 0.9999, log=True)
    gamma = 1.0  # Fix gamma to 1.0 for this environment

    # Sample other environment vars
    observation_horizon = trial.suggest_int('observation_horizon', 1, args.max_observation_horizon)
    step_reward_weight = trial.suggest_float(
        'step_reward_weight',
        1e-12,
        args.max_step_reward_weight,
        log=True,
    )

    env_kwargs = {
        'random_ode_param_variance': trial.suggest_float(
            'random_ode_param_variance',
            0.0,
            args.max_random_ode_param_variance,
        ),
        'random_initial_state_variance': trial.suggest_float(
            'random_initial_state_variance',
            0.0,
            args.max_random_initial_state_variance,
        ),
        'observation_horizon': observation_horizon,
        'time_step': args.time_step,
        'reward_weights': {
            'biomass_gain': step_reward_weight,  # Scale up small product changes
            'dot_penalty': step_reward_weight,  # Multiplier for the violation magnitude
            'acetate_penalty': 0.  # acetate accumulation penalty
        }
    }

    # 2. CREATE TRAINING ENV
    train_env = make_vec_env(env_id, n_envs=n_parallel_envs, vec_env_cls=SubprocVecEnv, env_kwargs=env_kwargs)

    # WRAP TRAINING ENV
    train_env = VecNormalize(
        train_env,
        training=True,  # Update stats (mean/std)
        norm_obs=True,  # Normalize observations
        norm_reward=True,  # Normalize rewards (helps Agent learn)
        clip_obs=10.,
        gamma=gamma
    )

    # 3. CREATE EVAL ENV
    eval_env_kwargs = {
        'random_ode_param_variance': args.evaluation_ode_parameter_variance,
        'random_initial_state_variance': args.evaluation_initial_state_variance,
        'observation_horizon': observation_horizon,
        'time_step': args.time_step,

    }
    eval_env = make_vec_env(
        env_id,
        n_envs=args.num_evaluation_envs,
        vec_env_cls=SubprocVecEnv,
        env_kwargs=eval_env_kwargs,
    )

    # WRAP EVAL ENV
    eval_env = VecNormalize(
        eval_env,
        training=False,  # DO NOT update stats. We will sync them from train_env.
        norm_obs=True,  # Normalize observations
        norm_reward=False,  # DO NOT normalize rewards. We want real Div/L performance metric.
        clip_obs=10.,
        gamma=gamma
    )

    # 4. Sample remaining hyperparameters
    try:
        # Pass the gamma we already sampled
        model_params = sample_ppo_hyperparameters(trial, vars(args), fixed_gamma=gamma)
    except optuna.exceptions.TrialPruned:
        train_env.close()
        eval_env.close()
        raise

    model = PPO(
        "MlpPolicy",
        train_env,
        tensorboard_log=args.log_dir,
        **model_params,
    )

    # 5. Pass train_env to Callback for Synchronization
    eval_callback = TrialEvalCallback(
        eval_env,
        trial,
        train_env_to_sync=train_env,  # Pass train_env for normalization sync
        n_eval_episodes=n_eval_episodes,
        eval_freq=eval_freq_n_envs,
        deterministic=True,
    )

    try:
        model.learn(total_timesteps=n_timesteps, callback=eval_callback, reset_num_timesteps=True)
    except AssertionError as e:
        print(f"Trial {trial.number} failed: {e}")
        train_env.close()
        eval_env.close()
        return float('nan')

    train_env.close()
    eval_env.close()

    if eval_callback.is_pruned:
        raise optuna.exceptions.TrialPruned()

    return eval_callback.last_mean_reward

def main():
    parser = argparse.ArgumentParser(description="Optuna Hyperparameter Tuning for PPO agent.")

    # Core control arguments that were previously hardcoded globals
    parser.add_argument(
        "--env-id",
        dest="env_id",
        type=str,
        default=DEFAULT_ENV_ID,
        help="Gymnasium environment ID to tune (default: ObservationEcoli-v0).",
    )
    parser.add_argument(
        "--n-eval-episodes",
        dest="n_eval_episodes",
        type=int,
        default=DEFAULT_N_EVAL_EPISODES,
        help="Number of evaluation episodes per evaluation step.",
    )
    parser.add_argument(
        "--n-parallel-envs",
        dest="n_parallel_envs",
        type=int,
        default=DEFAULT_N_PARALLEL_ENVS,
        help="Number of parallel environments to use during training.",
    )
    parser.add_argument(
        "--n-timesteps",
        dest="n_timesteps",
        type=int,
        default=DEFAULT_N_TIMESTEPS,
        help="Total training timesteps per Optuna trial.",
    )

    # Only arguments actually used in this script
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device for the neural network (cpu or cuda).",
    )

    # Environment-specific upper bounds for tuning ranges
    parser.add_argument(
        "--max-observation-horizon",
        dest="max_observation_horizon",
        type=int,
        default=1,
        help="Maximum observation horizon to explore.",
    )
    parser.add_argument(
        "--max-random-ode-param-variance",
        dest="max_random_ode_param_variance",
        type=float,
        default=0.5,
        help="Maximum random ODE parameter variance to explore.",
    )
    parser.add_argument(
        "--max-random-initial-state-variance",
        dest="max_random_initial_state_variance",
        type=float,
        default=0.5,
        help="Maximum variance for perturbing the initial states to explore.",
    )
    parser.add_argument(
        "--max-step-reward-weight",
        dest="max_step_reward_weight",
        type=float,
        default=0.01,
        help="Maximum of the factor of between step rewards",
    )

    parser.add_argument(
        "--evaluation-ode-parameter-variance",
        dest="evaluation_ode_parameter_variance",
        type=float,
        default=0.5,
        help="ODE parameter variance for evaluation.",
    )
    parser.add_argument(
        "--evaluation-initial-state-variance",
        dest="evaluation_initial_state_variance",
        type=float,
        default=0.5,
        help="Initial state variance for the evaluation environments.",
    )
    parser.add_argument(
        "--num-evaluation-envs",
        dest="num_evaluation_envs",
        type=int,
        default=10,
        help="How many evaluation environment should be used.",
    )
    parser.add_argument(
        "--time-step",
        dest="time_step",
        type=float,
        default=1.0,
        help="Hours between two consecutive actions.",
    )

    # Logging and tuning arguments
    parser.add_argument(
        "--log-dir",
        dest="log_dir",
        type=str,
        default="/tmp/optuna_logs/",
        help="Tensorboard log directory.",
    )
    parser.add_argument(
        "--eval-freq",
        dest="eval_freq",
        type=int,
        default=10000,
        help="Evaluate the agent every N steps (unadjusted).",
    )
    parser.add_argument(
        "--n-trials",
        dest="n_trials",
        type=int,
        default=100,
        help="Number of Optuna trials to run.",
    )

    args = parser.parse_args()

    # --- Optuna Study Setup ---
    # We maximize the mean reward
    sampler = TPESampler(seed=42)
    # Pruner stops unpromising trials early
    pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=10)

    study = optuna.create_study(
        study_name=f"PPO_Hyperparams_{args.env_id}",
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
    )

    print(f"--- Starting Optuna Study for {args.env_id} with {args.n_trials} trials ---")
    start = time.time()

    study.optimize(lambda trial: objective(trial, args), n_trials=args.n_trials, show_progress_bar=True)

    # --- Results ---
    print("\n--- Hyperparameter Tuning Results ---")
    print(f"Number of finished trials: {len(study.trials)}")
    print(f"Best trial mean reward: {study.best_value:.4f}")
    print("Best hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")

    print("Time taken: {:.2f} minutes".format((time.time() - start) / 60))


if __name__ == "__main__":
    main()