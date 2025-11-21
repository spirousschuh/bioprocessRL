import os
import time
import argparse
from typing import Any, Dict

# Stable-Baselines3 imports
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement

# Optuna imports
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

# Environment imports (must be present for gym.make to work)
import KiwiGym_env_CS1_0
import KiwiGym_env_CS1
import Case1.observation_env

# --- Default Global Configuration (can be overridden via CLI) ---
DEFAULT_ENV_ID = 'ObservationEcoli-v0'
DEFAULT_N_EVAL_EPISODES = 5   # Number of episodes for evaluation
DEFAULT_N_PARALLEL_ENVS = 10  # Number of parallel environments for training
DEFAULT_N_TIMESTEPS = 100000  # Total timesteps per Optuna trial

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class TrialEvalCallback(EvalCallback):
    """Callback that reports the mean reward to Optuna for pruning."""

    def __init__(self, eval_env, trial, n_eval_episodes=5, eval_freq=10000,
                 deterministic=True, verbose=0):
        # Create the inner callback for early stopping
        early_stop_cb = StopTrainingOnNoModelImprovement(
            max_no_improvement_evals=5,
            min_evals=10,
            verbose=0,
        )
        super().__init__(
            eval_env,
            best_model_save_path=None,  # Optuna handles best model saving logic
            log_path=None,
            eval_freq=eval_freq,
            deterministic=deterministic,
            render=False,
            verbose=verbose,
            n_eval_episodes=n_eval_episodes,
            callback_after_eval=early_stop_cb,
        )
        self.trial = trial
        self.last_mean_reward = -float('inf')
        self.is_pruned = False  # be explicit

    def _on_step(self) -> bool:
        # Let EvalCallback do its evaluation & early-stopping callback
        continue_training = super()._on_step()
        if not continue_training:
            return False

        # Report intermediate value to Optuna after each evaluation
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            # EvalCallback stores mean reward in self.last_mean_reward
            self.trial.report(self.last_mean_reward, self.num_timesteps)

            # Ask Optuna whether to prune
            if self.trial.should_prune():
                self.is_pruned = True
                print(f"Trial {self.trial.number} pruned at step {self.num_timesteps}.")
                return False

        return True


def sample_ppo_hyperparameters(trial: optuna.Trial, env_args: Dict[str, Any]) -> Dict[str, Any]:
    """Samples hyperparameters for PPO using Optuna's trial object."""

    # We suggest hyperparameters based on common RL search ranges
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
    n_steps = trial.suggest_categorical('n_steps', [50, 100, 200, 400, 800, 1600])
    batch_size = trial.suggest_categorical('batch_size', [50, 100, 200, 400, 800])
    gamma = trial.suggest_float('gamma', 0.9, 0.9999, log=True)
    ent_coef = trial.suggest_float('ent_coef', 1e-6, 5e-2, log=True)

    # Architecture
    num_neurons = trial.suggest_categorical('num_neurons', [64, 128, 256])

    # Policy kwargs are passed as a dictionary
    policy_kwargs = dict(net_arch=[num_neurons, num_neurons])

    # Check if batch_size is smaller than n_steps
    if batch_size > n_steps:
        # Invalid configuration, skip this trial
        raise optuna.exceptions.TrialPruned()

    # The returned dictionary is passed to the PPO constructor
    return {
        "learning_rate": learning_rate,
        "n_steps": n_steps,
        "batch_size": batch_size,
        "gamma": gamma,
        "ent_coef": ent_coef,
        "policy_kwargs": policy_kwargs,
        "device": env_args['device'],
        "verbose": 0,
    }


def objective(trial: optuna.Trial, args: argparse.Namespace) -> float:
    """Objective function for Optuna. Trains an agent and returns mean reward."""

    # Derive frequently used values from args
    env_id = args.env_id
    n_parallel_envs = args.n_parallel_envs
    n_eval_episodes = args.n_eval_episodes
    n_timesteps = args.n_timesteps

    eval_freq_n_envs = max(args.eval_freq // n_parallel_envs, 1)

    # Sample environment randomness and observation horizon using upper-bound args
    random_ode_param_variance = trial.suggest_float(
        'random_ode_param_variance',
        0.0,
        args.max_random_ode_param_variance,
    )
    random_initial_state_variance = trial.suggest_float(
        'random_initial_state_variance',
        0.0,
        args.max_random_initial_state_variance,
    )
    observation_horizon = trial.suggest_int(
        'observation_horizon',
        1,
        args.max_observation_horizon,
    )

    env_kwargs = {
        'random_ode_param_variance': random_ode_param_variance,
        'random_initial_state_variance': random_initial_state_variance,
        'observation_horizon': observation_horizon,
        'time_step': args.time_step,
    }

    train_env = make_vec_env(
        env_id,
        n_envs=n_parallel_envs,
        vec_env_cls=SubprocVecEnv,
        env_kwargs=env_kwargs,
    )
    eval_env = make_vec_env(
        env_id,
        n_envs=2,
        env_kwargs={
            'random_ode_param_variance': 0.0,
            'random_initial_state_variance': 0.0,
            'observation_horizon': observation_horizon,
            'time_step': args.time_step,
        },
    )

    # 3. Sample PPO Hyperparameters
    try:
        model_params = sample_ppo_hyperparameters(trial, vars(args))
    except optuna.exceptions.TrialPruned:
        # Pruning due to bad hyperparameter combination (e.g., batch_size > n_steps)
        train_env.close()
        eval_env.close()
        raise

    # 4. Create PPO Model
    model = PPO(
        "MlpPolicy",
        train_env,
        tensorboard_log=args.log_dir,
        **model_params,
    )

    # 5. Create Optuna-specific Callback
    # We use the custom TrialEvalCallback to report results and allow pruning
    eval_callback = TrialEvalCallback(
        eval_env,
        trial,
        n_eval_episodes=n_eval_episodes,
        eval_freq=eval_freq_n_envs,
        deterministic=True,
    )

    # 6. Train the Agent
    try:
        model.learn(
            total_timesteps=n_timesteps,
            callback=eval_callback,
            reset_num_timesteps=True,
        )
    except AssertionError as e:
        # Catch common SB3 errors during training (e.g., NaN losses) and mark as failed
        print(f"Trial {trial.number} failed due to training error: {e}")
        train_env.close()
        eval_env.close()
        return float('nan')  # Return NaN to signal a failed trial

    train_env.close()
    eval_env.close()

    # 7. Check if the trial was pruned
    if eval_callback.is_pruned:
        raise optuna.exceptions.TrialPruned()

    # Optuna maximizes the objective value, so we return the final mean reward
    # The EvalCallback saves the last mean reward to self.last_mean_reward
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