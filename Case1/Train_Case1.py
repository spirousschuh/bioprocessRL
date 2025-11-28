import os
import time

import KiwiGym_env_CS1_0
import KiwiGym_env_CS1
import Case1.observation_env

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv


os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import os
import argparse
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback

# These imports are needed for the environments to be registered with gym
import KiwiGym_env_CS1_0
import KiwiGym_env_CS1

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def setup_environment(env_id, args, is_eval=False):
    """Creates and wraps the gym environment."""
    # For evaluation, we usually only need a single environment
    env_n = 1 if is_eval else args.n_parallel

    # Use SubprocVecEnv for parallel training, but a DummyVecEnv for evaluation
    vec_env_class = SubprocVecEnv if env_n > 1 else None

    # Pass custom arguments to the environment constructor
    env_kwargs = {
        'random_ode_param_variance': args.random_ode_param_variance if not is_eval else 0.,
        'random_initial_state_variance': args.random_initial_state_variance if not is_eval else 0.,
        'observation_horizon': args.observation_horizon,
        'time_step': args.time_step,
    }

    return make_vec_env(
        env_id,
        n_envs=env_n,
        vec_env_cls=vec_env_class,
        env_kwargs=env_kwargs
    )


def create_model(env, args):
    """Creates the PPO model with specified hyperparameters."""


    return PPO(
        "MlpPolicy",
        env,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        ent_coef=args.ent_coef,
        policy_kwargs=dict(net_arch=[args.num_neurons, args.num_neurons]),
        verbose=1,
        tensorboard_log=args.log_dir,
        device=args.device,
    )


def train_agent(env_id, args):
    """Sets up and trains the PPO agent."""

    # --- Setup ---
    run_name = f"PPO_{env_id}"
    save_dir = os.path.join(args.save_path, env_id)
    os.makedirs(save_dir, exist_ok=True)

    # Training environment
    train_env = setup_environment(
        env_id,
        args,
    )

    # Evaluation environment
    eval_env = setup_environment(env_id, args, is_eval=True)

    # PPO Model
    model = create_model(train_env, args)

    # --- Callbacks ---
    # Checkpoint callback to save model periodically
    checkpoint_callback = CheckpointCallback(
        save_freq=max(args.checkpoint_freq // args.n_parallel, 1),
        save_path=save_dir,
        name_prefix="ppo_model"
    )

    # Evaluation callback to save the best model
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=save_dir,
        log_path=save_dir,
        eval_freq=max(args.eval_freq // args.n_parallel, 1),
        deterministic=True,
        render=False
    )

    # --- Training ---
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=[checkpoint_callback, eval_callback],
        tb_log_name=run_name,
        reset_num_timesteps=False
    )

    # Save the final model
    model.save(os.path.join(save_dir, f"{run_name}_final_model.zip"))
    train_env.close()
    eval_env.close()


def main():
    parser = argparse.ArgumentParser(description="Train PPO agent for kiwiGym environments.")

    # Training arguments
    parser.add_argument("--total-timesteps", dest="total_timesteps", type=int, default=100000, help="Total timesteps for training.")
    parser.add_argument("--n-parallel", dest="n_parallel", type=int, default=10, help="Number of parallel environments.")
    parser.add_argument("--learning-rate", dest="learning_rate", type=float, default=0.001, help="Learning rate.")
    parser.add_argument("--ent-coef", dest="ent_coef", type=float, default=0.001, help="Entropy coefficient.")
    parser.add_argument("--n-steps", dest="n_steps", type=int, default=10,
                        help="Number of steps to run for each environment per update.")
    parser.add_argument("--batch-size", dest="batch_size", type=int, default=50, help="Minibatch size.")
    parser.add_argument("--device", type=str, default="cpu", help="Device for the neural network (cpu or cuda).")
    parser.add_argument("--num-neurons", dest="num_neurons", type=int, default=128, help="Number of neurons in each hidden layer.")

    # Environment-specific arguments
    parser.add_argument("--observation-horizon", dest="observation_horizon", type=int, default=1)
    parser.add_argument("--random-ode-param-variance", dest="random_ode_param_variance", type=float, default=0., help="Using random_ode_param_variance for sampling random ODE parameters.")
    parser.add_argument("--random-initial-state-variance", dest="random_initial_state_variance", type=float, default=0., help="Variance for perturbing the initial states.")
    parser.add_argument("--time-step", dest="time_step", type=float, default=1., help="Hours between two consecutive actions.")

    # Logging and saving arguments
    parser.add_argument("--log-dir", dest="log_dir", type=str, default="/tmp/logs/ppo/", help="Tensorboard log directory.")
    parser.add_argument("--save-path", dest="save_path", type=str, default="./saved_models/", help="Path to save models.")
    parser.add_argument("--checkpoint-freq", dest="checkpoint_freq", type=int, default=20000, help="Save a checkpoint every N steps.")
    parser.add_argument("--eval-freq", dest="eval_freq", type=int, default=10000, help="Evaluate the agent every N steps.")
    parser.add_argument("--model-name", dest="model_name", type=str, default="kiwiGym-CS1", help="The name of the model to be trained.")

    args = parser.parse_args()

    # Default PPO n_steps and batch_size are often good starting points
    # If you want to keep your original logic:
    # step_per_episode = 10
    # eps_envs = 2
    # args.n_steps = step_per_episode * eps_envs
    # args.batch_size = args.n_steps * args.n_parallel

    model_name = args.model_name
    start = time.time()
    print(f"--- Starting training for {model_name} ---")
    train_agent(model_name, args)
    print(f"--- Finished training for {model_name} ---")
    print("Time taken: {:.2f} minutes".format((time.time() - start) / 60))

if __name__ == "__main__":
    main()
