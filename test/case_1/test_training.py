import os
import argparse
import pytest

# Import the function to be tested
from Case1.Train_Case1 import train_agent

# The environments must be imported to be registered with Gym
import KiwiGym_env_CS1
from Case1.observation_env import ObservationEcoliEnv


def test_short_training_run(tmp_path):
    """
    Tests that train_agent runs for a few steps without crashing using pytest.
    The tmp_path fixture provides a temporary directory for test artifacts.
    """
    # Define paths for logs and saved models within the temporary directory
    save_path = tmp_path / "saved_models/"
    log_dir = tmp_path / "logs/"

    # Mock arguments for a quick test run
    args = argparse.Namespace(
        total_timesteps=32,
        n_parallel=1,
        learning_rate=0.001,
        random_ode_param_variance=0.2,
        random_initial_state_variance=0.2,
        ent_coef=0.001,
        n_steps=16,
        batch_size=16,
        device="cpu",
        num_neurons=32,
        log_dir=str(log_dir),
        save_path=str(save_path),
        checkpoint_freq=50,
        eval_freq=50,
        observation_horizon=1,
        time_step=1.0,
    )

    env_id = "kiwiGym-CS1"

    try:
        # Run the training function
        train_agent(env_id, args)
    except Exception as e:
        pytest.fail(f"train_agent raised an exception unexpectedly: {e}")

    # Check if the final model file was created
    final_model_path = os.path.join(save_path, env_id, f"PPO_{env_id}_final_model.zip")
    assert os.path.exists(final_model_path), "Final model file was not created."


def test_observation_ecoli_training(tmp_path):
    """
    Tests that train_agent runs for a few steps without crashing using pytest.
    The tmp_path fixture provides a temporary directory for test artifacts.
    """
    # Define paths for logs and saved models within the temporary directory
    save_path = tmp_path / "saved_models/"
    log_dir = tmp_path / "logs/"

    # Mock arguments for a quick test run
    args = argparse.Namespace(
        total_timesteps=32,
        n_parallel=1,
        learning_rate=0.001,
        random_ode_param_variance=0.2,
        random_initial_state_variance=0.2,
        ent_coef=0.001,
        n_steps=16,
        batch_size=16,
        device="cpu",
        num_neurons=32,
        log_dir=str(log_dir),
        save_path=str(save_path),
        checkpoint_freq=50,
        eval_freq=50,
        observation_horizon=1,
        time_step=1.0,
    )

    env_id = 'ObservationEcoli-v0'

    # Run the training function
    train_agent(env_id, args)

    # Check if the final model file was created
    final_model_path = os.path.join(save_path, env_id, f"PPO_{env_id}_final_model.zip")
    assert os.path.exists(final_model_path), "Final model file was not created."
