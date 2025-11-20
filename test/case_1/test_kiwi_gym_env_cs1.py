import numpy as np
import gymnasium as gym
from gymnasium.utils.env_checker import check_env
from Case1 import KiwiGym_env_CS1  # ensures registration of `kiwiGym-CS1`

def test_kiwigym_cs1_env_basic():
    env = gym.make('kiwiGym-CS1')
    # validate standard Gym interface
    check_env(env.unwrapped)

    obs, info = env.reset(seed=0)
    assert obs is not None
    # observation shape matches the declared observation_space
    assert hasattr(env.observation_space, "shape")
    assert np.asarray(obs).shape == env.observation_space.shape

    # perform a few steps with valid integer actions
    for _ in range(10):
        action = int(env.action_space.sample())
        obs, reward, terminated, truncated, info = env.step(action)

        assert isinstance(reward, (int, float))
        assert np.asarray(obs).shape == env.observation_space.shape

        if terminated or truncated:
            obs, info = env.reset()

    env.close()


def test_kiwigym_cs1_env_with_different_parameters():
    # given
    other_pameters = np.array([0.4] * 18)

    env = gym.make('kiwiGym-CS1', initial_model_parameters=other_pameters)
    # validate standard Gym interface
    check_env(env.unwrapped)

    obs, info = env.reset(seed=0)
    assert obs is not None
    # observation shape matches the declared observation_space
    assert hasattr(env.observation_space, "shape")
    assert np.asarray(obs).shape == env.observation_space.shape

    # perform a few steps with valid integer actions
    for _ in range(10):
        action = int(env.action_space.sample())
        obs, reward, terminated, truncated, info = env.step(action)

        assert isinstance(reward, (int, float))
        assert np.asarray(obs).shape == env.observation_space.shape

        if terminated or truncated:
            obs, info = env.reset()

    env.close()

    assert np.all(obs >= 0.0)  # all concentrations non-negative


def test_kiwigym_cs1_env_with_different_offsets():
    # given
    env = gym.make(
        'kiwiGym-CS1',
        sample_offsets=[0.33],
    )
    # validate standard Gym interface
    check_env(env.unwrapped)

    obs, info = env.reset(seed=0)
    assert obs is not None
    # observation shape matches the declared observation_space
    assert hasattr(env.observation_space, "shape")
    assert np.asarray(obs).shape == env.observation_space.shape

    # perform a few steps with valid integer actions
    for _ in range(10):
        action = int(env.action_space.sample())
        obs, reward, terminated, truncated, info = env.step(action)

        assert isinstance(reward, (int, float))
        assert np.asarray(obs).shape == env.observation_space.shape

        if terminated or truncated:
            obs, info = env.reset()

    env.close()

    assert np.all(obs >= 0.0)  # all concentrations non-negative
