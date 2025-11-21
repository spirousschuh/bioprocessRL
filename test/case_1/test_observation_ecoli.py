import numpy as np
from gymnasium.utils.env_checker import check_env
from sympy.core.benchmarks.bench_sympify import timeit_sympify_1

from Case1.observation_env import ObservationEcoliEnv
import gymnasium as gym

def test_observation_ecoli():
    # given
    env = gym.make(
        'ObservationEcoli-v0',
        sample_offset=0.33,
        observation_horizon=2,
        time_step=0.5,
    )
    # validate standard Gym interface
    check_env(env.unwrapped)

    obs, info = env.reset(seed=0)
    assert obs is not None
    # observation shape matches the declared observation_space
    assert hasattr(env.observation_space, "shape")
    assert np.asarray(obs).shape == env.observation_space.shape

    # perform a few steps with valid integer actions
    counter = 0
    for _ in range(20):
        action = int(env.action_space.sample())
        obs, reward, terminated, truncated, info = env.step(action)
        print(np.max(obs))
        counter += 1
        assert isinstance(reward, (int, float))
        assert np.asarray(obs).shape == env.observation_space.shape

        if terminated or truncated:
            break

    env.close()

    assert np.all(obs >= 0.0)  # all concentrations non-negative
    assert counter == 20  # ran full 20 steps without termination
