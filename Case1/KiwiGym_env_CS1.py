"""Gym wrapper for the Case1 kiwi bioprocess simulation.

This module registers and exposes the `kiwiGym` simulation as a Gymnasium
environment named `kiwiGym-CS1` and implements the thin Gym wrapper class
`kiwiGymEnv_CS1` that adapts the simulator to the Gymnasium API.

Notes:
- The wrapper builds an observation vector that combines timing, previous
  feed actions and recent sampled/sensor measurements.
- This file focuses on adapting the simulation to Gym; the actual dynamics
  are implemented in `Case1.kiwiGym_CS1` and `Case1.method_kiwiGym`.
"""

import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register
from gymnasium.utils.env_checker import check_env

import numpy as np

from Case1.kiwiGym_CS1 import kiwiGym
from Case1.kiwiGym_CS1 import DEFAULT_ODE_PARAMETERS
from Case1.kiwiGym_CS1 import DEFAULT_INITIAL_STATES

# %%
register(
    id='kiwiGym-CS1',                                
    entry_point='Case1.KiwiGym_env_CS1:kiwiGymEnv_CS1',
)


class kiwiGymEnv_CS1(gym.Env):
    """Gymnasium environment wrapper around Case1.kiwiGym_CS1.kiwiGym.

    The wrapper exposes a discrete action space where each action is an index
    into a set of possible feed adjustments. The observation is a flattened
    vector that includes timing, previously applied actions and recent
    measurements normalized by an upper-bound vector.
    """

    # Annotate the simulator attribute at class level for static analysis tools
    kiwiGym: kiwiGym

    metadata = {"render_modes": ["human"], 'render_fps': 4}


    def __init__(
            self,
            initial_model_parameters=None,
            num_experiments=1,
            random_initial_state_variance=0.,
            random_ode_param_variance=0.33,
            ode_param_perturbation_type='gamma',
            render_mode=None,
            sample_offsets=None,
            feed_std=0.0,
            feed_to_zero_probability=0.0,
            **kwargs,
    ):
        """Create the Gym wrapper and build observation/action spaces.

        Args:
            render_mode: optional rendering mode passed to the underlying env.
        """
        self.render_mode = render_mode

        # Create the underlying simulation object (annotated for type checkers)
        self.kiwiGym: kiwiGym = kiwiGym(
            initial_model_parameters=initial_model_parameters,
            num_experiments=num_experiments,
            sample_offsets=sample_offsets,
        )

        self.random_ode_param_variance = random_ode_param_variance
        self.ode_param_perturbation_type = ode_param_perturbation_type
        self.random_initial_state_variance = random_initial_state_variance
        self.feed_std = feed_std
        self.feed_to_zero_probability = feed_to_zero_probability

        # Allowed discrete action values (feed adjustment set); actions map
        # to indices into this array.
        self.action_values = np.arange(-5, 5.5, 0.5)
        self.action_space = spaces.Discrete(
            len(self.action_values) * self.kiwiGym.num_experiments
        )  # Actions are indices across experiments

        # Build an upper-bound vector used to normalize observations. The
        # upper-bound vector concatenates three blocks:
        #  - a short timing block (e_vector)
        #  - a block for stored past actions (d_vector)
        #  - a block for measurement upper-bounds (y_vector)
        timing_block = np.array([14])
        past_actions_block = np.tile(
            [21],
            (self.kiwiGym.final_time - int(self.kiwiGym.pulse_times[0] - 0)) * self.kiwiGym.num_experiments,
        )
        measurement_upper_block = np.tile(
            [20] + [105] * 1,
            (self.kiwiGym.final_time) * self.kiwiGym.num_experiments,
        )
        self.observation_upper_bound = np.concatenate(
            [timing_block, past_actions_block, measurement_upper_block]
        )
        # Observation space is normalized between 0 and 1 (we perform normalization
        # by dividing by observation_upper_bound before returning observations).
        self.observation_space = spaces.Box(
            low=(0) * np.ones(len(self.observation_upper_bound)),
            high=(1) * np.ones(len(self.observation_upper_bound)),
            dtype=np.float64,
        )

    def reset(self, seed=None, options=None):
        """Reset environment and the underlying simulation.

        The wrapper draws a randomized model parameter vector around a mean,
        resets the simulation and integrates up to the first pulse batch to
        build the initial observation state.

        Returns:
            obs_corrected: normalized observation vector (numpy array)
            info: empty dict per Gym API
        """
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)

        # Base model parameters and randomized perturbation used to initialize
        # the simulation (keeps prior behaviour of the project).
        base_model_parameters = np.array(DEFAULT_ODE_PARAMETERS)
        if self.random_ode_param_variance > 0.0:
            if self.ode_param_perturbation_type == "gamma":
                randomized_model_parameters = base_model_parameters * np.random.gamma(
                    1. / self.random_ode_param_variance,
                    self.random_ode_param_variance,
                    len(base_model_parameters),
                )
            elif self.ode_param_perturbation_type == "uniform":
                randomized_model_parameters = base_model_parameters * np.random.uniform(
                    1. - self.random_ode_param_variance * 3 ** 0.5,
                    1. + self.random_ode_param_variance * 3 ** 0.5,
                    len(base_model_parameters),
                )
        else:
            randomized_model_parameters = base_model_parameters

        if self.random_initial_state_variance > 0.0:
            randomized_initial_states = DEFAULT_INITIAL_STATES * np.concatenate(
                [
                    np.random.gamma(
                        1. / self.random_initial_state_variance,
                        self.random_initial_state_variance,
                        # just perturb the first two states (X and S)
                        2,
                    ),
                    [1.0, 1.0, 1.0, 1.0],
                ],
                axis=0
            )
        else:
            randomized_initial_states = DEFAULT_INITIAL_STATES



        # Reset the underlying simulation. (Keep kwarg name `model_parameters`
        # to preserve existing call sites in the repository.)
        self.kiwiGym.reset(
            seed=seed,
            model_parameters=randomized_model_parameters,
            initial_states=randomized_initial_states,
        )

        # Initialize the flattened observation array (we will fill parts of it)
        observation_array = np.zeros(len(self.observation_upper_bound), dtype=float)

        # The first element encodes the number of steps before the pulse sequence
        pre_pulse_offset = 1
        observation_array[0] = int(self.kiwiGym.pulse_times[0] - pre_pulse_offset)  # no encoding

        # Integrate the simulator up to the first pulse time so that we have
        # a consistent initial set of measurements stored in the observation.
        # We use the same action value that the original script used.
        initial_action_value = [10]
        while self.kiwiGym.current_time < round(self.kiwiGym.pulse_times[0] - pre_pulse_offset):
            raw_observation_vector, _, _ = self.kiwiGym.perform_action(initial_action_value)
            # Compute target index range in the flattened observation array and
            # store the simulator-provided raw observation values there.
            observation_index_range = (
                np.arange(2 * self.kiwiGym.num_experiments)
                + 2 * self.kiwiGym.num_experiments * (self.kiwiGym.current_time - 1)
                + (self.kiwiGym.num_experiments) * (self.kiwiGym.final_time - int(self.kiwiGym.pulse_times[0]))
                + 1
            )
            observation_array[observation_index_range] = raw_observation_vector

        # Save the raw (unnormalized) observation on the wrapper instance
        self.obs = observation_array

        info = {}
        # Normalize by the observation upper-bound vector and return
        obs_corrected = observation_array / self.observation_upper_bound
        return obs_corrected, info

    def step(self, action):
        """Apply an action (index) mapped to a feed adjustment and step the sim.

        Args:
            action: integer index into the discrete action space

        Returns:
            obs_corrected: normalized observation after this action
            reward: scalar reward returned by the simulation
            terminated: boolean episode termination flag
            truncated: fixed False (no truncation logic)
            info: empty dict per Gym API
        """
        # Map action index to the numeric feed adjustment value and apply
        applied_action_value = self.action_values[action]
        raw_observation_vector, reward, terminated = self.kiwiGym.perform_action(
            applied_action_value,
            feed_std=self.feed_std,
            feed_to_zero_probability=self.feed_to_zero_probability,
        )

        # Retrieve the stored observation buffer and update its fields
        obs_array = self.obs
        # first element stores the current time in the wrapper's convention
        obs_array[0] = self.kiwiGym.current_time

        # Write the action index into the 'past actions' region of the observation
        action_index_range = (
            np.arange(self.kiwiGym.num_experiments)
            + self.kiwiGym.num_experiments * (self.kiwiGym.current_time - int(self.kiwiGym.pulse_times[0]) - 0)
            + 1
        )
        obs_array[action_index_range] = np.array(action) + 1

        # Write the latest raw observation block coming from the simulator
        observation_index_range = (
            np.arange(2 * self.kiwiGym.num_experiments)
            + 2 * self.kiwiGym.num_experiments * (self.kiwiGym.current_time - 1)
            + (self.kiwiGym.num_experiments) * (self.kiwiGym.final_time - int(self.kiwiGym.pulse_times[0] - 0))
            + 1
        )
        obs_array[observation_index_range] = raw_observation_vector

        # Persist the buffer back to the wrapper
        self.obs = obs_array

        # Render when requested and the episode just terminated
        if (self.render_mode == 'human') and (terminated is True):
            print('Action: ', applied_action_value)
            self.render()

        # Normalize using the upper bound vector (ensure float division)
        obs_corrected = obs_array.astype(np.float64) / self.observation_upper_bound.astype(np.float64)
        return obs_corrected, reward, terminated, False, {}

    def render(self):
        """Delegate rendering to the underlying kiwiGym simulator."""
        self.kiwiGym.render()

# %% For unit testing
if __name__ == "__main__":
    env = gym.make('kiwiGym-CS1')

    print("Check environment begin")
    check_env(env.unwrapped)
    print("Check environment end")

    # Reset environment
    obs = env.reset()[0]

    cnt = 0
    while cnt < 3:
        rand_action = [10]
        obs, reward, terminated, _, _ = env.step(rand_action)
        print(reward, rand_action)

        if terminated:
            print(env.unwrapped.kiwiGym.model_parameters[0])
            env.render()
            
            model_parameters = np.array([
                1.2578 * (1 + (np.random.random(1)[0] - .5) / 2),
                0.43041,
                0.6439,
                2.2048 * 0 + 7.0767,
                0.4063,
                0.1143 * 4,
                0.1848 * 4,
                287.74 * 0 + .4242,
                1.586 * .7,
                1.5874 * .7,
                0.3322 * .75,
                0.0371,
                0.0818,
                +9000,
                .1,
                5,
            ] + [850] * 3 + [90] * 3)
            obs = env.reset()[0]
        cnt += 1
