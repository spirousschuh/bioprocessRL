import gymnasium as gym
import numpy as np
from gymnasium import spaces, register
from copy import deepcopy
import matplotlib.pyplot as plt

from Case1 import method_kiwiGym

# --- Constants ---

DEFAULT_ODE_PARAMETERS = [
        # --- Kinetic Parameters ---
        1.2578,      # thetas[0]: qs_max - Max substrate uptake rate
        0.43041,     # thetas[1]: fracc_q_ox_max - Fraction of max oxidative quotient
        0.6439,      # thetas[2]: qa_max - Max by-product production rate
        7.0767,      # thetas[3]: Ksi - Substrate inhibition constant
        0.4063,      # thetas[4]: Ys_ox - Yield of substrate to oxygen
        0.1143 * 4,  # thetas[5]: Ya_p - Yield of by-product to product
        0.1848 * 4,  # thetas[6]: Ya_c - Yield of by-product to cells
        0.4242,      # thetas[7]: Kai - By-product inhibition constant
        1.586 * 0.7,   # thetas[8]: Yo_ox - Yield of oxygen (oxidative)
        1.5874 * 0.7,  # thetas[9]: Yo_a - Yield of oxygen to by-product
        0.3322 * 0.75, # thetas[10]: Yxs_of - Yield of cells to substrate (overflow)
        0.0371,      # thetas[11]: Ks - Substrate saturation constant
        0.0818,      # thetas[12]: Ka - By-product saturation constant
        9000,        # thetas[13]: ky_1 - Yield-related parameter 1
        0.1,         # thetas[14]: ky_2 - Yield-related parameter 2
        5,           # thetas[15]: ky_3 - Yield-related parameter 3
        850,        # thetas[16]: kla - kLa value
        90          # thetas[17]: k_sensor - Sensor constant
    ]


DEFAULT_INITIAL_STATES = [0.18, 4, 0, 100, 0, .01]  # Biomass, Substrate, By-product, DOT, Product, Volume



register(
    id='ObservationEcoli-v0',
    entry_point='Case1.observation_env:ObservationEcoliEnv',
)



class ObservationEcoliEnv(gym.Env):
    """
    A unified Gymnasium environment for the Kiwi bioprocess simulation.

    This class merges the Gym wrapper and the core simulator logic into a
    single, self-contained environment. It handles the simulation dynamics,
    state management, and interaction with the Gym API.
    """
    metadata = {"render_modes": ["human"], 'render_fps': 4}

    def __init__(
            self,
            initial_model_parameters=None,
            random_initial_state_variance=0.,
            random_ode_param_variance=0.33,
            render_mode=None,
            final_time=14,
            time_step=1,
            time_batch=5,
            mu_ref=0.145,
            sampling_times_per_hour=25,
            sample_offset=0.99,
            observation_horizon: int=1,

    ):
        super().__init__()

        # --- Configuration ---
        self.num_experiments = 1
        self.random_ode_param_variance = random_ode_param_variance
        self.random_initial_state_variance = random_initial_state_variance
        self.render_mode = render_mode
        self.final_time = final_time
        self.time_step = time_step
        self.current_time = 0

        # --- Model and State ---
        self.model_parameters = np.array(
            initial_model_parameters if initial_model_parameters is not None else DEFAULT_ODE_PARAMETERS
        )

        self.base_model_parameters = (
            np.array(initial_model_parameters)
            if initial_model_parameters is not None
            else np.array(DEFAULT_ODE_PARAMETERS)
        ).astype(float)
        # this will hold per-episode parameters
        self.model_parameters = self.base_model_parameters.copy()

        self.initial_states = np.array(DEFAULT_INITIAL_STATES)
        self.state = {}

        # --- Action Space ---
        self.action_values = np.arange(-5, 5.5, 0.5)
        self.action_space = spaces.Discrete(len(self.action_values))

        self.observation_lengths = {
            0: observation_horizon, # Biomass
            3: int(sampling_times_per_hour * observation_horizon), # DOT
        }

        # self.observation_horizons_slices = {
        #     0: slice(0, observation_horizon), # Biomass
        #     3: slice(
        #         observation_horizon,
        #         int(sampling_times_per_hour * observation_horizon) + observation_horizon,
        #     ), # DOT
        # }

        # --- Observation Space ---
        self.observation_upper_bound = np.concatenate([
            [self.final_time],
            20. * np.ones(observation_horizon),  # Biomass,
            101. * np.ones(int(sampling_times_per_hour * observation_horizon)), # DOT,
        ])
        self.observation_space = spaces.Box(
            low=0., high=1., shape=self.observation_upper_bound.shape, dtype=np.float64
        )

        # --- Feed and Control Profiles ---
        self.pulse_times = np.arange(time_batch + 5/60, self.final_time, 10/60)
        self.sample_offset = sample_offset

        self.control_inputs = {0: method_kiwiGym.ControlInputs(
            experiment_index=0, num_experiments=self.num_experiments,
            feed_concentration=200, induction_time=10., product_switch=0
        )}
        feed_vals = (32.406) * mu_ref * np.exp(mu_ref * (self.pulse_times - self.pulse_times[0]))
        feed_vals = np.round(feed_vals * 2) / 2
        feed_vals[feed_vals < 5] = 5
        feed_vals[3] = 0.
        self.feed_profiles = {0: {
            'time_pulse': self.pulse_times.tolist(),
            'Feed_pulse': feed_vals.tolist(),
            'time_sample': np.arange(self.final_time) + self.sample_offset,
            'time_sensor': np.linspace(0.04, self.final_time, sampling_times_per_hour * round(self.final_time)),
        }}
        self.feed_profiles_history = deepcopy(self.feed_profiles)
        self.initial_feed_profiles = deepcopy(self.feed_profiles)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # --- Randomize ODE parameters ---
        if self.random_ode_param_variance > 0.0:
            scale = self.random_ode_param_variance
            factors = self.np_random.gamma(1. / scale, scale, len(self.model_parameters))
            self.model_parameters = self.base_model_parameters * factors

        # --- Randomize initial states ---
        if self.random_initial_state_variance > 0.0:
            scale = self.random_initial_state_variance
            perturbation = self.np_random.gamma(1. / scale, scale, 2)
            randomized_initial_states = self.initial_states.copy()
            randomized_initial_states[:2] *= perturbation
        else:
            randomized_initial_states = self.initial_states.copy()

        # --- Reset simulation state ---
        self.current_time = 0
        self.time_interval = np.array([0, self.time_step])
        self.feed_profiles_history = deepcopy(self.initial_feed_profiles)
        self.terminated = False

        initial_state_template = {'t': 0, 'state': {}, 'sample': {}}
        initial_state_template['state'][0] = randomized_initial_states
        initial_state_template['sample'][0] = {j: [] for j in range(5)}
        self.state = initial_state_template
        self.initial_state_template = deepcopy(initial_state_template)

        # --- Integrate to first pulse to get initial observation ---
        obs_array = np.zeros(self.observation_space.shape, dtype=np.float64)
        pre_pulse_offset = 1
        obs_array[0] = int(self.pulse_times[0] - pre_pulse_offset)

        while self.current_time < round(self.pulse_times[0] - pre_pulse_offset):
            raw_obs, _, _ = self._perform_action([10]) # Use a neutral initial action
            obs_array[
                slice(1, 1 + sum(self.observation_lengths.values()))
            ] = raw_obs
        self.obs = obs_array

        return self.obs / self.observation_upper_bound, {}

    def step(self, action):
        # --- Apply action and step simulation ---
        applied_action_value = self.action_values[action]
        raw_obs, reward, terminated = self._perform_action(applied_action_value)

        # --- Update observation array ---
        obs_array = self.obs
        obs_array[0] = self.current_time

        obs_array[slice(1, 1 + sum(self.observation_lengths.values()))] = raw_obs
        self.obs = obs_array

        if self.render_mode == 'human' and terminated:
            self.render()

        normalized_obs = obs_array / self.observation_upper_bound
        return normalized_obs, reward, terminated, False, {}

    def _perform_action(self, action_step):
        # --- Apply feed changes ---
        feed_profiles_to_apply = deepcopy(self.feed_profiles_history)
        action_delay = 1

        t_pulse = np.array(feed_profiles_to_apply[0]['time_pulse'])
        feed_ref = np.array(feed_profiles_to_apply[0]['Feed_pulse'])
        feed_change = np.zeros_like(feed_ref)

        mask = (t_pulse <= (self.time_interval[1] + action_delay)) & (t_pulse >= (self.time_interval[0] + action_delay))
        feed_change[mask] = action_step

        feed_corrected = feed_ref + feed_change
        feed_corrected[(t_pulse >= t_pulse[0]) & (feed_corrected < 5)] = 5
        feed_profiles_to_apply[0]['Feed_pulse'] = feed_corrected.tolist()
        self.feed_profiles_history = feed_profiles_to_apply

        # --- Run simulation step ---
        self.state = method_kiwiGym.simulate_parallel(
            self.time_interval, self.state, self.control_inputs,
            self.model_parameters, self.feed_profiles_history
        )
        self.current_time = self.time_interval[1]

        # --- Construct observations ---
        measurements = []
        for channel, is_sensor in [(0, False), (3, True)]: # Biomass, DOT
            values = np.array(self.state['sample'][0][channel])
            vals_in_window = np.zeros(self.observation_lengths[channel])
            vals_in_window[-len(values):] = values[-len(vals_in_window):]

            measurements.append(vals_in_window)
        observation = np.concatenate(measurements)

        # --- Update time and check for termination ---
        self.time_interval += self.time_step
        reward = 0
        if self.current_time >= self.final_time:
            self.terminated = True
            reward = self._calculate_final_reward()
        else:
            self.terminated = False

        return observation, reward, self.terminated

    def _calculate_final_reward(self):
        n_sample = len(self.feed_profiles[0]['time_sample'])
        n_sensor = len(self.feed_profiles[0]['time_sensor'])
        sd_meas = np.array(([.2] * n_sample + [.2] * n_sample + [.5] * n_sample + [5] * n_sensor + [20] * n_sample))
        C2 = np.diag(sd_meas ** 2)

        state, div_min = method_kiwiGym.calculate_DIV(
            np.array([0, self.final_time]), self.initial_state_template, self.control_inputs,
            self.model_parameters, self.feed_profiles_history, C2
        )

        dot_min = min(state['sample'][0][3])
        div_constrain = ((20 - dot_min) * .05 + 1) ** 2 if dot_min < 20 else 1.
        div_calculated = div_min / div_constrain
        return (div_calculated - 7.065) / 7.065

    def render(self):
        print(f"Time: {self.current_time}, Done: {self.terminated}")
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

        ax1.plot(self.state['sample'][0][0], 'o-', label=f'Biomass Exp 0')
        ax2.plot(self.state['sample'][0][3], '.-', label=f'DOT Exp 0')
        ax1.set_ylabel("Biomass")
        ax1.legend()
        ax2.set_ylabel("DOT (%)")
        ax2.legend()
        ax2.set_xlabel("Time")
        plt.show()
