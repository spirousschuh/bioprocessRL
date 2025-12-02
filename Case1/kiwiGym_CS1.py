# %% Import
import numpy as np
from copy import deepcopy

from Case1 import method_kiwiGym

import matplotlib.pyplot as plt


num_experiments = 1
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
    ] + (
    [850] * num_experiments   # thetas[16...]: kla - kLa value (per experiment)
    + [90] * num_experiments  # thetas[...]: k_sensor - Sensor constant (per experiment)
)


DEFAULT_INITIAL_STATES = [0.18, 4, 0, 100, 0, .01]  # Biomass, Substrate, By-product, DOT, Product, Volume

# %%


class kiwiGym:
    def __init__(
            self,
            render_mode=None,
            current_time=0,
            num_experiments=1,
            final_time=14,
            time_step=1,
            sample_offsets=None,
            time_batch=5,
            mu_reference=None,
            initial_model_parameters=None,
            sampling_times_per_hour=25,
            minimal_observation=True,
            observation_horizon=1.,
    ):

        # Use safe defaults for list-like arguments
        if sample_offsets is None:
            sample_offsets = [0.99] * max(1, num_experiments)
        if mu_reference is None:
            mu_reference = [0.145] * max(1, num_experiments)

        # Define model parameters (bioprocess / kinetics / sensor params)
        # model_parameters is a 1D array that holds kinetic and sensor parameters
        self.model_parameters = np.array(initial_model_parameters) if initial_model_parameters is not None else np.array(DEFAULT_ODE_PARAMETERS)
        # Experiment / simulation configuration
        self.num_experiments = num_experiments
        self.final_time = final_time
        self.current_time = current_time
        self.time_step = time_step
        # time interval for the next integration step
        self.time_interval = np.array([current_time, current_time + self.time_step])
        # times at which feed pulses may occur
        self.pulse_times = np.arange(time_batch + 5/60, final_time, 10/60)
        # relative sample time offsets for each experiment
        self.sample_offsets = sample_offsets
        self.mu_reference = np.array(mu_reference)
        self.minimal_observation = minimal_observation
        self.observation_horizon = observation_horizon

        # --- Per-experiment data structures ---
        # initial_state_template: holds the template for states and samples
        initial_state_template = {'state': {}, 'sample': {}}  # States and samples template
        control_inputs = {}  # Fixed control inputs per experiment
        feed_profiles = {}  # Feed profile (time/values) per experiment

        # Initialize per-experiment templates and profiles
        for exp_idx in range(self.num_experiments):
            initial_state_template['t'] = self.time_interval[0]
            initial_state_template['state'][exp_idx] = [0.18, 4, 0, 100, 0, .01]
            initial_state_template['sample'][exp_idx] = {0: [], 1: [], 2: [], 3: [], 4: []}

            # Control inputs object (keeps metadata for each experiment)
            control_inputs[exp_idx] = method_kiwiGym.ControlInputs(
                experiment_index=exp_idx,
                num_experiments=self.num_experiments,
                feed_concentration=200,
                induction_time=10.,
                product_switch=0,
            )

            # Build a feed profile (pulse values) using mu_reference and exponential growth
            feed_profile_values = (32.406) * self.mu_reference[exp_idx] * np.exp(
                self.mu_reference[exp_idx] * (self.pulse_times - self.pulse_times[0])
            )
            # Round to nearest 0.5 uL and enforce a minimum
            feed_profile_values = np.round(feed_profile_values * 2) / 2
            feed_profile_values[feed_profile_values < 5] = 5

            feed_profile_values[3] = 0.
            feed_profiles[exp_idx] = {
                'time_pulse': self.pulse_times.tolist(),
                'Feed_pulse': feed_profile_values.tolist(),
                'time_sample': np.arange(self.final_time) + self.sample_offsets[exp_idx],
                'time_sensor': np.linspace(0.04, self.final_time, sampling_times_per_hour * round(self.final_time)),
            }
            

        # Save deep copies so we can reset state later
        self.initial_state_template = deepcopy(initial_state_template)
        self.state = deepcopy(initial_state_template)
        self.control_inputs = control_inputs
        self.feed_profiles = feed_profiles
        # historic profile is the active feed profile that gets modified by actions
        self.feed_profiles_history = deepcopy(self.feed_profiles)

        # Environment variables
        self.terminated = False
        # observation is a flattened vector of recent measurements; initialize to zeros
        # Keep the original shape logic (uses indexing from ControlInputs object)
        self.observation = np.zeros([self.control_inputs[0][0] * (4 + 25 * 0 + 1)])
        return
# %%    
    def reset(self, seed=None, model_parameters=[], initial_states=None):
        """Reset the environment to initial conditions. Optionally override model parameters."""
        # Optionally update model parameters
        if len(model_parameters) > 0:
            self.model_parameters = model_parameters

        if initial_states is None:
            initial_states = DEFAULT_INITIAL_STATES

        # Reset time
        self.current_time = 0
        self.time_interval = np.array([self.current_time, self.current_time + self.time_step])

        # Reset states and historic feed profiles
        initial_state_template = {'state': {}, 'sample': {}}
        for exp_idx in range(self.num_experiments):
            initial_state_template['t'] = self.time_interval[0]
            initial_state_template['state'][exp_idx] = initial_states
            initial_state_template['sample'][exp_idx] = {0: [], 1: [], 2: [], 3: [], 4: []}
        self.state = deepcopy(initial_state_template)
        self.feed_profiles_history = deepcopy(self.feed_profiles)
        # Reinitialize the flattened observation vector
        self.observation = np.zeros([self.control_inputs[0].num_experiments * (4 + 25 * 0 + 1)])
        self.terminated = False
        return
# %%    
    def render(self):
        """Simple plotting helper: show DOT (index 3) and Biomass (index 0) for each experiment."""
        for idx in range(self.control_inputs[0].num_experiments):
            plt.plot(self.state['sample'][idx][3], '.')
        plt.show()
        for idx in range(self.control_inputs[0].num_experiments):
            plt.plot(self.state['sample'][idx][0], 'o')
        plt.show()
        print('time: ', self.current_time, ' done: ', self.terminated, 'reward: ', getattr(self, 'reward', None))

# %%    
    def perform_action(self, action_step=[], feed_std=0.0, feed_to_zero_probability=0.0):
        """Apply an action (feed changes) for the current time interval and advance simulation.

        action_step: if left as default (empty list), the reference profile is used; otherwise
        action_step is applied to pulses that fall into the current time interval.
        """

        # If action_step is the default list object, use the reference (original) feed_profiles
        if action_step is list:
            feed_profiles_to_apply = deepcopy(self.feed_profiles)
        else:
            # Otherwise start from historic (possibly already modified) feed profiles
            feed_profiles_to_apply = deepcopy(self.feed_profiles_history)

            action_values = action_step
            action_delay = 0.5  # time offset used when mapping pulses to actions
            for exp_idx in range(self.control_inputs[0].num_experiments):
                t_pulse = np.array(feed_profiles_to_apply[exp_idx]['time_pulse'])
                feed_ref = np.array(feed_profiles_to_apply[exp_idx]['Feed_pulse'])

                feed_change = np.zeros(feed_ref.shape)
                # Apply action to pulses that fall into the current integration window
                feed_change[(t_pulse <= (self.time_interval[1] + action_delay)) & (t_pulse >= (self.time_interval[0] + action_delay))] = action_values

                feed_corrected = feed_ref + feed_change
                # Enforce minimum feed of 5 uL after the first pulse time
                feed_corrected[(t_pulse >= t_pulse[0]) & (feed_corrected < 5)] = 5

                feed_profiles_to_apply[exp_idx]['Feed_pulse'] = (feed_corrected).tolist()

        # Save the applied profile as historic
        self.feed_profiles_history = deepcopy(feed_profiles_to_apply)

        # perturb feed profiles with Gaussian noise if specified
        if feed_std > 0.0:
            for exp_idx in range(self.control_inputs[0].num_experiments):
                feed_array = np.array(self.feed_profiles_history[exp_idx]['Feed_pulse'])
                noise_factor = np.random.gamma(1. / feed_std, feed_std, size=feed_array.shape)
                feed_array_noisy = feed_array * noise_factor
                # Enforce minimum feed of 5 uL after the first pulse time
                t_pulse = np.array(self.feed_profiles_history[exp_idx]['time_pulse'])
                feed_array_noisy[(t_pulse >= t_pulse[0]) & (feed_array_noisy < 5)] = 5
                self.feed_profiles_history[exp_idx]['Feed_pulse'] = feed_array_noisy.tolist()

        if feed_to_zero_probability > 0.0:
            for exp_idx in range(self.control_inputs[0].num_experiments):
                feed_array = np.array(self.feed_profiles_history[exp_idx]['Feed_pulse'])
                random_values = np.random.rand(feed_array.shape[0])
                feed_array_zeroed = np.where(random_values < feed_to_zero_probability, 0.0, feed_array)
                self.feed_profiles_history[exp_idx]['Feed_pulse'] = feed_array_zeroed.tolist()

        # Run the simulation step (parallel for all experiments)
        self.state = method_kiwiGym.simulate_parallel(
            self.time_interval,
            self.state,
            self.control_inputs,
            self.model_parameters,
            self.feed_profiles_history,
        )
        # Advance current time to the end of the interval
        self.current_time = self.time_interval[1]

        ################ Construct observation vector from new state samples
        if len(self.observation) == 0:
            state_obs = np.zeros([self.control_inputs[0][0] * (4 + 1)])
        else:
            state_obs = np.array(self.observation)

        state_obs = state_obs[:, None]
        stacked_measurements = []
        for exp_idx in range(self.control_inputs[0].num_experiments):
            # We only include channels 0 (Biomass) and 3 (DOT) in the observation
            if self.minimal_observation:
                for measurement_channel in [0, 3]:
                    if measurement_channel == 0:
                        time_points = np.array(self.feed_profiles_history[exp_idx]['time_sample'])
                    else:
                        time_points = np.array(self.feed_profiles_history[exp_idx]['time_sensor'])



                    values = np.array(self.state['sample'][exp_idx][measurement_channel])
                    # print(values)
                    # restrict to values within the current interval
                    time_points_in_window = time_points[time_points <= self.time_interval[1]]
                    values_in_window = values[
                        (time_points_in_window > self.time_interval[0])
                        & (time_points_in_window <= self.time_interval[1])
                    ]

                    # For DOT we store a single aggregated value (minimum over the interval)
                    if measurement_channel == 3:
                        values_in_window = np.array([np.min(values_in_window)])

                    values_column = values_in_window[:, None]

                    if len(stacked_measurements) == 0:
                        stacked_measurements = values_column
                    else:
                        stacked_measurements = np.vstack((stacked_measurements, values_column))

            else:
                # gather measurements from the last observation_horizon
                # time_points = np.array(self.feed_profiles_history[exp_idx]['time_sample'])
                # biomass_ = np.array(self.state['sample'][exp_idx][0])  # Biomass
                # biomass_measurements = biomass_[
                #     np.logical_and(
                #         time_points > (self.current_time - self.observation_horizon),
                #         time_points <= self.current_time,
                #     )
                # ]
                # time_points = np.array(self.feed_profiles_history[exp_idx]['time_sensor'])
                # dot_ = np.array(self.state['sample'][exp_idx][3])  # DOT
                # dot_measurements = dot_[
                #     np.logical_and(
                #         time_points > (self.current_time - self.observation_horizon),
                #         time_points <= self.current_time,
                #     )
                # ]
                # stacked_measurements = np.concatenate([biomass_measurements, dot_measurements])
                raise NotImplementedError("Full observation mode not implemented yet.")

        state_obs = stacked_measurements
        self.observation = state_obs.flatten()
        ################
        # Update time interval, compute termination and reward if finished
        self.time_interval = np.array([self.current_time, self.current_time + self.time_step])

        if self.current_time >= self.final_time:
            self.terminated = True

            n_sample = len(self.feed_profiles[0]['time_sample'])
            n_sensor = len(self.feed_profiles[0]['time_sensor'])
            sd_meas = np.array(([.2] * n_sample + [.2] * n_sample + [.5] * n_sample + [5] * n_sensor + [20] * n_sample) * 1)
            C2 = np.diag(sd_meas ** 2)

            # Compute DIV objective (biomass and constraints)
            state, DIV_min = method_kiwiGym.calculate_DIV(
                np.array([0, self.final_time]),
                self.initial_state_template,
                self.control_inputs,
                self.model_parameters,
                self.feed_profiles_history,
                C2,
            )

            dot_min = min(state['sample'][0][3])

            if dot_min < 20:
                DIV_constrain = ((20 - dot_min) * .05 + 1) ** 2
            else:
                DIV_constrain = 1

            DIV_constr = np.array(DIV_constrain)
            DIV_calculated = DIV_min / DIV_constr
            DIV_normalized = (DIV_calculated - 7.065) / 7.065
            self.reward = DIV_normalized

            print("reward: ", self.reward, "Biomass: ", DIV_min, "Constrain: ", DIV_constrain)

        else:
            self.terminated = False
            self.reward = 0
        return self.observation, self.reward, self.terminated
