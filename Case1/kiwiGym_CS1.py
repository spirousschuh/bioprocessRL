# %% Import
import numpy as np
from copy import deepcopy

from Case1 import method_kiwiGym

import matplotlib.pyplot as plt
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
            model_parameters0=None,
    ):

        # Use safe defaults for list-like arguments
        if sample_offsets is None:
            sample_offsets = [0.99] * max(1, num_experiments)
        if mu_reference is None:
            mu_reference = [0.145] * max(1, num_experiments)

        # Define model parameters (bioprocess / kinetics / sensor params)
        # model_parameters is a 1D array that holds kinetic and sensor parameters
        self.model_parameters = np.array(model_parameters0) or np.array([
            1.2578, 0.43041, 0.6439,  7.0767,  0.4063,  0.1143*4,  0.1848*4,
            .4242,    1.586*.7, 1.5874*.7,  0.3322*.75,  0.0371,  0.0818,    9000,
            .1, 5
        ] + [850] * num_experiments + [90] * num_experiments)

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

            feed_profiles[exp_idx] = {
                'time_pulse': self.pulse_times.tolist(),
                'Feed_pulse': feed_profile_values.tolist(),
                'time_sample': np.arange(self.final_time) + self.sample_offsets[exp_idx],
                'time_sensor': np.linspace(0.04, self.final_time, 25 * round(self.final_time)),
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
    def reset(self, seed=None, model_parameters=[]):
        """Reset the environment to initial conditions. Optionally override model parameters."""
        # Optionally update model parameters
        if len(model_parameters) > 0:
            self.model_parameters = model_parameters

        # Reset time
        self.current_time = 0
        self.time_interval = np.array([self.current_time, self.current_time + self.time_step])

        # Reset states and historic feed profiles
        initial_state_template = {'state': {}, 'sample': {}}
        for exp_idx in range(self.num_experiments):
            initial_state_template['t'] = self.time_interval[0]
            initial_state_template['state'][exp_idx] = [0.18, 4, 0, 100, 0, .01]
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
    def perform_action(self, action_step=[]):
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
            action_delay = 1  # time offset used when mapping pulses to actions
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
            for measurement_channel in [0, 3]:
                if measurement_channel == 0:
                    time_points = np.array(self.feed_profiles_history[exp_idx]['time_sample'])
                else:
                    time_points = np.array(self.feed_profiles_history[exp_idx]['time_sensor'])

                values = np.array(self.state['sample'][exp_idx][measurement_channel])
                # restrict to values within the current interval
                time_points_in_window = time_points[time_points <= self.time_interval[1]]
                values_in_window = values[(time_points_in_window > self.time_interval[0]) & (time_points_in_window <= self.time_interval[1])]

                # For DOT we store a single aggregated value (minimum over the interval)
                if measurement_channel == 3:
                    values_in_window = np.array([np.min(values_in_window)])

                values_column = values_in_window[:, None]

                if len(stacked_measurements) == 0:
                    stacked_measurements = values_column
                else:
                    stacked_measurements = np.vstack((stacked_measurements, values_column))

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
            state, DIV_min = method_kiwiGym.calculate_DIV(np.array([0, self.final_time]), self.initial_state_template, self.control_inputs, self.model_parameters, self.feed_profiles_history, C2)

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
        return self.observation, self.reward, self.terminated#,
