import numpy as np
from Case1 import method_kiwiGym
from Case1.method_kiwiGym import ControlInputs


def test_optimizer_reference_run():
    """
    Tests the optimizer_reference function to ensure it runs and returns a valid result.
    """
    # 1. Set up experiment parameters
    number_of_experiments = 1
    feed_start_time = 1.
    final_time = 2  # Reduced for faster testing
    time_span = np.array([0, 2])
    feed_pulse_times = np.arange(feed_start_time, final_time, 10 / 60)

    sample_schedule_offset = 0.99
    initial_growth_rate_guess = 0.14

    feed_profile = (32.406) * initial_growth_rate_guess * np.exp(initial_growth_rate_guess * (feed_pulse_times - feed_pulse_times[0]))
    feed_profile = np.round(feed_profile * 2) / 2
    feed_profile[feed_profile < 5] = 5

    experiment_data = {
        0: {
            'time_pulse': feed_pulse_times.tolist(),
            'Feed_pulse': feed_profile.tolist(),
            'time_sample': np.arange(0, final_time, 1) + sample_schedule_offset,
            'time_sensor': np.linspace(0.04, final_time, 25 * round(final_time)),
        }
    }

    initial_conditions = {
        't': time_span[0],
        'state': {0: [0.18, 4, 0, 100, 0, .0]},
        'sample': {0: {}}
    }

    control_inputs = {
        0:
            ControlInputs(
                experiment_index=0,
                num_experiments=number_of_experiments,
                feed_concentration=200,
                induction_time=1000.,
                product_switch=0,
            )
    }

    model_parameters = np.array(
        [1.2578, 0.43041, 0.6439, 7.0767, 0.4063, 0.1143 * 4, 0.1848 * 4, .4242, 1.586 * .7, 1.5874 * .7,
         0.3322 * .75, 0.0371, 0.0818, 9000, .1, 5] + [850] * number_of_experiments + [90] * number_of_experiments
    )

    number_of_samples = len(experiment_data[0]['time_sample'])
    number_of_sensor_points = len(experiment_data[0]['time_sensor'])
    measurement_standard_deviations = np.array(([.2] * number_of_samples + [.2] * number_of_samples + [.5] * number_of_samples + [5] * number_of_sensor_points + [50] * number_of_samples) * 1)
    measurement_covariance_matrix = np.diag(measurement_standard_deviations ** 2)

    lower_bounds = [0.075] * number_of_experiments
    upper_bounds = [0.15] * number_of_experiments
    # Reduced optimization time for quick test execution
    optimization_options = [1, 1]

    # 2. Call the function to be tested
    optimization_result = method_kiwiGym.optimizer_reference(
        lower_bounds,
        upper_bounds,
        optimization_options,
        np.array([0, final_time]),
        initial_conditions,
        control_inputs,
        model_parameters,
        experiment_data,
        measurement_covariance_matrix,
        max_optimizer_iterations=10,
    )

    # 3. Assert the results
    assert optimization_result is not None, "Optimization result should not be None."
    assert optimization_result.success, "Optimization should be successful."
    assert len(optimization_result.x) == number_of_experiments, "The length of the optimal vector should match n_exp."
    assert optimization_result.x[0] >= lower_bounds[0], "Optimal value should be within the lower bound."
    assert optimization_result.x[0] <= upper_bounds[0], "Optimal value should be within the upper bound."
