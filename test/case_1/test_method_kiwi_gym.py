import pytest
import numpy as np
from method_kiwiGym import intM, function_simulation

@pytest.fixture
def default_initial_state():
    """Provides a default initial state vector for the simulation."""
    return np.array([0.18, 4, 0, 100, 0, 0.0])

@pytest.fixture
def default_control_inputs():
    """Provides a default set of control inputs."""
    # Corresponds to [feed_concentration, experiment_index, num_experiments, some_factor, product_switch]
    return [200, 0, 1, 10, 1]

@pytest.fixture
def default_model_parameters():
    """Provides a default set of model parameters."""
    base_params = [
        1.2578, 0.43041, 0.6439, 7.0767, 0.4063, 0.1143 * 4, 0.1848 * 4, 0.4242,
        1.586 * 0.7, 1.5874 * 0.7, 0.3322 * 0.75, 0.0371, 0.0818, 9000, 0.1, 5
    ]
    # Appending experiment-specific parameters (kla and k_sensor)
    experiment_params = [850, 90]
    return np.array(base_params + experiment_params)

@pytest.fixture
def default_dynamic_conditions():
    """Provides a default set of dynamic conditions for the simulation."""
    final_time = 1.0
    feed_pulse_times = np.arange(0, final_time, 10 / 60)
    # Use a simple constant feed profile for testing purposes
    feed_profile = np.full_like(feed_pulse_times, 10.0)
    return {
        'time_pulse': feed_pulse_times.tolist(),
        'Feed_pulse': feed_profile.tolist(),
    }

def test_intM_integration(default_initial_state, default_control_inputs, default_model_parameters):
    """
    Tests the intM ODE integration function to ensure it runs and returns results of the correct shape and type.
    """
    # 1. Define simulation time span
    time_span = np.linspace(0, 1, 25)

    # 2. Call the function to be tested
    time_result, state_result = intM(
        ts0=time_span,
        Xo0=default_initial_state,
        u0=default_control_inputs,
        TH0=default_model_parameters
    )

    # 3. Assert the results
    assert time_result is not None, "Time vector should not be None."
    assert state_result is not None, "State matrix should not be None."

    # Check dimensions
    assert time_result.shape == (len(time_span),), "Time vector should have the same length as the input time span."
    assert state_result.shape == (len(default_initial_state), len(time_span)), "State matrix shape should be (num_states, num_time_points)."

    # Check for non-negative values, as states like concentration cannot be negative
    assert np.all(state_result >= 0), "All state variable values should be non-negative."

    # Check if the initial state is correctly set
    assert np.allclose(state_result[:, 0], default_initial_state), "The first column of the state matrix should match the initial state."

def test_function_simulation(default_initial_state, default_control_inputs, default_model_parameters, default_dynamic_conditions):
    """
    Tests the function_simulation method to ensure it runs and returns results of the correct shape and type.
    """
    # 1. Define simulation time span
    time_span = np.array([0, 1])

    # 2. Call the function to be tested
    time_result, state_result = function_simulation(
        time_span=time_span,
        initial_state=default_initial_state,
        control_input=default_control_inputs,
        model_parameters=default_model_parameters,
        dynamic_conditions=default_dynamic_conditions
    )

    # 3. Assert the results
    # Check dimensions and types
    assert isinstance(time_result, np.ndarray), "Time result should be a numpy array."
    assert isinstance(state_result, np.ndarray), "State result should be a numpy array."
    assert state_result.shape[1] == len(default_initial_state), "State matrix should have columns equal to the number of states."

    # Check for non-negative values
    assert np.all(state_result >= 0), "All state variable values should be non-negative."

    # Check if the simulation starts with the correct initial state
    assert np.allclose(state_result[0, :], default_initial_state), "The first row of the state matrix should match the initial state."
