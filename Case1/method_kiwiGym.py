# %% Import
import numpy as np
# import json
import time
from scipy.integrate import solve_ivp
# from joblib import Parallel, delayed

from copy import deepcopy

from scipy.optimize import shgo,dual_annealing,minimize,differential_evolution

import matplotlib.pyplot as plt

# %%
def optimizer_reference(
    lower_bounds,
    upper_bounds,
    optimization_options,
    time_vector,
    initial_conditions,
    control_inputs,
    model_parameters,
    experiment_data,
    measurement_covariance,
    initial_parameter_covariance=[],
    max_optimizer_iterations=1000,
):
    """
    Optimizes a reference trajectory using dual annealing followed by a local search.

    Args:
        lower_bounds (list): Lower bounds for the optimization variables.
        upper_bounds (list): Upper bounds for the optimization variables.
        optimization_options (list): Optimization options, including max function evaluations.
        time_vector (np.array): Time vector for the simulation.
        initial_conditions (dict): Initial conditions for the state variables.
        control_inputs (dict): Control inputs.
        model_parameters (np.array): Initial guess for the model parameters.
        experiment_data (dict): Dictionary containing dynamic conditions like feed pulses.
        measurement_covariance (np.array): Covariance matrix of the measurement noise.
        initial_parameter_covariance (list, optional): Initial covariance matrix of the parameters. Defaults to [].
        max_optimizer_iterations (int, optional): Maximum iterations for the dual annealing optimizer. Defaults to 1000.

    Returns:
        scipy.optimize.OptimizeResult: The optimization result from the local search.
    """
    # Convert bounds to numpy arrays
    optimization_variable_lower_bounds = np.array(lower_bounds)
    optimization_variable_upper_bounds = np.array(upper_bounds)
    optimization_variable_mean = np.linspace(
        optimization_variable_lower_bounds[0],
        optimization_variable_upper_bounds[0],
        len(optimization_variable_lower_bounds)
    )

    # Create a list of tuples for the bounds, which is the format required by scipy.optimize
    optimization_bounds = [
        (optimization_variable_lower_bounds[i], optimization_variable_upper_bounds[i]) for i in range(len(optimization_variable_lower_bounds))
    ]

    # Estimate the average time for one objective function evaluation to calibrate the optimizer's runtime
    test_start_time = time.time()
    obj_fun(
        optimization_variable_lower_bounds,
        time_vector,
        initial_conditions,
        control_inputs,
        model_parameters,
        experiment_data,
        measurement_covariance,
        initial_parameter_covariance,
    )
    obj_fun(
        optimization_variable_upper_bounds,
        time_vector,
        initial_conditions,
        control_inputs,
        model_parameters,
        experiment_data,
        measurement_covariance,
        initial_parameter_covariance,
    )
    obj_fun(
        optimization_variable_mean,
        time_vector,
        initial_conditions,
        control_inputs,
        model_parameters,
        experiment_data,
        measurement_covariance,
        initial_parameter_covariance,
    )
    average_obj_fun_eval_time = (time.time() - test_start_time) / 3

    # Calculate the number of function evaluations for global and local optimization based on the allocated time
    num_global_evaluations = round(60 * optimization_options[0] / (average_obj_fun_eval_time * 1.2))
    num_local_evaluations = round(60 * optimization_options[1] / (average_obj_fun_eval_time * 1.2))

    print(f"Global evaluations: {num_global_evaluations}, Local evaluations: {num_local_evaluations}")

    # Perform global optimization using dual annealing to explore the search space broadly
    global_optimization_result = dual_annealing(
        lambda x: obj_fun(
            x,
            time_vector,
            initial_conditions,
            control_inputs,
            model_parameters,
            experiment_data,
            measurement_covariance,
            initial_parameter_covariance,
        ),
        bounds=optimization_bounds,
        maxfun=num_global_evaluations,
        no_local_search=True,
        maxiter=max_optimizer_iterations,
    )
    print(f"Global optimization result: {global_optimization_result.x}")

    # Perform local optimization using the result from the global optimization as a starting point to refine the solution
    local_optimization_result = minimize(
        lambda x: obj_fun(
            x,
            time_vector,
            initial_conditions,
            control_inputs,
            model_parameters,
            experiment_data,
            measurement_covariance,
            initial_parameter_covariance,
        ),
        global_optimization_result.x,
        bounds=optimization_bounds,
        method="Nelder-Mead",
        options={"maxfev": num_local_evaluations},
    )

    print(f"Local optimization result: {local_optimization_result.x}")
    return local_optimization_result

# %%

def obj_fun(
    optimization_variable,
    time_vector,
    initial_conditions,
    control_inputs,
    model_parameters,
    experiment_data,
    measurement_covariance,
    initial_parameter_covariance=[],
):
    """
    Objective function for the optimization. It calculates the determinant of the Fisher Information Matrix (FIM).

    Args:
        optimization_variable (np.array): The optimization variable (specific growth rate).
        time_vector (np.array): Time vector for the simulation.
        initial_conditions (dict): Initial conditions for the state variables.
        control_inputs (dict): Control inputs.
        model_parameters (np.array): Initial guess for the model parameters.
        experiment_data (dict): Dictionary containing dynamic conditions like feed pulses.
        measurement_covariance (np.array): Covariance matrix of the measurement noise.
        initial_parameter_covariance (list, optional): Initial covariance matrix of the parameters. Defaults to [].

    Returns:
        float: The value of the objective function (negative determinant of the FIM, with constraints).
    """
    # Create a deep copy of the experiment data to avoid modifying the original dictionary during optimization
    experiment_data_copy = deepcopy(experiment_data)

    # Update the feed pulse profile based on the current value of the optimization variable (growth rate)
    for i in range(control_inputs[0][0]):
        time_pulse = np.array(experiment_data_copy[i]["time_pulse"])
        # Exponential feed profile based on the desired growth rate
        feed_pulse = (32.406) * optimization_variable[i] * np.exp(optimization_variable[i] * (time_pulse - time_pulse[0]))
        feed_pulse = np.round(feed_pulse * 2) / 2  # Round to nearest 0.5
        feed_pulse[feed_pulse < 5] = 5  # Enforce a minimum feed rate
        experiment_data_copy[i]["Feed_pulse"] = feed_pulse.tolist()

    # Calculate the determinant of the Fisher Information Matrix (DIV) for the current parameters
    simulated_states, div_value = calculate_DIV(
        time_vector,
        initial_conditions,
        control_inputs,
        model_parameters,
        experiment_data_copy,
        measurement_covariance,
        initial_parameter_covariance,
    )

    # Apply constraints to the objective function based on the minimum dissolved oxygen tension (DOT)
    min_dot_values = []
    div_constraints = []
    for i in range(control_inputs[0][0]):
        min_dot = min(simulated_states["sample"][i][3])
        min_dot_values.append(min_dot)
        # Penalize the objective function if DOT falls below a critical value (e.g., 20)
        if min_dot < 20:
            div_constraints.append((1 + (20 - min_dot) * 10) * 1e0)
        else:
            div_constraints.append(1)

    # Normalize the DIV value by the sum of constraints
    div_constraints_array = np.array(div_constraints)
    print(
        f"Current optimization variable: {optimization_variable}, "
        f"Constrained DIV: {div_value / np.sum(div_constraints_array):.4e}, "
        f"Min DOT constraint: {min(min_dot_values):.2f}"
    )

    # Return the negative of the normalized DIV value because optimizers typically minimize
    return div_value * (-1) / np.sum(div_constraints_array)

# %%
def calculate_DIV(
    time_vector,
    initial_conditions,
    control_inputs,
    model_parameters,
    experiment_data,
    measurement_covariance=[],
    initial_parameter_covariance=[],
):
    """
    Calculates the determinant of the Fisher Information Matrix (DIV).

    Args:
        time_vector (np.array): Time vector for the simulation.
        initial_conditions (dict): Initial conditions for the state variables.
        control_inputs (dict): Control inputs.
        model_parameters (np.array): Initial guess for the model parameters.
        experiment_data (dict): Dictionary containing dynamic conditions like feed pulses.
        measurement_covariance (np.array, optional): Covariance matrix of the measurement noise. Defaults to [].
        initial_parameter_covariance (list, optional): Initial covariance matrix of the parameters. Defaults to [].

    Returns:
        tuple: A tuple containing the simulated states and the DIV value.
    """
    # Simulate the process in parallel for all experiments to get the necessary state trajectories
    simulated_states = simulate_parallel(
        time_vector, initial_conditions, control_inputs, model_parameters, experiment_data
    )

    # Extract the DIV value from the simulation results, which is stored in the first sample of the first experiment
    div_value = simulated_states["sample"][0][0][-1]

    return simulated_states, div_value
# %%

def simulate_parallel(time_span, initial_conditions, control_inputs, model_parameters, experiment_data):
    """
    Simulates the bioreactor experiments in parallel.

    Args:
        time_span (np.array): Time span for the simulation.
        initial_conditions (dict): Initial conditions for the state variables.
        control_inputs (dict): Control inputs.
        model_parameters (np.array): Model parameters.
        experiment_data (dict): Dictionary containing dynamic conditions.

    Returns:
        dict: A dictionary containing the simulated states and sampled values.
    """
    # Deepcopy the initial conditions to avoid modifying the original dictionary
    simulated_states = deepcopy(initial_conditions)
    # Get the list of bioreactors to simulate
    bioreactor_list = np.arange(control_inputs[0][0]).tolist()

    # Dictionary to store simulation results for each bioreactor
    simulation_results = {}

    # The code is commented out, but it suggests that parallel execution is possible using joblib
    # results = Parallel(n_jobs=-1)(
    #     delayed(simulate_interval)(i1, time_span, simulated_states, control_inputs, model_parameters, experiment_data)
    #     for i1 in bioreactor_list
    # )
    
    # for i1, result in zip(bioreactor_list, results):
    #     simulation_results[i1] = result

    # Sequentially simulate each bioreactor experiment
    for i in bioreactor_list:
        simulation_results[i] = simulate_interval(i, time_span, simulated_states, control_inputs, model_parameters, experiment_data)

    # Process the simulation results for each bioreactor
    for i in bioreactor_list:
        # Update the state of the bioreactor with the final state from the simulation interval
        simulated_states['state'][i] = simulation_results[i][-1, 1:]

        # Interpolate and store sample values for different states (Xv, S, A, P)
        for state_index in [0, 1, 2, 4]:
            all_sample_times = experiment_data[i]['time_sample']
            # Filter sample times to be within the current simulation interval
            sample_times_in_interval = all_sample_times[(all_sample_times > time_span[0]) & (all_sample_times <= time_span[1])]

            # Interpolate simulation results at the specified sample times
            interpolated_samples = np.interp(sample_times_in_interval, simulation_results[i][:, 0], simulation_results[i][:, state_index + 1])
            try:
                # Append interpolated samples to the existing list
                simulated_states['sample'][i][state_index] = simulated_states['sample'][i][state_index] + interpolated_samples.tolist()
            except:
                # If it's the first set of samples, create a new list
                simulated_states['sample'][i][state_index] = interpolated_samples.tolist()

        # Interpolate and store sensor values (DOT)
        all_sensor_times = experiment_data[i]['time_sensor']
        # Filter sensor times to be within the current simulation interval
        sensor_times_in_interval = all_sensor_times[(all_sensor_times > time_span[0]) & (all_sensor_times <= time_span[1])]
        # Interpolate DOT values at sensor measurement times
        interpolated_sensor_values = np.interp(sensor_times_in_interval, simulation_results[i][:, 0], simulation_results[i][:, 4])

        try:
            # Append interpolated sensor values to the existing list for DOT (index 3)
            simulated_states['sample'][i][3] = simulated_states['sample'][i][3] + interpolated_sensor_values.tolist()
        except:
            # If it's the first set of sensor values, create a new list
            simulated_states['sample'][i][3] = interpolated_sensor_values.tolist()

    return simulated_states

# %%
def simulate_interval(bioreactor_index, time_span, initial_conditions, control_inputs, model_parameters, experiment_data):
    """
    Simulates a single interval of the bioreactor.

    Args:
        bioreactor_index (int): Index of the bioreactor.
        time_span (np.array): Time span for the simulation.
        initial_conditions (dict): Initial conditions for the state variables.
        control_inputs (dict): Control inputs.
        model_parameters (np.array): Model parameters.
        experiment_data (dict): Dictionary containing dynamic conditions.

    Returns:
        np.array: An array containing the time and state variables over the interval.
    """
    # Assemble the control input vector for the simulation function
    u = [control_inputs[bioreactor_index][1]] + [bioreactor_index] + [control_inputs[bioreactor_index][0]] + [control_inputs[bioreactor_index][2]] + [1]
    # Get the initial state for the current bioreactor
    initial_state = np.array(initial_conditions['state'][bioreactor_index])
    # Get the dynamic conditions for the current bioreactor
    dynamic_conditions = experiment_data[bioreactor_index]
    # Run the simulation for the given interval
    time_points, state_trajectory = function_simulation(time_span, initial_state, u, model_parameters, dynamic_conditions)

    # Combine time and state trajectories into a single array
    return np.hstack((time_points[:, None], state_trajectory))

# %%
def function_simulation(time_span, initial_state, control_input, model_parameters, dynamic_conditions={}):
    """
    Performs the simulation of the fed-batch process over a given time interval.

    Args:
        time_span (np.array): Time span for the simulation.
        initial_state (np.array): Initial state vector.
        control_input (list): Control input parameters.
        model_parameters (np.array): Model parameters.
        dynamic_conditions (dict, optional): Dictionary containing dynamic conditions. Defaults to {}.

    Returns:
        tuple: A tuple containing the time vector and the simulated state variables.
    """
    # Extract the base model parameters
    simulation_model_parameters = model_parameters[0:16]

    # Append experiment-specific parameters (e.g., kla, k_sensor) to the parameter vector
    simulation_model_parameters = np.append(
        simulation_model_parameters,
        model_parameters[16 + int(control_input[1])],
    )
    simulation_model_parameters = np.append(
        simulation_model_parameters,
        model_parameters[16 + int(control_input[1]) + int(control_input[2])],
    )

    # Define the start and end times for the simulation interval
    start_time = time_span[0]
    end_time = time_span[-1]

    # Extract the feed pulse times and rates from the dynamic conditions
    all_time_pulses = np.array(dynamic_conditions['time_pulse'])
    all_feed_pulses = np.array(dynamic_conditions['Feed_pulse'])

    # Filter the feed pulses that occur within the current simulation interval
    time_feed_in_interval = all_time_pulses[(all_time_pulses >= start_time) & (all_time_pulses <= end_time)]
    feed_rate_in_interval = all_feed_pulses[(all_time_pulses >= start_time) & (all_time_pulses <= end_time)]

    # If there are no feed pulses in the interval, simulate with zero feed
    if len(time_feed_in_interval) == 0:
        time_feed_in_interval = np.array([start_time, end_time])
        feed_rate_in_interval = np.array([0, 0])
    else:
        # Ensure the simulation starts from the beginning of the interval, even if the first pulse is later
        if start_time < time_feed_in_interval[0]:
            time_feed_in_interval = np.append(start_time, time_feed_in_interval)
            feed_rate_in_interval = np.append(0, feed_rate_in_interval)
        # Ensure the simulation runs to the end of the interval, even if the last pulse is earlier
        if end_time > time_feed_in_interval[-1]:
            time_feed_in_interval = np.append(time_feed_in_interval, end_time)
            feed_rate_in_interval = np.append(feed_rate_in_interval, 0)

    # Initialize the state for the simulation loop
    current_state = initial_state.copy()

    # Initialize arrays to store the full time and state trajectories
    time_trajectory = np.array(start_time)
    state_trajectory = np.array([current_state]).transpose()

    # Loop through the feed pulse intervals and simulate each one
    for i in range(len(feed_rate_in_interval) - 1):
        # Define the time points for the current sub-interval
        sub_interval_time = np.linspace(time_feed_in_interval[i], time_feed_in_interval[i+1], 25 + 1)
        # Apply the feed pulse by increasing the substrate concentration
        current_state[1] = current_state[1] + feed_rate_in_interval[i] * 1e-6 * control_input[0] / 0.01
        # Integrate the ODEs over the sub-interval
        t, y = intM(sub_interval_time, current_state, control_input, simulation_model_parameters)
        # Update the state for the next sub-interval
        current_state = y[:, -1].copy()

        # Append the results of the sub-interval to the overall trajectory
        time_trajectory = np.append(time_trajectory, t[1:])
        state_trajectory = np.append(state_trajectory, y[:, 1:], axis=1)

    return time_trajectory, state_trajectory.transpose()
# %%
def odeFB(t,Xo,THo,u):
    """
    Defines the system of ordinary differential equations (ODEs) for the fed-batch bioreactor.

    Args:
        t (float): Current time.
        Xo (np.array): State vector [Xv, S, A, DOT, P, mu_m].
        THo (np.array): Model parameters.
        u (list): Control inputs.

    Returns:
        np.array: The derivatives of the state variables.
    """

    X=Xo.copy()
    thetas = THo.copy()
    # Ensure state variables are non-negative
    X = np.maximum(X, 1e-9)
    
    # Unpack state variables for clarity
    Xv=X[0]    # Viable cell concentration
    S=X[1]     # Substrate concentration
    A=X[2]     # By-product concentration
    DOT=X[3]   # Dissolved Oxygen Tension
    P=X[4]     # Product concentration
    mu_m=X[5]  # Specific growth rate

    # Clamp DOT to a maximum of 100
    DOT = np.minimum(DOT, 100)
            
    # Unpack model parameters
    qs_max=thetas[0]
    fracc_q_ox_max=thetas[1]
    qa_max=thetas[2]

    
    Ys_ox=thetas[4]
    Ya_p=thetas[5]
    Ya_c=thetas[6]
    Yo_ox=thetas[8]
    Yo_a=thetas[9]
    Yxs_of=thetas[10]
    
    Ks=thetas[11]
    n_ox=4
    
    Ka=thetas[12]
    Ksi=thetas[3]
    Kai=thetas[7]
    Ko=0.1057
    
    kla=thetas[16]
    k_sensor=thetas[17]
    
    ky_1=thetas[13]
    ky_2=thetas[14]
    ky_3=thetas[15]
    
    DO_star=100
    H=13000#
    
    # --- Kinetic model equations ---
    # Substrate uptake rate
    qs=qs_max*S/(S+Ks)*Ksi/(Ksi+A)
    q_ox_max=fracc_q_ox_max*qs_max
    
    # Steady-state calculations for oxygen transfer
    q_ox_ss=qs*(1/((qs/q_ox_max)**n_ox+1))**(1/n_ox)
    qac_ss=qa_max*A/(A+Ka)*Kai/(Kai+S)
    b_ss=Ko+(q_ox_ss*Yo_ox+qac_ss*Yo_a)*Xv*H/kla-DO_star
    c_ss=-DO_star*Ko
    DOT_ss=(-b_ss+(b_ss*b_ss-4*c_ss)**.5)/2

    # Oxygen-dependent rates
    q_ox=qs*(1/((qs/q_ox_max)**n_ox+1))**(1/n_ox)*DOT_ss/(DOT_ss+Ko)
    q_of=qs-q_ox
    
    qac=qa_max*A/(A+Ka)*Kai/(Kai+S)*DOT_ss/(DOT_ss+Ko)
    
    qap=q_of*Ya_p
    
    # Specific growth rate
    mu=q_ox*Ys_ox+qac*Ya_c+Yxs_of*q_of

    # Product formation switch
    if t>=u[3]:
        s_prod=u[4]
    else:
        s_prod=0
        
    # Product formation rate
    f_qp=ky_1*mu_m/(mu_m+ky_2+(ky_3*mu_m)**2)
    q_prod=s_prod*f_qp

    # --- Differential equations for state variables ---
    dXv=(mu)*Xv
    dS=-(qs)*Xv
    dA=qap*Xv-qac*Xv
    dDOT=k_sensor*(DOT_ss-DOT)
    dP=q_prod*Xv
    dmu_m=(mu-mu_m)/(.167)
    
    dX=np.array([dXv,dS,dA,dDOT,dP,dmu_m])
    return dX
# %%     
def intM(ts0,Xo0,u0,TH0):    
    """
    Integrates the ODEs over a given time span.

    Args:
        ts0 (np.array): Time span for the integration.
        Xo0 (np.array): Initial state vector.
        u0 (list): Control inputs.
        TH0 (np.array): Model parameters.

    Returns:
        tuple: A tuple containing the time vector and the integrated state variables.
    """

    tspan=np.array([ts0[0],ts0[-1]])
    Xo1=Xo0.tolist().copy()


    # Use solve_ivp with the BDF method, which is suitable for stiff ODEs
    sol=solve_ivp(lambda t,y: odeFB(t,y,TH0,u0) ,tspan,Xo1,method="BDF", rtol=1e-5, atol=1e-5,t_eval=ts0)
    y_interm=sol.y
    # Ensure that state variables do not become negative
    y_interm[y_interm<0]=0
    y_return=y_interm.copy()

    return sol.t,y_return