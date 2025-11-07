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

    # Create a list of tuples for the bounds
    optimization_bounds = [
        (optimization_variable_lower_bounds[i], optimization_variable_upper_bounds[i]) for i in range(len(optimization_variable_lower_bounds))
    ]

    # Estimate the average time for one objective function evaluation
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

    # Calculate the number of function evaluations for global and local optimization
    num_global_evaluations = round(60 * optimization_options[0] / (average_obj_fun_eval_time * 1.2))
    num_local_evaluations = round(60 * optimization_options[1] / (average_obj_fun_eval_time * 1.2))

    print(f"Global evaluations: {num_global_evaluations}, Local evaluations: {num_local_evaluations}")

    # Perform global optimization using dual annealing
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

    # Perform local optimization using the result from the global optimization
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
    # Create a deep copy of the experiment data to avoid modifying the original dictionary
    experiment_data_copy = deepcopy(experiment_data)

    # Update the feed pulse based on the optimization variable
    for i in range(control_inputs[0][0]):
        time_pulse = np.array(experiment_data_copy[i]["time_pulse"])
        feed_pulse = (32.406) * optimization_variable[i] * np.exp(optimization_variable[i] * (time_pulse - time_pulse[0]))
        feed_pulse = np.round(feed_pulse * 2) / 2
        feed_pulse[feed_pulse < 5] = 5
        experiment_data_copy[i]["Feed_pulse"] = feed_pulse.tolist()

    # Calculate the determinant of the Fisher Information Matrix (DIV)
    simulated_states, div_value = calculate_DIV(
        time_vector,
        initial_conditions,
        control_inputs,
        model_parameters,
        experiment_data_copy,
        measurement_covariance,
        initial_parameter_covariance,
    )

    # Apply constraints based on the minimum dissolved oxygen tension (DOT)
    min_dot_values = []
    div_constraints = []
    for i in range(control_inputs[0][0]):
        min_dot = min(simulated_states["sample"][i][3])
        min_dot_values.append(min_dot)
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

    # Return the negative of the normalized DIV value
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
    # Simulate the process in parallel for all experiments
    simulated_states = simulate_parallel(
        time_vector, initial_conditions, control_inputs, model_parameters, experiment_data
    )

    # Extract the DIV value from the simulation results
    div_value = simulated_states["sample"][0][0][-1]

    return simulated_states, div_value
# %%

def simulate_parallel(ts,XX0,uu,TH_param,DD):  
    """
    Simulates the bioreactor experiments in parallel.

    Args:
        ts (np.array): Time span for the simulation.
        XX0 (dict): Initial conditions for the state variables.
        uu (dict): Control inputs.
        TH_param (np.array): Model parameters.
        DD (dict): Dictionary containing dynamic conditions.

    Returns:
        dict: A dictionary containing the simulated states and sampled values.
    """
    XX=deepcopy(XX0)
    brxtor_list=np.arange(uu[0][0]).tolist()
    
    ty={}
    
    # results = Parallel(n_jobs=-1)(
    #     delayed(simulate_interval)(i1, ts,XX,uu,TH_param,DD)
    #     for i1 in brxtor_list
    # )
    

    # for i1, result in zip(brxtor_list, results):
    #     ty[i1] = result
        
    for i1 in brxtor_list: 
        ty[i1]=simulate_interval(i1, ts,XX,uu,TH_param,DD)

        
    for i1 in brxtor_list:
        XX['state'][i1]=ty[i1][-1,1:]
        
        
        for i2 in [0,1,2,4]:#range(4):
            ts_sample_all=DD[i1]['time_sample'] 
            ts_sample=ts_sample_all[(ts_sample_all>ts[0]) & (ts_sample_all<=ts[1])]
            
            sample_interp=np.interp(ts_sample,ty[i1][:,0],ty[i1][:,i2+1])
            try:
                XX['sample'][i1][i2]=XX['sample'][i1][i2]+sample_interp.tolist() 
                
            except:
                XX['sample'][i1][i2]=sample_interp.tolist()
                
                
        ts_sensor_all=DD[i1]['time_sensor'] 
        ts_sensor=ts_sensor_all[(ts_sensor_all>ts[0]) & (ts_sensor_all<=ts[1])]
        sensor_interp=np.interp(ts_sensor,ty[i1][:,0],ty[i1][:,4])

        try:
            XX['sample'][i1][3]=XX['sample'][i1][3]+sensor_interp.tolist() 
        except:
            XX['sample'][i1][3]=sensor_interp.tolist()

    return XX

# %%
def simulate_interval(index_mbr,ts,XX,uu,TH_param,DD):
    """
    Simulates a single interval of the bioreactor.

    Args:
        index_mbr (int): Index of the bioreactor.
        ts (np.array): Time span for the simulation.
        XX (dict): Initial conditions for the state variables.
        uu (dict): Control inputs.
        TH_param (np.array): Model parameters.
        DD (dict): Dictionary containing dynamic conditions.

    Returns:
        np.array: An array containing the time and state variables over the interval.
    """
    u=[uu[index_mbr][1]]+[index_mbr]+[uu[index_mbr][0]]+[uu[index_mbr][2]]+[1]
    X = np.array(XX['state'][index_mbr])
    D = DD[index_mbr]
    t, y = function_simulation(ts, X, u, TH_param, D)

    return np.hstack((t[:,None],y))

# %%
def function_simulation(ts0,Xo0,u0,THs,D0={}):
    """
    Performs the simulation of the fed-batch process over a given time interval.

    Args:
        ts0 (np.array): Time span for the simulation.
        Xo0 (np.array): Initial state vector.
        u0 (list): Control input parameters.
        THs (np.array): Model parameters.
        D0 (dict, optional): Dictionary containing dynamic conditions. Defaults to {}.

    Returns:
        tuple: A tuple containing the time vector and the simulated state variables.
    """
    TH1=THs[0:16]


    TH1=np.append(TH1,THs[16+int(u0[1])])
    TH1=np.append(TH1,THs[16+int(u0[1])+int(u0[2])]) 
    
    
    ts_start=ts0[0]
    ts_end=ts0[-1]
    
    time_pulse_all=np.array(D0['time_pulse'])
    Feed_pulse_all=np.array(D0['Feed_pulse'])
    
    t_u=time_pulse_all[(time_pulse_all>=ts_start) & (time_pulse_all<=ts_end)]
    uu_base_design=Feed_pulse_all[(time_pulse_all>=ts_start) & (time_pulse_all<=ts_end)]
 
    uu=uu_base_design

    if len(t_u)==0:
        t_u=np.array([ts_start,ts_end])
        uu=np.array([0,0])
    else:
        if ts_start<t_u[0]:
            t_u=np.append(ts_start,t_u)
            uu=np.append(0,uu)
        if ts_end>t_u[-1]:
            t_u=np.append(t_u,ts_end)
            uu=np.append(uu,0)

    Xo1=Xo0.copy()
    
    tt=np.array(ts_start)
    yy=np.array([Xo1])
    yy=yy.transpose()
    
    ni=0
    
    for i in uu[:-1]:
        ts1=np.linspace(t_u[ni],t_u[ni+1],25+1)
        Xo1[1]=Xo1[1]+uu[ni]*1e-6*u0[0]/0.01
        t,y=intM(ts1,Xo1,u0,TH1)
        Xo1=y[:,-1].copy()

        
        tt=np.append(tt,t[1:])
        yy=np.append(yy,y[:,1:],axis=1)
        ni=ni+1

    return tt,yy.transpose()
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
    TH=THo.copy()
    X = np.maximum(X, 1e-9)
    
    Xv=X[0]
    S=X[1]
    A=X[2]
    DOT=X[3] 
    P=X[4]
    mu_m=X[5]

    DOT = np.minimum(DOT, 100)
            
    qs_max=TH[0]
    fracc_q_ox_max=TH[1]
    qa_max=TH[2]

    
    Ys_ox=TH[4]
    Ya_p=TH[5]
    Ya_c=TH[6]
    Yo_ox=TH[8]
    Yo_a=TH[9]
    Yxs_of=TH[10]
    
    Ks=TH[11]
    n_ox=4
    
    Ka=TH[12]
    Ksi=TH[3]
    Kai=TH[7]
    Ko=0.1057
    
    kla=TH[16]
    k_sensor=TH[17]
    
    ky_1=TH[13] 
    ky_2=TH[14]
    ky_3=TH[15]
    
    DO_star=100
    H=13000#
    
    qs=qs_max*S/(S+Ks)*Ksi/(Ksi+A)
    q_ox_max=fracc_q_ox_max*qs_max
    
    q_ox_ss=qs*(1/((qs/q_ox_max)**n_ox+1))**(1/n_ox)
    qac_ss=qa_max*A/(A+Ka)*Kai/(Kai+S)
    b_ss=Ko+(q_ox_ss*Yo_ox+qac_ss*Yo_a)*Xv*H/kla-DO_star
    c_ss=-DO_star*Ko
    DOT_ss=(-b_ss+(b_ss*b_ss-4*c_ss)**.5)/2

    
    q_ox=qs*(1/((qs/q_ox_max)**n_ox+1))**(1/n_ox)*DOT_ss/(DOT_ss+Ko)
    q_of=qs-q_ox
    
    qac=qa_max*A/(A+Ka)*Kai/(Kai+S)*DOT_ss/(DOT_ss+Ko)
    
    qap=q_of*Ya_p
    
    mu=q_ox*Ys_ox+qac*Ya_c+Yxs_of*q_of

    if t>=u[3]:
        s_prod=u[4]
    else:
        s_prod=0
        
    
    f_qp=ky_1*mu_m/(mu_m+ky_2+(ky_3*mu_m)**2)

    q_prod=s_prod*f_qp


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



    sol=solve_ivp(lambda t,y: odeFB(t,y,TH0,u0) ,tspan,Xo1,method="BDF", rtol=1e-5, atol=1e-5,t_eval=ts0)
    y_interm=sol.y
    y_interm[y_interm<0]=0
    y_return=y_interm.copy()

    return sol.t,y_return