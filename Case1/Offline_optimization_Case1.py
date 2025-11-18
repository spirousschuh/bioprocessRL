# %% Import libraries
import numpy as np
import time
import method_kiwiGym

# %% Create design
# Set up experiment parameters
n_exp=1 # Number of experiments
t_final=14 # Final time in days
ts=np.array([0,1]) # Time span for the simulation
pulses_time=np.arange(5+5/60,t_final,10/60) # Time points for feed pulses

# Initialize dictionaries for initial states, control inputs, and dynamic conditions
initial_states={'state':{},'sample':{}}
control_inputs={}
feed_profiles={}

# Define a sample schedule
sample_offsets=[0.99]

# Initial guess for the specific growth rate (mu_set)
ux=np.array([ 0.14790724  ])

# Loop to set up each experiment
for i in range(n_exp):
    initial_states['t']=ts[0] # Initial time
    initial_states['state'][i]=[0.18,4,0,100,0,.0] # Initial state: [Xv, S, A, DOT, P, mu_m]
    initial_states['sample'][i]={} # Initialize sample dictionary for the experiment
    # control_inputs[i]=[n_exp,200,10] # Control input parameters
    control_inputs[i] =method_kiwiGym.ControlInputs(
        num_experiments=n_exp,
        feed_concentration=200,
        experiment_index=i,
        induction_time=10.,
        product_switch=0,
    )
    # Calculate the feed profile based on an exponential growth model
    feed_profile_i=(32.406)*ux[i]*np.exp(ux[i]*(pulses_time-pulses_time[0]))
    feed_profile_i=np.round(feed_profile_i*2)/2 # Round the feed rate
    feed_profile_i[feed_profile_i<5]=5 # Set a minimum feed rate

    # Store dynamic conditions for the experiment
    feed_profiles[i]={
        'time_pulse':pulses_time.tolist(), # Time points for feed pulses
        'Feed_pulse':feed_profile_i.tolist(), # Feed rate at each pulse
        'time_sample':np.arange(0,t_final,1)+sample_offsets[i], # Time points for sampling
        #np.arange(8,16.1,8),
        'time_sensor':np.linspace(0.04,t_final,25*round(t_final)), # Time points for sensor measurements
    }
    
# Define model parameters (model_parameters)
model_parameters=np.array(
    [1.2578, 0.43041, 0.6439,  7.0767,  0.4063,  0.1143*4,  0.1848*4,    .4242,    1.586*.7, 1.5874*.7,  0.3322*.75,  0.0371,  0.0818,    9000, .1, 5]+[850]*n_exp+[90]*n_exp
)

# %% Set up and run optimization
t_check1=time.time() # Start time for performance measurement
n_sample=len(feed_profiles[0]['time_sample']) # Number of samples
n_sensor=len(feed_profiles[0]['time_sensor']) # Number of sensor measurements
# Define standard deviation of measurements for constructing the covariance matrix
sd_meas=np.array(([.2]*n_sample+[.2]*n_sample+[.5]*n_sample+[5]*n_sensor+[50]*n_sample)*1)
C2=np.diag(sd_meas**2) # Covariance matrix of measurement errors

# Define bounds for the optimization variable (mu_set)
LB=[0.075]*n_exp # Lower bound
UB=[0.15]*n_exp # Upper bound
# Define optimization options [global optimization time, local optimization time]
optim_options=[25, 5]

# Call the optimizer function to find the optimal mu_set
u_opt=method_kiwiGym.optimizer_reference(LB,UB,optim_options,np.array([0,t_final]),initial_states,control_inputs,model_parameters,feed_profiles,C2)

# Print the optimal mu_set
print('Optimal mu_set: ',u_opt)
