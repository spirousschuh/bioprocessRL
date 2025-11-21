import numpy as np
import time
import method_kiwiGym

# %% Create design
n_exp=3

t_final=14

ts=np.array([0,1])
pulses_time=np.arange(5+5/60,t_final,10/60)


initial_states={'state':{},'sample':{}}
control_inputs={}
feed_profiles={}


sample_offsets=[0.33,0.66,0.99]

ux=np.array([0.12846724, 0.14790724, 0.07986599 ])


for i in range(n_exp):
    initial_states['t']=ts[0]
    initial_states['state'][i]=[0.18,4,0,100,0,.0]
    initial_states['sample'][i]={}
    control_inputs[i]=[n_exp,200,10]


    feed_profile_i=(32.406)*ux[i]*np.exp(ux[i]*(pulses_time-pulses_time[0]))
    feed_profile_i=np.round(feed_profile_i*2)/2
    feed_profile_i[feed_profile_i<5]=5

    
    feed_profiles[i]={'time_pulse':pulses_time.tolist(),'Feed_pulse':feed_profile_i.tolist(),'time_sample':np.arange(0,t_final,1)+sample_offsets[i],#np.arange(8,16.1,8),#
           'time_sensor':np.linspace(0.04,t_final,25*round(t_final))}
    

model_parameters=np.array([1.2578, 0.43041, 0.6439,  7.0767,  0.4063,  0.1143*4,  0.1848*4,    .4242,    1.586*.7, 1.5874*.7,  0.3322*.75,  0.0371,  0.0818,    9000, .1, 5]+[850]*n_exp+[90]*n_exp)

# %% 
t_check1=time.time()
n_sample=len(feed_profiles[0]['time_sample'])
n_sensor=len(feed_profiles[0]['time_sensor'])
sd_meas=np.array(([.2]*n_sample+[.2]*n_sample+[.5]*n_sample+[5]*n_sensor+[50]*n_sample)*1)  
C2=np.diag(sd_meas**2)

state,DIV,DIV_min=method_kiwiGym.calculate_DIV(np.array([0,t_final]),initial_states,control_inputs,model_parameters,feed_profiles,C2)


print(ux)
print('DIV ',DIV,DIV_min)

t_check2=time.time()
print(t_check2-t_check1)

# %% 
LB=[0.075]*n_exp
UB=[0.15]*n_exp
optim_options=[25, 5]
u_opt=method_kiwiGym.optimizer_reference(LB,UB,optim_options,np.array([0,t_final]),initial_states,control_inputs,model_parameters,feed_profiles,C2)

print('Optimal mu_set: ',u_opt)
