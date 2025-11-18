import numpy as np
import time
import method_kiwiGym as method_kiwiGym

# %% Create design
n_exp=3
t_final=14

ts=np.array([0,1])
pulses_time=np.arange(5+5/60,t_final,10/60)

initial_states={'state':{},'sample':{}}
control_inputs={}
feed_profiles={}
feed_profilesj={}

sample_offsets=[0.33,0.66,0.99]


ux=np.array([0.14529732, 0.075    ,  0.11614164])
 
for i in range(n_exp):
    initial_states['t']=ts[0]
    initial_states['state'][i]=[0.18,4,0,100,0,.01*0]
    initial_states['sample'][i]={}
    control_inputs[i]=[n_exp,200,10]

    feed_profile_i=(36.33*0+32.406)*ux[i]*np.exp(ux[i]*(pulses_time-pulses_time[0]))
    feed_profile_i=np.round(feed_profile_i*2)/2
    feed_profile_i[feed_profile_i<5]=5

    
    feed_profiles[i]={'time_pulse':pulses_time.tolist(),'Feed_pulse':feed_profile_i.tolist(),'time_sample':np.arange(0,t_final,1)+sample_offsets[i],#np.arange(8,16.1,8),#
           'time_sensor':np.linspace(0.04,t_final,2*25*round(t_final))}
    feed_profilesj[i]={'time_pulse':pulses_time.tolist(),'Feed_pulse':feed_profile_i.tolist(),'time_sample':np.arange(0,t_final,1)+sample_offsets[i],#np.arange(8,16.1,8),#
           'time_sensor':np.linspace(0.04,t_final,2*25*round(t_final))}    

model_parameters=np.array([1.2578, 0.43041, 0.6439,  7.0767,  0.4063,  0.1143*4,  0.1848*4,    .4242,    1.586*.7, 1.5874*.7,  0.3322*.75,  0.0371,  0.0818,    9000, .1, 5]+[850]*n_exp+[90]*n_exp)
# %% 

n_sample=len(feed_profiles[0]['time_sample'])
n_sensor=len(feed_profiles[0]['time_sensor'])
sd_meas=np.array(([.2]*n_sample+[.2]*n_sample+[.2]*n_sample+[5]*n_sensor+[50]*n_sample)*1)  #*n_exp
C2=np.diag(sd_meas**2)


sd_measj=np.array(([.2]*n_sample+[.2]*n_sample+[.2]*n_sample+[5]*len(feed_profilesj[0]['time_sensor'])+[50]*n_sample)*1)  #*n_exp
C2j=np.diag(sd_measj**2)

THsd0=model_parameters[0:18]*0+.1
THsd0[13]=.5
THsd0[14]=.5
THsd0[15]=.5
THsd0[16]=.5
THsd0[17]=.5
Cov_TH0=np.diag(THsd0**2)


t_check1=time.time()
Si,Q,FIM,state,traceFIM,FIM_crit,ei=method_kiwiGym.calculate_FIM(np.array([0,t_final]),initial_states,control_inputs,model_parameters,feed_profiles,C2,Cov_TH0)
t_check2=time.time()
print(t_check2-t_check1)
print('FIM crit ',FIM_crit)
TH_sd=(np.diag(np.linalg.inv(FIM)))**.5


# %% 
LB=[0.075]*n_exp
UB=[0.15]*n_exp

optim_options=[50, 10]
u_opt=method_kiwiGym.optimizer_reference(LB,UB,optim_options,np.array([0,t_final]),initial_states,control_inputs,model_parameters,feed_profiles,C2,Cov_TH0)

print('Optimal mu_set: ',u_opt)