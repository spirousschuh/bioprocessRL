# %% Import
import numpy as np
from copy import deepcopy

import method_kiwiGym

import matplotlib.pyplot as plt
# %%
    
class kiwiGym:
    def __init__(self,render_mode=None,model_parameters0=[]):

        current_time=0
        num_experiments=3
        final_time=14
        time_step=1
        sample_offsets=[0.33,0.66,0.99]
        time_batch=5
        mu_reference=[0.12846724, 0.14790724, 0.07986599]
        
        #Define Model Parameters
        if len(model_parameters0)==0:
            self.model_parameters=np.array([1.2578, 0.43041, 0.6439,  7.0767,  0.4063,  0.1143*4,  0.1848*4,    .4242,    1.586*.7, 1.5874*.7,  0.3322*.75,  0.0371,  0.0818,    9000, .1, 5]+[850]*num_experiments+[90]*num_experiments)
        else:
            self.model_parameters=np.array(model_parameters0)   

        self.num_experiments=num_experiments
        self.final_time=final_time
        self.current_time=current_time
        self.time_step=time_step
        self.time_interval=np.array([current_time,current_time+self.time_step])
        self.pulses_time=np.arange(time_batch+5/60,final_time,10/60)
        self.sample_offsets=sample_offsets
        self.mu_reference=np.array(mu_reference)    
        
        # MBR specific variables
        initial_states={'state':{},'sample':{}} #States and samples
        control_inputs={} #Fixed process variables
        feed_profiles={} #Profile process variables
    
        for i in range(self.num_experiments):
            initial_states['t']=self.time_interval[0]
            initial_states['state'][i]=[0.18,4,0,100,0,.0]
            initial_states['sample'][i]={0:[],1:[],2:[],3:[],4:[],}
            control_inputs[i]=[self.num_experiments,200,10]
            
            feed_profile_i=(32.406)*self.mu_reference[i]*np.exp(self.mu_reference[i]*(self.pulses_time-self.pulses_time[0]))
            feed_profile_i=np.round(feed_profile_i*2)/2
            feed_profile_i[feed_profile_i<5]=5
            
            feed_profiles[i]={'time_pulse':self.pulses_time.tolist(),'Feed_pulse':feed_profile_i.tolist(),'time_sample':np.arange(self.final_time)+self.sample_offsets[i],
                   'time_sensor':np.linspace(0.04,self.final_time,25*round(self.final_time))}

        self.initial_states=deepcopy(initial_states)
        self.state=deepcopy(initial_states)
        self.control_inputs=control_inputs
        self.feed_profiles=feed_profiles
        self.feed_profiles_history=deepcopy(self.feed_profiles)
        
        #KiwiGymEnv variables
        self.terminated=False
        self.obs=np.zeros([self.control_inputs[0][0]*(1+1)])
        return
# %%    
    def reset(self, seed=None,model_parameters=[]):
        #Change parameters
        if len(model_parameters)>0:
            self.model_parameters=model_parameters
        
        #Reset time    
        self.current_time=0
        self.time_interval=np.array([self.current_time,self.current_time+self.time_step])
        
        initial_states={'state':{},'sample':{}}
        for i in range(self.num_experiments):
            initial_states['t']=self.time_interval[0]
            initial_states['state'][i]=[0.18,4,0,100,0,.0]
            initial_states['sample'][i]={0:[],1:[],2:[],3:[],4:[],}
        self.state=deepcopy(initial_states)
        self.feed_profiles_history=deepcopy(self.feed_profiles)
        self.obs=np.zeros([self.control_inputs[0][0]*(4*0+25*0+1+1)])#.tolist()
        self.terminated=False
        return 
# %%    
    def render(self):
        #Show DOT and Biomass
        for i2 in range(self.control_inputs[0][0]):
            plt.plot(self.state['sample'][i2][3],'.')
        plt.show()
        for i2 in range(self.control_inputs[0][0]):
            plt.plot(self.state['sample'][i2][0],'o')
        plt.show()
        print('time: ',self.current_time,' done: ',self.terminated,'reward: ',self.reward)

# %%    
    def perform_action(self,action_step=[]):

        # If there is no action, use the reference profile. Else, modify the current profile.
        if len(action_step)==0:
            feed_profiles_action=deepcopy(self.feed_profiles)
        else:
            feed_profiles_action=deepcopy(self.feed_profiles_history)

            action=action_step
            time_step_before=1
            for i in range(self.control_inputs[0][0]):
                t_pulse=np.array(feed_profiles_action[i]['time_pulse'])
                feed_profiles_ref=np.array(feed_profiles_action[i]['Feed_pulse'])
                
                feed_profiles_change=np.zeros(feed_profiles_ref.shape)
                
                feed_profiles_change[(t_pulse<=(self.time_interval[1]+time_step_before)) & (t_pulse>=(self.time_interval[0]+time_step_before))]=action[i]
                
                feed_profiles_corrected=feed_profiles_ref+feed_profiles_change
                
                feed_profiles_corrected[(t_pulse>=t_pulse[0]) & (feed_profiles_corrected<5)]=5 

                feed_profiles_action[i]['Feed_pulse']=(feed_profiles_corrected).tolist()

                
        self.feed_profiles_history=deepcopy(feed_profiles_action)

        #Apply action during time interval
        state_plus1=method_kiwiGym.simulate_parallel(self.time_interval,self.state,self.control_inputs,self.model_parameters,self.feed_profiles_history)
        self.state=state_plus1
        self.current_time=self.time_interval[1]
        
        ################ Construct observation vector
        if len(self.obs)==0:
            state_obs=np.zeros([self.control_inputs[0][0]*(1+1)]) 
        else:
            state_obs=np.array(self.obs)
            
        state_obs=state_obs[:,None]
        x3=[]
        for i1 in range(self.control_inputs[0][0]): 
            for i2 in [0,3]:
                if i2==0:
                    t1=np.array(self.feed_profiles_history[i1]['time_sample'])
                elif i2==3:
                    t1=np.array(self.feed_profiles_history[i1]['time_sensor'])
                    
                x1=np.array(state_plus1['sample'][i1][i2])
                t1b=t1[t1<=self.time_interval[1]]
                x1b=x1[(t1b>self.time_interval[0]) & (t1b<=self.time_interval[1])]
                
                if i2==3:
                    x1b=np.array([np.min(x1b)])

                x2=x1b[:,None]
    
                if len(x3)==0:
                    x3=x2
                else:
                    x3=np.vstack((x3,x2))

        state_obs=x3
        self.obs=state_obs.flatten()#.tolist()
        ################
        self.time_interval=np.array([self.current_time,self.current_time+self.time_step])

        if self.current_time>=self.final_time:
            self.terminated=True
            
            n_sample=len(self.feed_profiles[0]['time_sample'])
            n_sensor=len(self.feed_profiles[0]['time_sensor'])
            sd_meas=np.array(([.2]*n_sample+[.2]*n_sample+[.5]*n_sample+[5]*n_sensor+[50]*n_sample)*1)  #*n_exp
            C2=np.diag(sd_meas**2)
            
            # #Biomass profile divergence
            state,DIV,DIV_min=method_kiwiGym.calculate_DIV(np.array([0,self.final_time]),self.initial_states,self.control_inputs,self.model_parameters,self.feed_profiles_history,C2)
            DIV_constrain=[]
            DOT_min=[] 
            Glc_max=[]
            
            for i2 in range(self.control_inputs[0][0]): 
                dot_min=min(state['sample'][i2][3])
                DOT_min.append(dot_min)

                if dot_min<20:
                    dot_constrain=((20-dot_min)*.50+1)**2
                else:
                    dot_constrain=1  
                                       
                glc_constrain=0
                DIV_constrain.append(dot_constrain+glc_constrain)
                
            DIV_constr=np.array(DIV_constrain)
            DIV_calculated=DIV_min*3/np.sum(DIV_constr)
            DIV_normalized=(DIV_calculated-1.1)/1.1
            self.reward=DIV_normalized
            
            print('calculating reward...')
            print("reward: ",self.reward, "div: ",DIV_min ,"dot: ",min(DOT_min))#
        else:
            self.terminated=False
            self.reward=0
        return self.obs, self.reward, self.terminated