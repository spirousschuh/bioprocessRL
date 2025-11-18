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
            time_current=0,
            number_mbr=1,
            time_final=14,
            time_step=1,
            sample_schedule=[0.99, 0.99, 0.99],
            time_batch=5,
            mu_reference=[0.145],
            TH_param0=None,
    ):

        #Define Model Parameters
        self.TH_param = np.array(TH_param0) or np.array([1.2578, 0.43041, 0.6439,  7.0767,  0.4063,  0.1143*4,  0.1848*4,    .4242,    1.586*.7, 1.5874*.7,  0.3322*.75,  0.0371,  0.0818,    9000, .1, 5]+[850]*number_mbr+[90]*number_mbr)

        self.number_mbr=number_mbr
        self.time_final=time_final
        self.time_current=time_current
        self.time_step=time_step
        self.time_interval=np.array([time_current,time_current+self.time_step])
        self.time_pulses=np.arange(time_batch+5/60,time_final,10/60)
        self.sample_schedule=sample_schedule
        self.mu_reference=np.array(mu_reference)    
        
        # MBR specific variables
        XX0={'state':{},'sample':{}} #States and samples
        uu={} #Fixed process variables
        DD={} #Profile process variables
    
        for i in range(self.number_mbr):
            XX0['t']=self.time_interval[0]
            XX0['state'][i]=[0.18,4,0,100,0,.01]
            XX0['sample'][i]={0:[],1:[],2:[],3:[],4:[],}

            uu[i] = method_kiwiGym.ControlInputs(
                experiment_index=i,
                num_experiments=self.number_mbr,
                feed_concentration=200,
                induction_time=10.,
                product_switch=0,
            )

            
            feed_profile_i=(32.406)*self.mu_reference[i]*np.exp(self.mu_reference[i]*(self.time_pulses-self.time_pulses[0]))
            feed_profile_i=np.round(feed_profile_i*2)/2
            feed_profile_i[feed_profile_i<5]=5
            
            DD[i]={
                'time_pulse':self.time_pulses.tolist(),
                'Feed_pulse':feed_profile_i.tolist(),
                'time_sample':np.arange(self.time_final)+self.sample_schedule[i],
                'time_sensor':np.linspace(0.04,self.time_final,25*round(self.time_final)),
            }
            

        self.XX0=deepcopy(XX0)
        self.XX=deepcopy(XX0)
        self.uu=uu
        self.DD=DD
        self.DD_historic=deepcopy(self.DD)
        
        #KiwiGymEnv variables
        self.terminated=False
        self.obs=np.zeros([self.uu[0][0]*(4+25*0+1)])
        return
# %%    
    def reset(self, seed=None,TH_param=[]):
        #Change parameters
        if len(TH_param)>0:
            self.TH_param=TH_param
        
        #Reset time    
        self.time_current=0
        self.time_interval=np.array([self.time_current,self.time_current+self.time_step])
        
        #Reset States, profiles and Env variables          
        XX0={'state':{},'sample':{}}
        for i in range(self.number_mbr):
            XX0['t']=self.time_interval[0]
            XX0['state'][i]=[0.18,4,0,100,0,.01]
            XX0['sample'][i]={0:[],1:[],2:[],3:[],4:[],}
        self.XX=deepcopy(XX0)
        self.DD_historic=deepcopy(self.DD)
        self.obs=np.zeros([self.uu[0].num_experiments*(4+25*0+1)])#.tolist()
        self.terminated=False
        return 
# %%    
    def render(self):
        #Show DOT and Biomass
        for i2 in range(self.uu[0].num_experiments):
            plt.plot(self.XX['sample'][i2][3],'.')#
        plt.show()
        for i2 in range(self.uu[0].num_experiments):
            plt.plot(self.XX['sample'][i2][0],'o')
        plt.show()
        print('time: ',self.time_current,' done: ',self.terminated,'reward: ',self.reward)

# %%    
    def perform_action(self,action_step=[]):

        # If there is no action, use the reference profile. Else, modify the current profile.
        if action_step is list:
            DD_action=deepcopy(self.DD)
        else:
            DD_action=deepcopy(self.DD_historic)

            action=action_step
            time_step_before=1
            for i in range(self.uu[0].num_experiments):
                # print(DD_action)
                t_pulse=np.array(DD_action[i]['time_pulse'])
                DD_ref=np.array(DD_action[i]['Feed_pulse'])
                
                DD_change=np.zeros(DD_ref.shape)
                
                DD_change[(t_pulse<=(self.time_interval[1]+time_step_before)) & (t_pulse>=(self.time_interval[0]+time_step_before))]=action
                
                DD_corrected=DD_ref+DD_change
                DD_corrected[(t_pulse>=t_pulse[0]) & (DD_corrected<5)]=5 #After that, min 5 ul
                
                DD_action[i]['Feed_pulse']=(DD_corrected).tolist()

                
        self.DD_historic=deepcopy(DD_action)

        #Apply action during time interval

        XX_plus1 = method_kiwiGym.simulate_parallel(
            self.time_interval,
            self.XX,
            self.uu,
            self.TH_param,
            self.DD_historic,
        )
        self.XX=XX_plus1
        self.time_current=self.time_interval[1]
        
        ################ Construct observation vector
        if len(self.obs)==0:
            XX_obs=np.zeros([self.uu[0][0]*(4+1)]) #
        else:
            XX_obs=np.array(self.obs)
            
        XX_obs=XX_obs[:,None]
        x3=[]
        for i1 in range(self.uu[0].num_experiments):
            for i2 in [0,3]:
                if i2==0:
                    t1=np.array(self.DD_historic[i1]['time_sample'])
                else:
                    t1=np.array(self.DD_historic[i1]['time_sensor'])
                    
                x1=np.array(XX_plus1['sample'][i1][i2])
                t1b=t1[t1<=self.time_interval[1]]
                x1b=x1[(t1b>self.time_interval[0]) & (t1b<=self.time_interval[1])]
                
                if i2==3:
                    x1b=np.array([np.min(x1b)])

                x2=x1b[:,None]
    
                if len(x3)==0:
                    x3=x2
                else:
                    x3=np.vstack((x3,x2))

        XX_obs=x3
        self.obs=XX_obs.flatten()#.tolist()
        ################
        #Update time interval, terminated state and calculate reward
        self.time_interval=np.array([self.time_current,self.time_current+self.time_step])

        if self.time_current>=self.time_final:
            self.terminated=True
            
            n_sample=len(self.DD[0]['time_sample'])
            n_sensor=len(self.DD[0]['time_sensor'])
            sd_meas=np.array(([.2]*n_sample+[.2]*n_sample+[.5]*n_sample+[5]*n_sensor+[20]*n_sample)*1)  #*n_exp
            C2=np.diag(sd_meas**2)

            
            # #Biomass 
            XX,DIV_min=method_kiwiGym.calculate_DIV(np.array([0,self.time_final]),self.XX0,self.uu,self.TH_param,self.DD_historic,C2)

            dot_min=min(XX['sample'][0][3])

            if dot_min<20:
                DIV_constrain=((20-dot_min)*.05+1)**2
            else:
                DIV_constrain=1
                
            DIV_constr=np.array(DIV_constrain)
            DIV_calculated=DIV_min/DIV_constr
            DIV_normalized=(DIV_calculated-7.065)/7.065
            self.reward=DIV_normalized
            

            print("reward: ", self.reward , "Biomass: ", DIV_min, "Constrain: ", DIV_constrain)

        else:
            self.terminated=False
            self.reward=0
        return self.obs, self.reward, self.terminated#, 