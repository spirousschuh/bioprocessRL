
import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register
from gymnasium.utils.env_checker import check_env

import numpy as np

from kiwiGym_CS3 import kiwiGym

# %%
register(
    id='kiwiGym-CS3_0',                                
    entry_point='KiwiGym_env_CS3_0:kiwiGymEnv_CS3_0', 
)

class kiwiGymEnv_CS3_0(gym.Env):

    metadata = {"render_modes": ["human"], 'render_fps': 4}


    def __init__(self, render_mode=None):
        self.render_mode = render_mode

        self.kiwiGym=kiwiGym()

        self.action_values = np.arange(-5, 5.5, 0.5)
        self.action_space = spaces.MultiDiscrete([len(self.action_values)] * self.kiwiGym.num_experiments)  # Actions are indices

        e_vector=np.array([14])
        d_vector=np.tile([21],(self.kiwiGym.final_time-int(self.kiwiGym.pulses_time[0]))*self.kiwiGym.num_experiments)
        y_vector=np.tile([20,10,10]+[105]*1+[200e3],(self.kiwiGym.final_time)*self.kiwiGym.num_experiments)
        self.observation_upper_bound=np.concatenate([e_vector,d_vector,y_vector])
        
        self.observation_space = spaces.Box(
            low=(0)*np.ones(len(self.observation_upper_bound)),
            high=(1)*np.ones(len(self.observation_upper_bound)),
            dtype=np.float64
        )

    def reset(self, seed=None, options=None):
        
        super().reset(seed=seed) 
        if seed is not None:
           np.random.seed(seed)

        model_parameters_mean=np.array([[1.2578, 0.43041, 0.6439,  7.0767,  0.4063,  0.1143*4,  0.1848*4,    .4242,    1.586*.7, 1.5874*.7,  0.3322*.75,  0.0371,  0.0818,    9000, .1, 5]]+[850]*3+[90]*3)
        TH_reset=model_parameters_mean*(1+0.00*(np.random.random(len(model_parameters_mean))-.5)/2)

        self.kiwiGym.reset(seed=seed,model_parameters=TH_reset)

        obs = np.ones(len(self.observation_upper_bound))*(0)
        
        time_step_before=1
        obs[0]=int(self.kiwiGym.pulses_time[0]-time_step_before)# No encoding
        
        # Integrate up to time batch
        while self.kiwiGym.current_time<round(self.kiwiGym.pulses_time[0]-time_step_before):

            action_val = [10,10,10]
            obs_raw,_,_ = self.kiwiGym.perform_action(action_val)
            index_obs=np.arange(5*self.kiwiGym.num_experiments)+5*self.kiwiGym.num_experiments*(self.kiwiGym.current_time-1)+(self.kiwiGym.num_experiments)*(self.kiwiGym.final_time-int(self.kiwiGym.pulses_time[0]))+1
            obs[index_obs]=obs_raw
        self.obs=obs
        
        info = {}

        
        obs_corrected=obs/self.observation_upper_bound
        return obs_corrected, info

    def step(self, action):
        # Perform action
        action_val = [self.action_values[i] for i in action]
        obs_raw,reward,terminated = self.kiwiGym.perform_action(action_val)

        obs=self.obs
        #Take the position of obs_norm and allocate in self.obs
        obs[0]=self.kiwiGym.current_time 
        
        index_action=np.arange(self.kiwiGym.num_experiments)+self.kiwiGym.num_experiments*(self.kiwiGym.current_time-int(self.kiwiGym.pulses_time[0])-0)+1
        obs[index_action]=np.array(action)+1 #actions start in 1
        
        index_obs=np.arange(5*self.kiwiGym.num_experiments)+5*self.kiwiGym.num_experiments*(self.kiwiGym.current_time-1)+(self.kiwiGym.num_experiments)*(self.kiwiGym.final_time-int(self.kiwiGym.pulses_time[0]))+1
        obs[index_obs]=obs_raw

        self.obs=obs
        
        info = {}

        # Render environment
        if (self.render_mode=='human') & (terminated==True):
            print('Action: ',action_val)
            self.render()

        obs_corrected=obs/self.observation_upper_bound
        return obs_corrected, reward, terminated, False, info

    def render(self):
        self.kiwiGym.render()

# %% For unit testing
if __name__=="__main__":
    env = gym.make('kiwiGym-CS3_0')

    # Use this to check our custom environment
    print("Check environment begin")
    check_env(env.unwrapped)
    print("Check environment end")

    # Reset environment
    obs = env.reset()[0]


    index_iter=np.arange(3*(5+1))
    cnt=0
    while(cnt<3):

        rand_action =np.array([1,1,1])
        obs, reward, terminated, _, _ = env.step(rand_action)
        print(reward,rand_action)

        if(terminated):
            print(env.unwrapped.kiwiGym.model_parameters[0])
            env.render()
            
            model_parameters=np.array([1.2578, 0.43041, 0.6439,  7.0767,  0.4063,  0.1143*4,  0.1848*4,    .4242,    1.586*.7, 1.5874*.7,  0.3322*.75,  0.0371,  0.0818,    9000, .1, 5]+[850]*3+[90]*3)
            obs = env.reset()[0]
        cnt+=1
    

