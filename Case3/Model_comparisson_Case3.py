import gymnasium as gym
import numpy as np
import os
from stable_baselines3 import PPO
import KiwiGym_env_CS3
# %%
if __name__=="__main__":


    reward_CS3=[]
    reward_CS3_0=[]
    reward_NA=[]
    
    load_dir = "saved_models/model_CS3"
    model_name="model_CS3_final"
    model=PPO.load(os.path.join(load_dir,model_name),device="cpu")
    
    load_dir_base = "saved_models/model_CS3"
    model_name_0="model_CS3_0_final"
    model_0=PPO.load(os.path.join(load_dir_base,model_name_0),device="cpu")
    #######
    for i in range(100):
        print("iter: ", i)
        env = gym.make('kiwiGym-CS3') 
        obs,_=env.reset()    
        TH_env=env.unwrapped.kiwiGym.model_parameters

        while(True):
            action, _ = model.predict(obs,deterministic=True)  
            print(action)
            obs, reward, terminated, _, _ = env.step(action)
            
            if(terminated):
                reward_CS3.append(reward)
                break
        #######
        obs,_=env.reset() 
        env.unwrapped.kiwiGym.model_parameters=TH_env
        while(True):
            action_0, _ = model_0.predict(obs,deterministic=True)  
            obs, reward, terminated, _, _ = env.step(action_0)
            print(action)
            if(terminated):
                print("########")
                reward_CS3_0.append(reward)
                break      
        #######
        obs,_=env.reset() 
        env.unwrapped.kiwiGym.model_parameters=TH_env
        while(True):
            obs, reward, terminated, _, _ = env.step([10,10,10])
            if(terminated):
                print("########")
                reward_NA.append(reward)
                break 
    
    
    print(" mean reward of each agent: ")
    print("reward_CS3_mean: ",np.mean(reward_CS3),"reward_CS3_0_mean: ",np.mean(reward_CS3_0),"reward_NA_mean: " ,np.mean(reward_NA))
    # print(" std ")
    # print("reward_CS3_std: ",np.std(reward_CS3),"reward_CS3_0_std: ",np.std(reward_CS3_0),"reward_NA_std: " ,np.std(reward_NA))