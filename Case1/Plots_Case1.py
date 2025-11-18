from stable_baselines3 import PPO
import gymnasium as gym

import matplotlib.pyplot as plt
import seaborn as sns
import os

from Case1 import KiwiGym_env_CS1


os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


# Auxiliar function to get the species from the environment
def aux_get_species_from_env(env):
    mbr = 0
    result = {}
    for species in range(5):
        if species !=3:
            tt=env.unwrapped.kiwiGym.feed_profiles_history[mbr]['time_sample']
        else:
            tt=env.unwrapped.kiwiGym.feed_profiles_history[mbr]['time_sensor']
        
        result[species] = {"tt": tt, "X": env.unwrapped.kiwiGym.state['sample'][mbr][species]}
        
    return result

# Plot the results of the different models for one set of parameters (4F, 4F_0, 4F_no_actions)
def plot_model_comparative():  
    sns.set_theme(style="darkgrid")

    load_dir = "saved_models/model_CS1"
    
    env = gym.make('kiwiGym-CS1') 
    obs,_=env.reset()    

    experiments = 100
    models = ["model_CS1_0_final", "model_CS1_final", "no_agent"]            
    results = {model_name: [] for model_name in models}

    for _ in range(experiments):
        obs,_ = env.reset() 
        TH_env=env.unwrapped.kiwiGym.model_parameters
    
        for model_name in models:

            if model_name != "no_agent":
                model=PPO.load(os.path.join(load_dir,model_name),device="cuda")

            obs,_ = env.reset()         
            env.unwrapped.kiwiGym.model_parameters=TH_env

            while(True):
                if model_name == "no_agent":
                    action = 10
                else:
                    action, _ = model.predict(obs,deterministic=True)  

                obs, reward, terminated, _, _ = env.step(action)

                if(terminated):
                    break

            # get results
            results[model_name].append(aux_get_species_from_env(env))

    # Plot results
    for model_name in models:
        for species, sp_name in enumerate(["Biomass", "Glucose", "Acetate", "DOT", "Fluo_RFP"]):
            for it in range(experiments):
                plt.plot(results[model_name][it][species]["tt"], results[model_name][it][species]["X"], '.')

            if species == 3:
                plt.ylim(0, 105)
                plt.axhline(y=20, color="#A8A5A5", linestyle='--')
                plt.text(x=0.2, y=20 + 1.5, s="DOT constraint",   color='#A8A5A5', fontsize=9)

            plt.xlabel(f"Time $[h]$", fontweight='bold')
            plt.ylabel(f"{sp_name}", fontweight='bold')
            # plt.legend(fontsize=9, title="References", title_fontsize=10)
            plt.tight_layout()
            # plt.show()

            os.makedirs(os.path.dirname("plots/"), exist_ok=True)
            plt.savefig(f"plots/{model_name}_{sp_name}.png", dpi=600)
            plt.clf()


plot_model_comparative()
