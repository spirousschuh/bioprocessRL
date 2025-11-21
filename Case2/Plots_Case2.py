from stable_baselines3 import PPO
import gymnasium as gym
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
import json
import os
from os import listdir
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

import KiwiGym_env_CS2


os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


# Plot training data and the fitted curve
def plot_model_training():

    sns.set_theme(style="darkgrid")

    # get event data from TensorBoard
    log_dir_step1 = "logs/hyperparameters/ns_20_lr_0.001_ec_0.001_cp_True_1"
    event_step1= EventAccumulator(log_dir_step1)
    event_step1.Reload()

    # get episode mean rewards
    reward1 = event_step1.Scalars("rollout/ep_rew_mean")
    reward_df1 = pd.DataFrame([(e.step, e.value) for e in reward1], columns=["step", "reward"])

    default_colors = sns.color_palette(n_colors=10)

    # Plot training data
    _, ax = plt.subplots()
    ax.plot(reward_df1["step"], reward_df1["reward"], label="Train", color=default_colors[9])

    # Plot Fit curve
    coef = np.polyfit(reward_df1["step"], reward_df1["reward"], deg=20)
    poly = np.poly1d(coef)

    x_fit = np.linspace(reward_df1["step"].min(), reward_df1["step"].max(), 100)
    y_fit = poly(x_fit)

    ax.plot(x_fit, y_fit, label="Fit", linestyle='--', color="#333333")

    # Set plot labels and legend
    # ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    plt.ylim(-1, 4.6)
    plt.xlabel(f"Steps", fontweight='bold')
    plt.ylabel(f"Mean Reward", fontweight='bold')
    plt.legend(fontsize=9, title="References", title_fontsize=10)
    plt.tight_layout()
    # plt.show()

    os.makedirs(os.path.dirname("plots/"), exist_ok=True)
    plt.savefig(f"plots/Fig_learning_curve.png", dpi=600)
    

# Auxiliar function to process the model name for better readability in the plot legend
def aux_process_model_name(model_name, reward_df):
    model_hyp = model_name.split('_')

    hyp_pairs = [f"{model_hyp[i]}: {model_hyp[i+1]}" for i in range(0, len(model_hyp) - 1, 2)]

    if hyp_pairs[0] == "ns: 20":
        hyp_pairs[0] = "ns: 160"
    else:
        hyp_pairs[0] = "ns: 240"

    if hyp_pairs[-1] == "cp: False":
        hyp_pairs[-1] = "nn: 128x128"
    else:
        hyp_pairs[-1] = "nn: 64x64"

    hyp_pairs.insert(0, "rew: {:.2f}".format(reward_df["reward"].iloc[-1]))

    return "  ".join(hyp_pairs)


# Plot all training logs from the specified directory
def plot_all_training_logs():

    sns.set_theme(style="darkgrid")

    dir_with_logs = "logs/hyperparameters/"
    model_list = listdir(dir_with_logs)
    colors = sns.color_palette("ch:start=.2,rot=-.3",  len(model_list) + 15)[3:]

    all_models = []
    names = []
    for model in model_list:
        try:
            print(model)

            # get event data from TensorBoard
            event= EventAccumulator(f"{dir_with_logs}{model}")
            event.Reload()

            # get episode mean rewards
            reward = event.Scalars("rollout/ep_rew_mean")
            reward_df = pd.DataFrame([(e.step, e.value) for e in reward], columns=["step", "reward"])

            # change model name for graph legend
            names.append(aux_process_model_name(model, reward_df))

            all_models.append(reward_df)
          
        except Exception as e:
            print(model, "Error:", e)

    # Sort models by the last reward value
    all_models = sorted(zip(names, all_models), key=lambda x: x[1]["reward"].iloc[-1])

    # Plot training data
    for idx, (model_name, reward_df) in enumerate(all_models):
        plt.plot(reward_df["step"], reward_df["reward"], label=model_name,    
                 color="#000000" if idx == len(all_models) - 1 else colors[idx])    
        
    # Get handles to invert legend order
    handles, labels = plt.gca().get_legend_handles_labels()
    
    # Set plot labels and legend
    plt.ylim(-1, 5)
    plt.xlabel(f"Steps", fontweight='bold')
    plt.ylabel(f"Mean Reward", fontweight='bold')
    plt.legend(handles[::-1], labels[::-1], fontsize=9, title="References", 
               title_fontsize=10, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

    # os.makedirs(os.path.dirname("plots/plots_3mbr/"), exist_ok=True)
    # plt.savefig(f"plots/plots_3mbr/Figure_learning_curve(new).png", dpi=600)


# Auxiliar function to get the species from the environment
def aux_get_species_from_env(env):
    result = {}
    for mbr in range(3):
        result[mbr] = {}
        for species in range(5):
            if species !=3:
                tt=env.unwrapped.kiwiGym.feed_profiles_history[mbr]['time_sample']
            else:
                tt=env.unwrapped.kiwiGym.feed_profiles_history[mbr]['time_sensor']
            
            result[mbr][species] = {"tt": tt, "X": env.unwrapped.kiwiGym.state['sample'][mbr][species]}
        
    return result

# Plot the results of the different models for one set of parameters (4F, 4F_0, 4F_no_actions)
def plot_model_comparative():  
    sns.set_theme(style="darkgrid")

    env = gym.make('kiwiGym-CS2')    
    load_dir = "saved_models/model_CS2/"  

    experiments = 100
    # experiments = 1
    models = ["model_CS2_0_final", "model_CS2_final", "no_agent"]
    results = {model_name: [] for model_name in models}

    for a in range(experiments):
        obs,_ = env.reset() 
        TH_env=env.unwrapped.kiwiGym.model_parameters

        for model_name in models:

            if model_name != "no_agent":
                model=PPO.load(os.path.join(load_dir,model_name),device="cpu")

            obs,_ = env.reset()         
            env.unwrapped.kiwiGym.model_parameters=TH_env

            while(True):
                if model_name == "no_agent":
                    action = [10, 10, 10]
                else:
                    action, _ = model.predict(obs,deterministic=True)  

                # print(action)
                obs, reward, terminated, _, _ = env.step(action)

                if(terminated):
                    break

            # get results
            results[model_name].append(aux_get_species_from_env(env))
    
    # Plot results - https://seaborn.pydata.org/tutorial/color_palettes.html
    default_colors = sns.color_palette(n_colors=10)
    colors = [default_colors[7], default_colors[8], default_colors[9]]

    for model_name in models:
        for species, sp_name in enumerate(["Biomass", "Glucose", "Acetate", "DOT", "Fluo_RFP"]):
            for it in range(experiments):
                for mbr in range(3):
                    # plt.plot(results[model_name][it][mbr][species]["tt"], results[model_name][it][mbr][species]["X"], color=colors[mbr])
                    plt.plot(results[model_name][it][mbr][species]["tt"], results[model_name][it][mbr][species]["X"], '.', color=colors[mbr])

            if species ==3:
                plt.ylim(0, 105)
                plt.axhline(y=20, color="#A8A5A5", linestyle='--')
                plt.text(x=0.2, y=20 + 1.5, s="DOT constraint",   color='#A8A5A5', fontsize=9)

            plt.xlabel(f"Time $[h]$", fontweight='bold')
            plt.ylabel(f"{sp_name}", fontweight='bold')
            legend_elements = [Line2D([0], [0], marker='.', color='none', label=f"MBR {i+1}",
                    markerfacecolor=colors[i], markersize=10, markeredgecolor="none") for i in range(3)]
            plt.legend(handles=legend_elements, fontsize=9, title="References", title_fontsize=10)
            plt.tight_layout()
            # plt.show()

            os.makedirs(os.path.dirname("plots/"), exist_ok=True)
            plt.savefig(f"plots/{model_name}_{sp_name}.png", dpi=600)
            plt.clf()


# plot_all_training_logs()
# plot_model_training()
plot_model_comparative()
