import numpy as np
import json

config = dict()

# MBRs definition
config["runID"] = 623
config["exp_ids"] = np.arange(19419, 19443).tolist()
config["num_experiments"] = 3
config["mbr_groups"] = np.array(config["exp_ids"]).reshape(8, 3, order='F').tolist()

# other variables
config["iter"] = 0
config["time_batch"] = 5
config["final_time"] = 14

config["action_values"] = np.arange(-5, 5.5, 0.5).tolist()
config["mu_reference"] = [0.14529732, 0.075    ,  0.11614164]
config["model_file"] = "model_case3.zip"

config["species"] = ['OD600','Glucose','Acetate','DOT','Fluo_RFP']
config["normalization_vector"] = [20,10,10,105,200e3]

# init mbrs actions
config["mbrs_actions"] = {exp_id: [] for exp_id in config["exp_ids"]}

# create feeding pulses
pulses_time = np.arange(config["time_batch"] + 5/60, config["final_time"], 10/60)
config["feed_pulses_reference"] = []
for i in range(config["num_experiments"]):
    feed_profile_i = (32.406) * config["mu_reference"][i] * np.exp(config["mu_reference"][i] * (pulses_time - pulses_time[0]))
    # feed_profile_i[pulses_time >= 10] = (36.33) * config["mu_reference"][i] * np.exp(config["mu_reference"][i] * (10 - pulses_time[0]))
    feed_profile_i = np.round(feed_profile_i * 2) / 2
    feed_profile_i[feed_profile_i < 5] = 5

    config["feed_pulses_reference"].append({'time_pulse': pulses_time.tolist(), 'Feed_pulse': feed_profile_i.tolist()})

# save config file
with open('config.json', "w") as outfile:
    json.dump(config, outfile) 