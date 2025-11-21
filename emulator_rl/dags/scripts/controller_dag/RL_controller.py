import json
import numpy as np
from stable_baselines3 import PPO
import sys


# function to parse data from ilab database to neural network input
# %%
def create_input_from_db(row_mbrs, db_output, config):

    current_time = config["time_batch"] + config["iter"]

    # create vector with (3) MBR action per time step
    actions = 1+np.transpose(np.array([config["mbrs_actions"][str(mbr)] for mbr in row_mbrs]))

    e_vector = np.array([current_time - 1*0])
    d_vector = np.concatenate((actions.flatten(), np.tile([0], (config["final_time"] - current_time) * config["num_experiments"])))
    y_vector = np.array([])

    # add all the measurements to the vector (including batch phase)
    for it in range(current_time - 1):
        mbr_measurements = []
        for mbr in row_mbrs:                 
            for measurement in config["species"]: #["OD600", "DOT"]:
                if measurement == "DOT":
                    # get min value in the corresponding hour iteration HERE!!!! ERROR IN LOOPING!!!!
                    dot_time = np.array(list(db_output[str(mbr)]["measurements_aggregated"][measurement]["measurement_time"].values()))
                    dot_values = np.array(list(db_output[str(mbr)]["measurements_aggregated"][measurement][measurement].values()))
                    
                    mbr_measurements.append(min(dot_values[(dot_time >= it * 3600) & (dot_time <= (it + 1) * 3600)]))
                else:
                    if measurement == "OD600":
                        mbr_measurements.append(db_output[str(mbr)]["measurements_aggregated"][measurement][measurement][str(it)] / 2.7027)
                    else:
                        mbr_measurements.append(db_output[str(mbr)]["measurements_aggregated"][measurement][measurement][str(it)])

        y_vector = np.concatenate((y_vector, mbr_measurements))

    # afeed_profiles zeros to y_vector to complete the final_time (current time + 1: because delay in measurements)
    y_vector = np.concatenate((y_vector, np.tile([0]*len(config['species']), (config["final_time"] - current_time + 1) * config["num_experiments"])))

    # normalize
    vector = np.concatenate([e_vector, d_vector, y_vector.flatten()])

    normalize_vector = np.concatenate((
        np.array([config["final_time"]]),
        np.tile([21], (config["final_time"] - config["time_batch"]) * config["num_experiments"]), 
        # np.tile([20,  105], config["final_time"] * config["num_experiments"])
        np.tile(config['normalization_vector'], config["final_time"] * config["num_experiments"])

    ))


    return vector / normalize_vector

# -------------------- CONTROLLER SCRIPT --------------------------
   
db_output = json.load(open(sys.argv[1], "r"))
config = json.load(open(sys.argv[2], "r"))
feed = json.load(open(sys.argv[3], "r"))

# %%
# load trained model
model = PPO.load(config["model_file"], print_system_info=True, env=None)

# get config variables
mbr_groups = config["mbr_groups"]
action_values = config["action_values"]

current_time = (config["time_batch"] + config["iter"]) * 3600

# iterate over MBRs rows
for row_mbrs in mbr_groups:

    # preprocess data from JSON to network input
    vector_input = create_input_from_db(row_mbrs, db_output, config)
    # load model and predict actions per row    
    actions, _ = model.predict(vector_input,deterministic=True)  
     
    if actions.size == 1:
        actions = [actions]

    print(row_mbrs, actions)  
    
    # calculate new feeding pulses from reference and update feed profile cummulative
    for idx, mbr in enumerate(row_mbrs):

        mbr_time_pulse = np.array(feed[str(mbr)]["measurement_time"])
        mbr_feed_pulse = np.diff(np.array(feed[str(mbr)]["setpoint_value"]), prepend=0)

        # add action to corresponding hour in feed pulse       
        current_hour = (mbr_time_pulse >= current_time) & (mbr_time_pulse <= current_time + 3600)
        mbr_feed_pulse[current_hour] += action_values[actions[idx]]
        
        # check min pulse volume constraint
        # if (current_time>=config["time_batch"]*3600) & (current_time<3600+config["time_batch"]*3600):
        #     mbr_feed_pulse_aux=mbr_feed_pulse[current_hour]
        #     mbr_feed_pulse_aux[mbr_feed_pulse_aux < 5]  = 0
        #     mbr_feed_pulse[current_hour]=mbr_feed_pulse_aux
        # else:
        mbr_feed_pulse_aux=mbr_feed_pulse[current_hour]
        mbr_feed_pulse_aux[mbr_feed_pulse_aux < 5]  = 5
        mbr_feed_pulse[current_hour]=mbr_feed_pulse_aux

        # update feed profile and convert to cummulative
        feed[str(mbr)]['setpoint_value'] = np.cumsum(mbr_feed_pulse).tolist()

        # update mbr actions
        config["mbrs_actions"][str(mbr)].append(int(actions[idx]))

# update iteration
config["iter"] = config["iter"] + 1

# save updated profile to JSON            
with open(sys.argv[3], "w") as json_file:
    json.dump(feed, json_file)   

# save updated profile to JSON            
with open(sys.argv[2], "w") as json_file:
    json.dump(config, json_file)  

