import numpy as np
import json

# %% Config Panel
t_duration=14.0


species_list=['Xv','Glucose','Acetate','DOT','Fluo_RFP','Volume'] #Model species used in the model

species_IC=[0.18,3,0,100,0,0] #Initial states for the species listed above
glucose_IC=[4,4,4,4,4,4,4,4]*3 #Initial states for the species listed above



pulses_time=np.arange(5+5/60,t_duration,10/60) #Time in hours
time_samples_columns={'col1':np.arange(.33,t_duration,1).tolist()+[t_duration],'col2':np.arange(.66,t_duration,1).tolist()+[t_duration],'col3':np.arange(.99,t_duration,1).tolist()+[t_duration]} #Time in hours
sampling_rate_DOT=2/60 #Time in hours

time_samples_analysis=np.arange(1,t_duration,1).tolist()

Noise_concentration=5 # in %
Noise_time=1 # in %

# mbr_list=np.arange(21340,21364,1) #names of the bioreactors
mbr_list=np.arange(19419,19443,1) #names of the bioreactors

Glucose_feed=[200]*len(mbr_list) # in g/l
Induction_time=[10]*len(mbr_list) #Time in hours
Inductor_conc=[1]*len(mbr_list) # 0 to 1 for now

Params_ref=np.array([1.2578, 0.43041, 0.6439,  2.2048*0+7.0767,  0.4063,  0.1143*4,  0.1848*4,    287.74*0+.4242,    1.586*.7, 1.5874*.7,  0.3322*.75,  0.0371,  0.0818,    9000, .1, 5])#
Params={}
for i in range(int(mbr_list.shape[0]/3)):
    Params_ref_row=(Params_ref*(1+0.66*(np.random.random(len(Params_ref))-.5)/2)).tolist()
    Params[i]=Params_ref_row+(850*(1+0.66*(np.random.random(1)-.5)/2)).tolist()+(90*(1+0.66*(np.random.random(1)-.5)/2)).tolist()
    Params[i+8]=Params_ref_row+(850*(1+0.66*(np.random.random(1)-.5)/2)).tolist()+(90*(1+0.66*(np.random.random(1)-.5)/2)).tolist()
    Params[i+16]=Params_ref_row+(850*(1+0.66*(np.random.random(1)-.5)/2)).tolist()+(90*(1+0.66*(np.random.random(1)-.5)/2)).tolist()
# for i in range(mbr_list.shape[0]):
#     Params_ref_row=(Params_ref*(1+0.66*(np.random.random(len(Params_ref))-.5)/2)).tolist()
#     Params[i]=Params_ref_row+(850*(1+0.66*(np.random.random(1)-.5)/2)).tolist()+(90*(1+0.66*(np.random.random(1)-.5)/2)).tolist()

time_execution=[]#np.arange(0,t_duration+1,1).tolist()# # leave empty for Real Time, otherwise use time in hours

acceleration=60# =1 for real time, or choose between [2, 4, 60, 54000]

# %% Create & Fill config file
EMULATOR_config={}

EMULATOR_config['Params']=Params

Exp_list=[str(il) for il in mbr_list]
EMULATOR_config['Species_list']=species_list
EMULATOR_config['Brxtor_list']=Exp_list

n1=0
for i1 in Exp_list:
    EMULATOR_config[i1]={}
    EMULATOR_config[i1]['IC']={}
    nn=0
    for i2 in EMULATOR_config['Species_list']:
        EMULATOR_config[i1]['IC'][i2]=species_IC[nn]+0
        nn=nn+1
    EMULATOR_config[i1]['IC']['Glucose']=glucose_IC[n1]+0
    n1=n1+1
    
n2=0
for i1 in Exp_list:
    EMULATOR_config[i1]['Glucose_feed']=float(Glucose_feed[n2])
    EMULATOR_config[i1]['Induction_time']=float(Induction_time[n2])
    EMULATOR_config[i1]['Inductor_conc']=float(Inductor_conc[n2])
    
    EMULATOR_config[i1]['Pulse_profile']={'time_pulse':pulses_time.tolist(),'Feed_pulse':(5+np.zeros(len(pulses_time.tolist()))).tolist()}
    EMULATOR_config[i1]['time_sample']={}
    for i2 in EMULATOR_config['Species_list']:
        if n2<8:
            EMULATOR_config[i1]['time_sample'][i2]= time_samples_columns['col1'] 
        elif n2<16:
            EMULATOR_config[i1]['time_sample'][i2]= time_samples_columns['col2'] 
        else:
            EMULATOR_config[i1]['time_sample'][i2]= time_samples_columns['col3'] 
    n2=n2+1
    
    
    # EMULATOR_config[i1]['time_sample']['DOT']=np.arange(0,t_duration+sampling_rate_DOT,sampling_rate_DOT).tolist()
    EMULATOR_config[i1]['time_sample']['DOT']=np.linspace(0.04,t_duration,int(t_duration)*25).tolist()





EMULATOR_config['number_br']=len(mbr_list)

EMULATOR_config['time_execution']=time_execution

EMULATOR_config['Noise_concentration']=Noise_concentration/100
EMULATOR_config['Noise_time']=Noise_time/100

EMULATOR_config['acceleration']=acceleration
EMULATOR_config['experiment_duration']=t_duration

EMULATOR_config['time_samples_analysis']=time_samples_analysis
# %% Save design
with open('EMULATOR_config.json', "w") as outfile:
    json.dump(EMULATOR_config, outfile) 
