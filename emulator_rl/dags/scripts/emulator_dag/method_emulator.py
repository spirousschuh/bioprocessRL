# %% Import
import time
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from copy import deepcopy

from scipy.integrate import solve_ivp
from scipy.optimize import shgo,dual_annealing,minimize
from scipy.optimize import approx_fprime

import json
from function_simulation import function_simulation

# %% Simulator
def simulate(time_initial,final_time,EMULATOR_state,EMULATOR_design,EMULATOR_config):
    NEW_EMULATOR_state=deepcopy(EMULATOR_state)
    
    nn=0   
    for i1 in EMULATOR_config['Brxtor_list']:
        ts0=np.array([time_initial,final_time])
        Xo0=np.array([])
        for i2 in EMULATOR_config['Species_list']:
            Xo0=np.append(Xo0,EMULATOR_state[i1]['Current'][i2])
        u0=np.array([EMULATOR_design[i1]['Glucose_feed'],nn,EMULATOR_config['number_br'],EMULATOR_design[i1]['Induction_time'],EMULATOR_design[i1]['Inductor_conc']])
        # THs=np.array(EMULATOR_config['Params'])
        THs=EMULATOR_config['Params']
        # print(THs)
        D0=EMULATOR_design[i1]['Pulses'].copy()
        
        t,y=function_simulation(ts0,Xo0,u0,THs,D0)
        
        nn2=0
        for i2 in EMULATOR_config['Species_list']:
            NEW_EMULATOR_state[i1]['All'][i2]['time']=np.append(np.array(NEW_EMULATOR_state[i1]['All'][i2]['time']),t[1:].flatten()).tolist()
            NEW_EMULATOR_state[i1]['All'][i2]['Value']=np.append(np.array(NEW_EMULATOR_state[i1]['All'][i2]['Value']),y[1:,nn2].flatten()).tolist()
            
            NEW_EMULATOR_state[i1]['Current'][i2]=float(y[-1,nn2])

            nn2=nn2+1
 
        nn=nn+1
    return NEW_EMULATOR_state
# %% Sampler
def sample(time_initial,final_time,EMULATOR_state,EMULATOR_design,EMULATOR_config):
    NEW_EMULATOR_state=deepcopy(EMULATOR_state)

    for i1 in EMULATOR_config['Brxtor_list']:
        for i2 in EMULATOR_config['Species_list']:
            if i2=='DOT':
                ts_sample_all=np.array(EMULATOR_design[i1]['time_sample'][i2])
            else:
                ts_sample_all=np.array(EMULATOR_design[i1]['time_sample'][i2])*(1+np.random.normal(0,1,size=len(np.array(EMULATOR_design[i1]['time_sample'][i2])))*EMULATOR_config['Noise_time']*0)
           
            ts_sample=ts_sample_all[(ts_sample_all>time_initial) & (ts_sample_all<=final_time)]

            tX_state=EMULATOR_state[i1]['All'][i2]['time']
            X_state=EMULATOR_state[i1]['All'][i2]['Value']
            
            if i2=='DOT':
                X_interp=np.interp(ts_sample,tX_state,X_state)
                X_interp=X_interp*(1+np.random.normal(0,1,size=len(X_interp))*1/5*EMULATOR_config['Noise_concentration'])
            else:
                X_interp=np.interp(ts_sample,tX_state,X_state)
                X_interp=X_interp*(1+np.random.normal(0,1,size=len(X_interp))*EMULATOR_config['Noise_concentration'])

            NEW_EMULATOR_state[i1]['Sample'][i2]['time']=np.append(np.array(NEW_EMULATOR_state[i1]['Sample'][i2]['time']),ts_sample).tolist()
            NEW_EMULATOR_state[i1]['Sample'][i2]['Value']=np.append(np.array(NEW_EMULATOR_state[i1]['Sample'][i2]['Value']),X_interp).tolist()    
    
    return NEW_EMULATOR_state
# %% Write
def write(filename,time_initial,final_time,EMULATOR_state,EMULATOR_design,EMULATOR_config):
    n_check=0
    while n_check==0:
        try:    
            with open(filename) as json_file:   
                File_dict = json.load(json_file)
                n_check=1
        except:
            print('fail write (load) emulator')
    time_samples_analysis=EMULATOR_config['time_samples_analysis']        
    for i1 in EMULATOR_config['Brxtor_list']:
      
        for i2 in EMULATOR_config['Species_list']:
            if i2=='Xv':
                # tsf=(np.array(list(File_dict[i1]['measurements_aggregated']['OD600']['measurement_time'].values()))).tolist()
                # Xsf=list(File_dict[i1]['measurements_aggregated']['OD600']['OD600'].values())
                
                ts_new=np.array(EMULATOR_state[i1]['Sample'][i2]['time'])
                Xsf_new=np.array(EMULATOR_state[i1]['Sample'][i2]['Value'])*2.7027
                
                time_samples_iter=np.array(time_samples_analysis)
                time_samples_iter=time_samples_iter[time_samples_iter<=(final_time-1)]
                if len(time_samples_iter)>0:
                    time_analysis=time_samples_iter[-1]
                    Xsf_new=Xsf_new[ts_new<=time_analysis]
                    ts_new=ts_new[ts_new<=time_analysis]
                    tsf=(ts_new*3600).tolist()
                    Xsf=Xsf_new.tolist()
                    
                else:
                    tsf=[]
                
    
            
                File_dict[i1]['measurements_aggregated']['OD600']['measurement_time']={}
                File_dict[i1]['measurements_aggregated']['OD600']['OD600']={}
                for i3 in range(len(tsf)):
                    File_dict[i1]['measurements_aggregated']['OD600']['measurement_time'][str(i3)]=tsf[i3]
                    
                    File_dict[i1]['measurements_aggregated']['OD600']['OD600'][str(i3)]=Xsf[i3]
                    
            elif i2=='DOT':
                tsf=(np.array(list(File_dict[i1]['measurements_aggregated'][i2]['measurement_time'].values()))).tolist()
                Xsf=list(File_dict[i1]['measurements_aggregated'][i2][i2].values())

                ts_new=np.array(EMULATOR_state[i1]['Sample'][i2]['time'])
                Xsf_new=np.array(EMULATOR_state[i1]['Sample'][i2]['Value'])
                
                Xsf_new=Xsf_new[(ts_new>time_initial) & (ts_new<=final_time)]
                ts_new=ts_new[(ts_new>time_initial) & (ts_new<=final_time)]
                
                tsf=tsf+(ts_new*3600).tolist()
                Xsf=Xsf+Xsf_new.tolist()
                
                File_dict[i1]['measurements_aggregated'][i2]['measurement_time']={}
                File_dict[i1]['measurements_aggregated'][i2][i2]={}
                for i3 in range(0,len(tsf)):
                    File_dict[i1]['measurements_aggregated'][i2]['measurement_time'][str(i3)]=tsf[i3]
                    File_dict[i1]['measurements_aggregated'][i2][i2][str(i3)]=Xsf[i3]        
            else:
                # tsf=(np.array(list(File_dict[i1]['measurements_aggregated'][i2]['measurement_time'].values()))).tolist()
                # Xsf=list(File_dict[i1]['measurements_aggregated'][i2][i2].values())

                ts_new=np.array(EMULATOR_state[i1]['Sample'][i2]['time'])
                Xsf_new=np.array(EMULATOR_state[i1]['Sample'][i2]['Value'])
                
                time_samples_iter=np.array(time_samples_analysis)
                time_samples_iter=time_samples_iter[time_samples_iter<=(final_time-1)]
                if len(time_samples_iter)>0:
                    time_analysis=time_samples_iter[-1]
                    Xsf_new=Xsf_new[ts_new<=time_analysis]
                    ts_new=ts_new[ts_new<=time_analysis]
                    tsf=(ts_new*3600).tolist()
                    Xsf=Xsf_new.tolist()
                    
                else:
                    tsf=[]
                # Xsf_new=Xsf_new[(ts_new>time_initial) & (ts_new<=final_time)]
                # ts_new=ts_new[(ts_new>time_initial) & (ts_new<=final_time)]
                
                # tsf=tsf+(ts_new*3600).tolist()
                # Xsf=Xsf+Xsf_new.tolist()
                
                File_dict[i1]['measurements_aggregated'][i2]['measurement_time']={}
                File_dict[i1]['measurements_aggregated'][i2][i2]={}
                for i3 in range(0,len(tsf)):
                    File_dict[i1]['measurements_aggregated'][i2]['measurement_time'][str(i3)]=tsf[i3]
                    File_dict[i1]['measurements_aggregated'][i2][i2][str(i3)]=Xsf[i3]
                            
        # Feed
        tsf=(np.array(list(File_dict[i1]['measurements_aggregated']['Cumulated_feed_volume_glucose']['measurement_time'].values()))).tolist()
        Xsf=list(File_dict[i1]['measurements_aggregated']['Cumulated_feed_volume_glucose']['Cumulated_feed_volume_glucose'].values())

        ts_new=np.array(EMULATOR_design[i1]['Pulses']['time_pulse'])#np.array(EMULATOR_state[i1]['Sample'][i2]['time'])
        Xsf_pulses=np.array(EMULATOR_design[i1]['Pulses']['Feed_pulse'])#np.array(EMULATOR_state[i1]['Sample'][i2]['Value'])
        
        Xsf_pulses_new=Xsf_pulses[(ts_new>time_initial) & (ts_new<=final_time)]
        ts_new=ts_new[(ts_new>time_initial) & (ts_new<=final_time)].copy()

        tsf=tsf+(ts_new*3600).tolist()
        if len(Xsf)==1:
            last_pulse=Xsf[0]+0
        elif len(Xsf)>1:
            last_pulse=Xsf[-1]+0
        else:
            last_pulse=0
        Xsf_all=Xsf+(np.cumsum(Xsf_pulses_new)+last_pulse).tolist()
            
        File_dict[i1]['measurements_aggregated']['Cumulated_feed_volume_glucose']['measurement_time']={}
        File_dict[i1]['measurements_aggregated']['Cumulated_feed_volume_glucose']['Cumulated_feed_volume_glucose']={}
        for i3 in range(0,len(tsf)):
            File_dict[i1]['measurements_aggregated']['Cumulated_feed_volume_glucose']['measurement_time'][str(i3)]=tsf[i3]+0
            File_dict[i1]['measurements_aggregated']['Cumulated_feed_volume_glucose']['Cumulated_feed_volume_glucose'][str(i3)]=Xsf_all[i3]+0
        
    n_check=0
    while n_check==0:
        try:    
            with open(filename, "w") as outfile:
                json.dump(File_dict, outfile) 
            n_check=1
        except:
            print('fail write emulator')  
            time.sleep(1.05)
    # n_check=0
    # while n_check==0:
    #     try:    
    #         with open('db_emulator_streamlit.json', "w") as outfile:
    #             json.dump(File_dict, outfile) 
    #         n_check=1
    #     except:
    #         print('fail write emulator') 
    #         time.sleep(1.05)
    # n_check=0
    # while n_check==0:
    #     try:    
    #         with open('db_emulator_dot.json', "w") as outfile:
    #             json.dump(File_dict, outfile) 
    #         n_check=1
    #     except:
    #         print('fail write emulator') 
    #         time.sleep(1.05)
            
    return File_dict
# %% Read
def read(filename,EMULATOR_design,EMULATOR_config):
    NEW_EMULATOR_design=deepcopy(EMULATOR_design)
    n_check=0
    while n_check==0:
        try:    
            with open(filename) as json_file:   
                File_dict = json.load(json_file)
            n_check=1
        except:
            print('fail read emulator')
            time.sleep(1.05)
       
    for i1 in EMULATOR_config['Brxtor_list']:
        f0=np.array(list(File_dict[i1]['setpoints']['Feed_glc_cum_setpoints'].values()))

        tf0=np.array(list(File_dict[i1]['setpoints']['cultivation_age'].values()))

        mask_f0=f0!=None
        f_acc=f0[mask_f0]
        f=np.hstack([f_acc[0],np.diff(f_acc)])
        
        tf=tf0[mask_f0]/3600

        NEW_EMULATOR_design[i1]['Pulses']['time_pulse']=tf.tolist()
        NEW_EMULATOR_design[i1]['Pulses']['Feed_pulse']=f.tolist()
    return NEW_EMULATOR_design