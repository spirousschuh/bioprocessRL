# %% Import
import numpy as np
# import json
import time
from scipy.integrate import solve_ivp
# from joblib import Parallel, delayed

from copy import deepcopy

from scipy.optimize import shgo,dual_annealing,minimize,differential_evolution

import matplotlib.pyplot as plt
# %% 
def optimizer_reference(LB,UB,optim_options,tt,initial_states,control_inputs,model_parameters0,feed_profiles,Cov_y,Cov_TH0=[]):
   
    ux_lb=np.array(LB)
    ux_ub=np.array(UB)
    ux_mean=np.linspace(LB[0],UB[0],len(LB))
    
    bounds_ux=list(range(len(ux_lb)))
    for i1 in range(len(ux_lb)):
        bounds_ux[i1]=(ux_lb[i1],ux_ub[i1])
    
    t_test0=time.time()    
    e1=obj_fun(ux_lb,tt,initial_states,control_inputs,model_parameters0,feed_profiles,Cov_y,Cov_TH0)
    e2=obj_fun(ux_ub,tt,initial_states,control_inputs,model_parameters0,feed_profiles,Cov_y,Cov_TH0)
    e3=obj_fun(ux_mean,tt,initial_states,control_inputs,model_parameters0,feed_profiles,Cov_y,Cov_TH0) 
    t_test=(time.time()-t_test0)/3

    n_fun_eval0=round(60*optim_options[0]/(t_test*1.2))
    n_fun_eval1=round(60*optim_options[1]/(t_test*1.2))
    
    print(n_fun_eval0,n_fun_eval1)  
    
    ux_opt0 = dual_annealing(lambda ux: obj_fun(ux,tt,initial_states,control_inputs,model_parameters0,feed_profiles,Cov_y,Cov_TH0), bounds=bounds_ux, maxfun=n_fun_eval0, no_local_search=True)
    print('global opt ',ux_opt0.x)
    ux_opt1 = minimize(lambda ux: obj_fun(ux,tt,initial_states,control_inputs,model_parameters0,feed_profiles,Cov_y,Cov_TH0), ux_opt0.x, bounds=bounds_ux, method='Nelder-Mead', options={'maxfev': n_fun_eval1})

    print('local opt ',ux_opt1.x)
    return ux_opt1        

# %%

def obj_fun(ux,tt,initial_states,control_inputs,model_parameters0,feed_profiles,Cov_y,Cov_TH0=[]):
    feed_profilesx=deepcopy(feed_profiles)
        
    for i2 in range(control_inputs[0][0]):
        t_pulse=np.array(feed_profilesx[i2]['time_pulse'])
        Feed_pulse=(36.33*0+32.406)*ux[i2]*np.exp(ux[i2]*(t_pulse-t_pulse[0]))
        Feed_pulse=np.round(Feed_pulse*2)/2
        Feed_pulse[Feed_pulse<5]=5
        feed_profilesx[i2]['Feed_pulse']=Feed_pulse.tolist()
    
        
    Si,Q,FIM,state_th0,traceFIM,FIM_crit,ei=calculate_FIM(tt,initial_states,control_inputs,model_parameters0,feed_profilesx,Cov_y,Cov_TH0)
    
    DOT_min=[]
    FIM_constrain=[]
    for i2 in range(control_inputs[0][0]):
        dot_min=min(state_th0['sample'][i2][3])
        DOT_min.append(dot_min)
        if dot_min<20:
            # FIM_crit=1e-11
            FIM_constrain.append((1+(20-dot_min)*10)*1e0)
        else:
            FIM_constrain.append(1)
    
    FIM_constr=np.array(FIM_constrain)
    print(ux,'',FIM_crit/3*np.sum(FIM_constr),'constraint ',min(DOT_min))
    return FIM_crit*(1)/3*np.sum(FIM_constr)

# %%
def calculate_FIM(tt,initial_states,control_inputs,model_parameters0,feed_profiles,Cov_y=[],Cov_TH0=[]):
    state_th0=simulate_parallel(tt,initial_states,control_inputs,model_parameters0,feed_profiles)

    Si={}
    for i1 in range(len(model_parameters0)):

        model_parameters=model_parameters0.copy()
        model_parameters[i1]=model_parameters0[i1]*(1+1e-5)

        state_th=simulate_parallel(tt,initial_states,control_inputs,model_parameters,feed_profiles)

        Si[i1]={}

        for i2 in range(control_inputs[0][0]):
            Si[i1][i2]={}
            for i3 in range(5):
                if i2==1:
                    if i1==17:
                        Si[16][i2][i3]=(np.array(state_th['sample'][i2][i3])-np.array(state_th0['sample'][i2][i3]))/(model_parameters[i1]-model_parameters0[i1]+1e-9)*model_parameters0[i1]
                    elif i1==20:
                        Si[17][i2][i3]=(np.array(state_th['sample'][i2][i3])-np.array(state_th0['sample'][i2][i3]))/(model_parameters[i1]-model_parameters0[i1]+1e-9)*model_parameters0[i1] 
                    else:
                        Si[i1][i2][i3]=(np.array(state_th['sample'][i2][i3])-np.array(state_th0['sample'][i2][i3]))/(model_parameters[i1]-model_parameters0[i1]+1e-9)*model_parameters0[i1]                      
                elif i2==2:        
                    if i1==18:
                        Si[16][i2][i3]=(np.array(state_th['sample'][i2][i3])-np.array(state_th0['sample'][i2][i3]))/(model_parameters[i1]-model_parameters0[i1]+1e-9)*model_parameters0[i1]
                    elif i1==21:
                        Si[17][i2][i3]=(np.array(state_th['sample'][i2][i3])-np.array(state_th0['sample'][i2][i3]))/(model_parameters[i1]-model_parameters0[i1]+1e-9)*model_parameters0[i1] 
                        
                    else:
                        Si[i1][i2][i3]=(np.array(state_th['sample'][i2][i3])-np.array(state_th0['sample'][i2][i3]))/(model_parameters[i1]-model_parameters0[i1]+1e-9)*model_parameters0[i1]
                else:
                    if i1==19:
                        Si[17][i2][i3]=(np.array(state_th['sample'][i2][i3])-np.array(state_th0['sample'][i2][i3]))/(model_parameters[i1]-model_parameters0[i1]+1e-9)*model_parameters0[i1]
                    else:
                        Si[i1][i2][i3]=(np.array(state_th['sample'][i2][i3])-np.array(state_th0['sample'][i2][i3]))/(model_parameters[i1]-model_parameters0[i1]+1e-9)*model_parameters0[i1]



        
    Q={}       
    for i1 in range(control_inputs[0][0]): 
        for i2 in range(len(model_parameters)-4):
                for i3 in range(5):
                    q1=Si[i2][i1][i3]
                    q2=q1[:,None]
                    if i3==0:
                        q3=q2
                    else:
                        q3=np.vstack((q3,q2))     
                if i2==0:
                    q4=q3
                else:
                    q4=np.hstack((q4,q3))
                    
        Qi=q4
        FIM_left_i=Qi.transpose()@ np.linalg.inv(Cov_y)
        FIM_i=FIM_left_i@Qi

        Q[i1]=Qi            
        if i1==0:
            FIM=FIM_i
        else:
            FIM=FIM+FIM_i            
    
    if len(Cov_TH0)>0:
        FIM= FIM+1*np.linalg.inv(Cov_TH0)

    
    
    traceFIM=np.trace(FIM)
    ei,ev=np.linalg.eig(FIM)
    
    FIM_crit=min(ei[0:-2])

        
    Cov_diag=np.diag(np.linalg.inv(FIM))
    FIM_crit=np.sum((Cov_diag[[13,14,15]]))

    
    
    return Si,Q,FIM,state_th0,traceFIM,FIM_crit,ei

# %%

def simulate_parallel(ts,initial_states,control_inputs,model_parameters,feed_profiles):  
    state=deepcopy(initial_states)
    brxtor_list=np.arange(control_inputs[0][0]).tolist()
    
    ty={}
        
    for i1 in brxtor_list: 
        ty[i1]=simulate_interval(i1, ts,state,control_inputs,model_parameters,feed_profiles)
        
    for i1 in brxtor_list:
        state['state'][i1]=ty[i1][-1,1:]
        
        
        for i2 in [0,1,2,4,5]:
            ts_sample_all=feed_profiles[i1]['time_sample'] 
            ts_sample=ts_sample_all[(ts_sample_all>ts[0]) & (ts_sample_all<=ts[1])]
            
            sample_interp=np.interp(ts_sample,ty[i1][:,0],ty[i1][:,i2+1])

            try:
                state['sample'][i1][i2]=state['sample'][i1][i2]+sample_interp.tolist() 
                
            except:
                state['sample'][i1][i2]=sample_interp.tolist()
                
                
        ts_sensor_all=feed_profiles[i1]['time_sensor'] 
        ts_sensor=ts_sensor_all[(ts_sensor_all>ts[0]) & (ts_sensor_all<=ts[1])]
        sensor_interp=np.interp(ts_sensor,ty[i1][:,0],ty[i1][:,4])

        try:
            state['sample'][i1][3]=state['sample'][i1][3]+sensor_interp.tolist() 

        except:
            state['sample'][i1][3]=sensor_interp.tolist()
    return state
# %%
def simulate_interval(index_mbr,ts,state,control_inputs,model_parameters,feed_profiles):

            
            u=[control_inputs[index_mbr][1]]+[index_mbr]+[control_inputs[index_mbr][0]]+[control_inputs[index_mbr][2]]+[1]
            X = np.array(state['state'][index_mbr])
            D = feed_profiles[index_mbr]
            t, y = function_simulation(ts, X, u, model_parameters, D)

            return np.hstack((t[:,None],y))

            raise
# %%
def function_simulation(ts0,Xo0,u0,THs,D0={}):
    TH1=THs[0:16]


    TH1=np.append(TH1,THs[16+int(u0[1])])
    TH1=np.append(TH1,THs[16+int(u0[1])+int(u0[2])]) 
    
    
    ts_start=ts0[0]
    ts_end=ts0[-1]
    
    time_pulse_all=np.array(D0['time_pulse'])
    Feed_pulse_all=np.array(D0['Feed_pulse'])
    
    t_u=time_pulse_all[(time_pulse_all>=ts_start) & (time_pulse_all<=ts_end)]
    control_inputs_base_design=Feed_pulse_all[(time_pulse_all>=ts_start) & (time_pulse_all<=ts_end)]
 
    control_inputs=control_inputs_base_design

    if len(t_u)==0:
        t_u=np.array([ts_start,ts_end])
        control_inputs=np.array([0,0])
    else:
        if ts_start<t_u[0]:
            t_u=np.append(ts_start,t_u)
            control_inputs=np.append(0,control_inputs)
        if ts_end>t_u[-1]:
            t_u=np.append(t_u,ts_end)
            control_inputs=np.append(control_inputs,0)

    Xo1=Xo0.copy()
    
    tt=np.array(ts_start)
    yy=np.array([Xo1])
    yy=yy.transpose()
    
    ni=0
    
    for i in control_inputs[:-1]:
        ts1=np.linspace(t_u[ni],t_u[ni+1],25+1)
        Xo1[1]=Xo1[1]+control_inputs[ni]*1e-6*u0[0]/0.01
        t,y=intM(ts1,Xo1,u0,TH1)
        Xo1=y[:,-1].copy()

        
        tt=np.append(tt,t[1:])
        yy=np.append(yy,y[:,1:],axis=1)
        ni=ni+1


    return tt,yy.transpose()

# %%    
def odeFB(t,Xo,THo,u):

    X=Xo.copy()
    TH=THo.copy()
    
    
    f_ode=rates(t,X,TH,u)
    
    dXv=f_ode[0]
    dS=f_ode[1]
    dA=f_ode[2]
    feed_profilesOT=f_ode[3]
    dP=f_ode[4]
    dmu=f_ode[5]
    
    dX=np.array([dXv,dS,dA,feed_profilesOT,dP,dmu])
    return dX
# %%  
def rates(t,X,TH,u):
 
    X = np.maximum(X, 1e-9)
    
    Xv=X[0]
    S=X[1]
    A=X[2]
    DOT=X[3] 
    P=X[4]
    mu_m=X[5]

    DOT = np.minimum(DOT, 100)
            
    qs_max=TH[0]
    fracc_q_ox_max=TH[1]
    qa_max=TH[2]
    
    
    Ys_ox=TH[4]
    Ya_p=TH[5]
    Ya_c=TH[6]

    Yo_ox=TH[8]
    Yo_a=TH[9]
    Yxs_of=TH[10]
    
    Ks=TH[11]
    n_ox=4
    
    Ka=TH[12]
    Ksi=TH[3]
    Kai=TH[7]
    Ko=0.1057
    
    kla=TH[16]
    k_sensor=TH[17]
    
    ky_1=TH[13] 
    ky_2=TH[14]
    ky_3=TH[15]
    
    DO_star=100
    H=13000#
    
    qs=qs_max*S/(S+Ks)*Ksi/(Ksi+A)
    q_ox_max=fracc_q_ox_max*qs_max
    
    q_ox_ss=qs*(1/((qs/q_ox_max)**n_ox+1))**(1/n_ox)
    qac_ss=qa_max*A/(A+Ka)*Kai/(Kai+S)
    b_ss=Ko+(q_ox_ss*Yo_ox+qac_ss*Yo_a)*Xv*H/kla-DO_star
    c_ss=-DO_star*Ko
    DOT_ss=(-b_ss+(b_ss*b_ss-4*c_ss)**.5)/2
    
    q_ox=qs*(1/((qs/q_ox_max)**n_ox+1))**(1/n_ox)*DOT_ss/(DOT_ss+Ko)
    q_of=qs-q_ox
    
    qac=qa_max*A/(A+Ka)*Kai/(Kai+S)*DOT_ss/(DOT_ss+Ko)
    
    qap=q_of*Ya_p
    
    mu=q_ox*Ys_ox+qac*Ya_c+Yxs_of*q_of

    if t>=u[3]:
        s_prod=u[4]
    else:
        s_prod=0
        
    f_qp=ky_1*mu_m/(mu_m+ky_2+(ky_3*mu_m)**2)

    q_prod=s_prod*f_qp

    f_ode=[(mu)*Xv,-(qs)*Xv,qap*Xv-qac*Xv,k_sensor*(DOT_ss-DOT),q_prod*Xv,(mu-mu_m)/(.167)]
    return f_ode
# %%     
def intM(ts0,Xo0,u0,TH0):    

    tspan=np.array([ts0[0],ts0[-1]])
    Xo1=Xo0.tolist().copy()



    sol=solve_ivp(lambda t,y: odeFB(t,y,TH0,u0) ,tspan,Xo1,method="BDF", rtol=1e-5, atol=1e-5,t_eval=ts0)
    y_interm=sol.y
    y_interm[y_interm<0]=0
    y_return=y_interm.copy()

    return sol.t,y_return