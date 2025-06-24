import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
import pandas as pd
import random as rnd

params=pd.read_csv("Lotka_Volterra_Params.csv")

added_params=pd.DataFrame({'Prey Birth Rate' : [0.1,0.2,0.3,0.4],'Predation Rate':[0.3,0.6,0.6,0.4],'Feeding Rate' : [0.9,0.1,0.4,0.6],'Pred. Death Rate' : [0.1,0.2,0.5,0.6]})
params=pd.concat([params,added_params])
params['Carrying Capacity']=[rnd.randint(5,15) for _ in range(len(params['Prey Birth Rate']))]
def Lotka_Volterra_ODE(t,state,params):
    X,Y=state
    dX = (params['Prey Birth Rate']*X*(1-(X/params['Carrying Capacity']))) - (X*Y*params['Predation Rate']) 
    dY = (params['Feeding Rate']*X*Y) - params['Pred. Death Rate']*Y 
    return [dX, dY]

params_dict=params.iloc[2].to_dict()
init_conds=[5,5]
time_span=[0,100]
time_eval=np.linspace(time_span[0],time_span[1],1000)

solution=solve_ivp(Lotka_Volterra_ODE,t_span=time_span,t_eval=time_eval,args=(params_dict,),y0=init_conds)

fig,ax=plt.subplots(1,2)
ax[0].plot(solution.t,solution.y[0],label='Prey')
ax[0].plot(solution.t,solution.y[1],label='Predator')
ax[1].plot(solution.y[0],solution.y[1])
plt.show()

