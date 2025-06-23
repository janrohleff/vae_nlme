"""
Created on 16.07.2025

@author: Jan Rohleff

Decoder functions for Case Study 1 (Theophylline) and Case Study 2 (Neonates).
"""
#########################################################
# Import
#########################################################
import torch
import torchode as to

#########################################################
# Decoder
#########################################################   
def Decoder_theophylline(z_normal, time, h, dose):
    nbatch = dose.shape[0]
    z = h(z_normal)
    sol = lambda t: dose*z[:,0]/(z[:,2]*(z[:,0] - z[:,1]))*(torch.exp(-z[:,1]*t) - torch.exp(-z[:,0]*t))   
    pred_x = torch.zeros(nbatch, time.shape[1], 1)
    for t in range(time.shape[1]):
        pred_x[:,t,0] = sol(time[:,t])

    return pred_x

def Decoder_neonates(z_normal, time, h, ODE_settings, args = None):
    nbatch = time.shape[0]
    z = h(z_normal)
    state0 = z[:,0].view(nbatch,1)
    problem = to.InitialValueProblem(y0=state0, t_eval=ODE_settings['t_eval'] )
    sol = ODE_settings['solver'].solve(problem, args = z ) 
    pred_x = sol.ys[:,:,0].view(nbatch, time.shape[1], 1)

    return pred_x
    