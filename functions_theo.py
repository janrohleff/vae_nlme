"""
Created on 16.07.2025

@author: Jan Rohleff

Initialize functions + EBE function for Case Study 1 (Theopyhlline).
"""

import torch
import numpy as np
from scipy.optimize import minimize

def load_data(path):
    #########################################################
    # Load Data
    #########################################################
    data = torch.load(path) # load data
    lengths = (data.shape[1]*torch.ones(data.shape[0])).int()
    dose = data[:,0,2]       
    weight_pop = data[:,0,3].mean()
    covariates = data[:,0,3:]

    #########################################################
    # Standardize input data
    #########################################################
    data_in = data[:,:,:2].clone()
    data_mean = data[:,:,1].mean()
    data_std = data[:,:,1].std()
    data_in[:,:,0] =  data_in[:,:,0]/data_in[:,:,0].max() 
    data_in[:,:,1] = (data_in[:,:,1] - data_mean)/(data_std)
    covariates_in = data[:,0,3:].clone()
    covariates_in[:,0] = (covariates_in[:,0] - covariates_in[:,0].mean())/covariates_in[:,0].std()

    return data, data_in, lengths, dose, weight_pop, covariates, covariates_in

def initalize_C(nbatch, z_dim, n_cov, covariates, weight_pop):
    C = torch.zeros(nbatch, z_dim, z_dim + z_dim*n_cov)

    for i in range(nbatch):
        C[i,:,:z_dim] = torch.eye(z_dim)
        count = 0
        for k in range(z_dim):
            for j in range(n_cov):
                if j == 1:
                    C[i, k, j + z_dim + count] = covariates[i,j]
                else:    
                    if k < 2:
                        C[i, k, j + z_dim + count] = torch.log(covariates[i,j]/weight_pop)
                    else:
                        C[i, k, j + z_dim+ count] = covariates[i,j] - weight_pop
            count += n_cov
            
    C_regression = torch.zeros(z_dim, nbatch, 1 + n_cov)
    for k in range(z_dim):
        for i in range(nbatch):
            C_regression[k,i,0] = 1
            for j in range(n_cov):
                if j == 1:
                    C_regression[k,i,j+1] = covariates[i,j]
                else:
                    if k < 2:
                        C_regression[k,i,j+1] = torch.log(covariates[i,j]/weight_pop)
                    else:
                        C_regression[k,i,j+1] = covariates[i,j] - weight_pop

    return C, C_regression

def EmpiricalBayesEstimate(data, phi_pop, omega_pop, mu, res, C, h):
    a = res[0].numpy()
    b = res[1].numpy()
    Cz = torch.matmul(C, phi_pop).detach().numpy()
    data = data.detach().numpy()
    omega_pop = omega_pop.detach().numpy()
    mu = mu.detach().numpy()
    C = C.detach().numpy()
    def EBE(phi):
        pred_x = np.zeros(data.shape[0])
        phi_h = h(torch.tensor(phi)).numpy()
        f = lambda t: data[0,2]*phi_h[0]/(phi_h[2]*(phi_h[0] - phi_h[1]))*(np.exp(-phi_h[1]*t) - np.exp(-phi_h[0]*t))
        for t in range(data.shape[0]):
            pred_x[t] = f(data[t,0])
        sigma = a + b * pred_x
        epsilon = (data[:,1]- pred_x )/sigma
        tmp = 0.5*np.sum((phi - Cz)**2/ omega_pop)
        loss = np.sum(0.5*(epsilon)**2 + np.log(sigma)) + tmp
        return loss
    phi = minimize(EBE, mu, method='Nelder-Mead',
                   options={'xatol': 1e-6, 'maxiter': 200}).x
    return torch.tensor(phi)