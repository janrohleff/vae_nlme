"""
Created on 16.07.2025

@author: Jan Rohleff

Initialize functions for Case Study 2 (Neonates).
"""
import torch

def load_data(path_data, path_lengths):
    #########################################################
    # Load Data
    #########################################################
    data = torch.load(path_data) # load data
    lengths = torch.load(path_lengths) # load lengths      

    ###########################################################
    # Covariates
    ###########################################################
    GA = data[:,0,4].clone()
    Mage = data[:,0,5].clone()
    tmp = torch.zeros(data.shape[0], data.shape[1])
    tmp = torch.zeros(data.shape[0], data.shape[1])
    mask = torch.arange(data.shape[1]).unsqueeze(0) < lengths.unsqueeze(1)
    tmp[mask] = 1

    data[:,:,4] = tmp * torch.log(GA/GA.mean()).unsqueeze(1)
    data[:,:,5] = tmp * torch.log(Mage/Mage.mean()).unsqueeze(1)


    #########################################################
    # Standardize input data
    #########################################################
    data_in =  data[:,:,:2].clone()
    data_mean = data[:,:,1].mean()
    data_std = data[:,:,1].std()
    data_in[:,:,0] = data_in[:,:,0]/data_in[:,:,0].max()
    data_in[:,:,1] = (data_in[:,:,1] - data_mean)/(data_std)
    covariates = data[:,0,2:]

    return data, data_in, lengths, covariates

def initalize_C(nbatch, z_dim, n_cov, covariates):
    C = torch.zeros(nbatch, z_dim, z_dim + z_dim*n_cov)
    for i in range(nbatch):
        C[i,:,:z_dim] = torch.eye(z_dim)
        count = 0
        for k in range(z_dim):
            for j in range(n_cov):
                C[i, k, j + z_dim+ count] = covariates[i,j]
            count += n_cov
          
    C_regression = torch.zeros(z_dim, nbatch, 1 + n_cov)
   
    for k in range(z_dim):
        for i in range(nbatch):
            C_regression[k,i,0] = 1
            for j in range(n_cov):
                C_regression[k,i,j+1] = covariates[i,j]
    return C, C_regression
