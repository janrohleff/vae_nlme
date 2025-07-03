"""
Created on 16.07.2025

@author: Jan Rohleff

Parameter estimation + Covariate selection of a nonlinear mixed effects model using a variational autoencoder (VAE) for theophylline data (Case Study ).
"""
#########################################################
# Import
#########################################################
import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
from functions import *
from functions_theo import *
from VAE.decoder import *
from VAE.encoder import *
from ParaUpdate.pop_parameter import *
from visualization import *
import torch

# Pick a manual seed for randomization
torch.manual_seed(1)

#########################################################
# Load Data
#########################################################
path = 'VAE_nlme/Data/'
data, data_in, lengths, dose, weight_pop, covariates, covariates_in = load_data_multiple(path + 'theophylline_multiple_data.pt')
#########################################################
# Dimensions
#########################################################
nbatch = data.shape[0]       # Number of individuals
x_dim  = 2                   # Dimensionality of the observations
z_dim  = 3                   # Number of individual parameters (ka, ke, V)
n_cov  = covariates.shape[1] # Number of covariates
#########################################################
# Prior distribution
#########################################################
h         = lambda x: x.exp()
h_inverse = lambda x: x.log()
#########################################################
# Initialization LSTM Encoder
#########################################################
h_dim   = 25                                     # Hidden dimension of the LSTM
sigma0  = torch.tensor([1e-2, 5e-3, 1e-1]).log() # Initial standard deviation of the posterior distribution      
mu0     = torch.tensor([1, 0.5, 15])             # Initial mean of the posterior distribution
Encoder = LSTM_Encoder(x_dim, h_dim, z_dim, nbatch, n_cov, mu0, sigma0, h_inverse)
#Encoder = torch.compile(Encoder)
#########################################################
# Initialization Decoder
#########################################################
Decoder = lambda z_normal, time, h: Decoder_theophylline_multiple(z_normal, time, h, dose)
#########################################################
# Full Covariate Model
#########################################################
C, C_regression   = initalize_C(nbatch, z_dim, n_cov, covariates, weight_pop)
names_co          = ['beta_ka_weight', 'beta_ka_sex', 'beta_ke_weight', 'beta_ke_sex', 'beta_V_weight', 'beta_V_sex']
penalized_indices = np.arange(1,n_cov+ 1)   
M                 = C.shape[2] - z_dim   
#########################################################
# Iterations Setup
#########################################################
iters_burn_in = 100                              # Number of burn-in iterations
kl_iter       = 50                               # Number of iterations for KL annealing
gamma_iter    = 250                              # Number of iterations for population smoothing
iters         = 300                              # Overall number of iterations
L_iter        = 5                                # Number of gradient steps per iteration
burn_in_iter  = L_iter * iters_burn_in           # Overall burn-in gradient steps
alpha_KL      = torch.linspace(0.01, 1, kl_iter) # KL annealing factor
smoothing     = False
#########################################################
# Initialize population parameters updates
#########################################################
pop =  pop_parameter(z_dim, nbatch, gamma_iter, data, C, C_regression, C[:,:z_dim,:z_dim], penalized_indices, n_cov, kl_iter, lengths, 2)
#########################################################
# Burn in
#########################################################
Encoder, optimizer, pred_x, mu, L, a, b, z_pop_iter_bi, omega_pop_iter_bi, a_iter_bi, elbo_iter_bi = initalizeEncoder(iters_burn_in, L_iter, Encoder, Decoder, data, data_in, z_dim, covariates_in, lengths, h, pop)
pred_x_mean = pred_x.detach()
#########################################################
# Setup Optimizer
#########################################################
optimizer.param_groups[0]['lr'] = 2e-3 # Learning rate
#########################################################
# Initialize tensors to store results
#########################################################
z_pop_iter     = torch.zeros(iters, z_dim + M)
omega_pop_iter = torch.zeros(iters, z_dim)
a_iter         = torch.zeros(iters)
Elbo_iter      = torch.zeros(iters)
#########################################################
# Training
#########################################################
print('')
print('#############################################')
print('Training VAE')
for iter in range(1,iters + 1):
    #########################################################
    # Update population parameters
    #########################################################
    if iter > 1:
        pred_x_mean = data_matrix[L_iter-2:L_iter].mean(dim = 0)
    if iter > gamma_iter:
        z_pop, omega_pop, a, _ = pop.update_pop(mu.detach(), L.detach(), pred_x_mean, iter, covariate_selection=True, smoothing=True, update_pop=True)
    else:
        z_pop, omega_pop, a, mu_smooth = pop.update_pop(mu.detach(), L.detach(), pred_x_mean, iter, covariate_selection=True, update_pop=True)
    
    data_matrix = torch.zeros(L_iter, nbatch, data.shape[1], 1) # Initialize tensor to store prediction and ELBO over L_iter gradient steps
    ELBO        = torch.zeros(L_iter)
    for l in range(L_iter):
        #########################################################
        # Encoder
        #########################################################
        z_normal, mu, L, log_sigma, eps = Encoder(data_in, covariates_in, lengths)
        #########################################################
        # Decoder
        #########################################################
        pred_x = Decoder(z_normal, data[:,:,0], h)
        data_matrix[l] = pred_x.clone().detach()
        #########################################################
        # Loss computation
        #########################################################
        z_pop_batch = torch.zeros(nbatch, z_dim)
        for i in range(nbatch):
            z_pop_batch[i] = torch.matmul(C[i], z_pop)
        p_x_z = p_x_z_compute(data[:,:,1].view(data.shape[0],data.shape[1],1), pred_x, [a,b], lengths) # Compute likelihood p(x|z)
        p_z   = p_z_compute(z_normal, z_pop_batch, omega_pop)                                          # Compute prior p(z)
        q_z   = q_z_x_compute(eps, torch.diagonal(L, dim1=1, dim2=2))                                  # Compute posterior q(z|x)
        DKL   = p_z - q_z                                                                              # Compute KL divergence DKL = p(z) - q(z|x)
        if iter < kl_iter:
            elbo = p_x_z + alpha_KL[iter] *DKL 
        else:
            elbo = p_x_z + DKL 
        # Store ELBO
        ELBO[l] = p_x_z + DKL 
        
        elbo.backward()
        optimizer.step()
        optimizer.zero_grad()
        
    #########################################################
    # Print
    #########################################################
    if iter < gamma_iter:
        print(f"Iteration {iter}/{gamma_iter}", end="\r")
    else:
        print(f"Iteration {iter - gamma_iter}/{iters- gamma_iter} (smoothing)", end="\r")

    #########################################################
    # Save Iteration results
    #########################################################
    z_pop_iter[iter-1]     = torch.hstack((h(z_pop[:z_dim]), z_pop[z_dim:]))
    omega_pop_iter[iter-1] = omega_pop.sqrt()
    a_iter[iter-1]         = a
    Elbo_iter[iter-1]      = ELBO.mean()   
#########################################################
# Compute log likelihood
#########################################################
print('')
print('#############################################')
print('Compute log likelihood and empirical Bayes estimate')
LL_lin_mu = LogLikelihood_linearization(z_pop, omega_pop, [a,b], data, mu_smooth, C, h, data[:,:,0], lengths, Decoder)
phi_opt   = torch.zeros(nbatch, z_dim)
for i in range(nbatch):
    phi_opt[i,:] = EmpiricalBayesEstimate(data[i,:,:], z_pop, omega_pop, mu[i], [a,b], C[i], h)
    
LL_lin   = LogLikelihood_linearization(z_pop, omega_pop, [a,b], data, phi_opt, C, h, data[:,:,0], lengths, Decoder)
LL_is, _ = LogLikelihood_sample(1000, z_pop, omega_pop, [a,b], data, mu, L, C, h, data[:,:,0], lengths, Decoder)
print('#############################################')
#########################################################
# Output results
#########################################################
z_pop_iter     = torch.vstack((torch.hstack((z_pop_iter_bi ,torch.zeros(iters_burn_in,M))), z_pop_iter))
omega_pop_iter = torch.vstack((omega_pop_iter_bi, omega_pop_iter))
a_iter         = torch.hstack((a_iter_bi, a_iter))

printOutput_theo(z_pop, omega_pop, a, b, z_dim, nbatch, lengths.sum(), h, names_co, LL_lin_mu, LL_is)
plotConvergence_pop_theo_multiple(Elbo_iter, a_iter, z_pop_iter, omega_pop_iter, iters, kl_iter, gamma_iter, iters_burn_in)
plotConvergence_covariate_theo_multiple(z_pop_iter, iters, kl_iter, gamma_iter, iters_burn_in)
