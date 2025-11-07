"""
Created on 16.07.2025

@author: Jan Rohleff

All functions for the VAE-nlme model.
"""

#########################################################
# Import
#########################################################
import torch
import numpy as np

def kldiv_normal_normal(mean1:torch.Tensor, var1:torch.Tensor, mean2:torch.Tensor, var2:torch.Tensor, log_diag):
    """
    KL divergence between normal distributions, KL( N(mean1, diag(exp(lnvar1))) || N(mean2, diag(exp(lnvar2))) ),
    where diag(exp(lnvar1)) is a full covariance matrix and diag(exp(lnvar2)) is a diagonal covariance matrix.
    """
    k = mean1.shape[0]
    
    return 0.5 * (torch.trace(1/var2*var1) - k + torch.sum((mean2-mean1).pow(2)/var2) + torch.log(var2).sum()- 2*log_diag.sum())
   
def p_x_z_compute(data, x_mean, res, lengths):
    a = res[0]
    b = res[1]
    sigma = a + b * x_mean
    mask = torch.arange(data.shape[1]).unsqueeze(0) < lengths.unsqueeze(1)
    data_masked = data[mask]
    x_mean_masked = x_mean[mask]
    sigma_masked = sigma[mask]
    err =(data_masked - x_mean_masked) / sigma_masked
    ln_pi = torch.log(2 * torch.tensor(np.pi))
    N_tot = lengths.sum()
    p_x_z = 0.5 * torch.sum(err.pow(2)) + 0.5 * N_tot * ln_pi + torch.log(sigma_masked).sum()
    
    return p_x_z

def p_z_compute(z, z_pop, omega_pop):
    ln_pi = torch.log(2*torch.tensor(np.pi))
    p_z = 0.5*torch.sum((z-z_pop).pow(2)/omega_pop + omega_pop.log() + ln_pi)
    
    return p_z  

def q_z_x_compute(eps, sigma):
    ln_pi = torch.log(2*torch.tensor(np.pi))
    q_z_x = 0.5 * torch.sum(eps.pow(2) + ln_pi + 2*sigma.log())
     
    return q_z_x

def p_z_compute_batch(z, z_pop, omega_pop):
    ln_pi = torch.log(2*torch.tensor(np.pi))
    p_z = 0.5*torch.sum(((z-z_pop).pow(2)/omega_pop + omega_pop.log() + ln_pi), dim =1)
    return p_z   

def q_z_x_compute_batch(eps, sigma):
    ln_pi = torch.log(2*torch.tensor(np.pi))
    q_z_x = 0.5 * torch.sum( (eps.pow(2) + ln_pi + 2 * sigma.log()), dim =1)
        
    return q_z_x

def p_x_z_compute_lengths_batch(data, x_mean, res, lengths):
    a = res[0]
    b = res[1]
    sigma = (a + b * x_mean).detach()
    p_x_z = torch.zeros(data.shape[0])
    for i in range(data.shape[0]):
        err = (data[i,:lengths[i]] - x_mean[i,:lengths[i]]) / sigma[i,:lengths[i]]
        p_x_z[i] = 0.5 * torch.sum(err.pow(2)) + 0.5 * lengths[i] * torch.log(2 * torch.tensor(np.pi)) + torch.log(sigma[i,:lengths[i]]).sum()
    
    return p_x_z

def LogLikelihood_linearization(z_pop_hat, omega_pop, res, data, phi, C, h, time, lengths, Decoder):
    N = data.shape[0]
    z_dim = omega_pop.shape[0]
    z_pop = torch.zeros(N, z_dim)
    for i in range(N):
        z_pop[i] = torch.matmul(C[i], z_pop_hat)
       
    # Functions 
    phi_normal = phi.clone().detach()
    phi0 = phi.clone().detach().requires_grad_(True)
    pred_x = Decoder(phi0, time, h)
    
    # Compute log likelihood
    mu = torch.zeros(N, data.shape[1])
    df_dphi = torch.zeros(N, data.shape[1], z_dim)
    ff = torch.zeros(N,data.shape[1])
    for t in range(data.shape[1]):
        ff[:,t] = pred_x[:,t,0]
        output = ff[:,t].sum().backward(retain_graph=True)
        df_dphi[:,t] = phi0.grad
        mu[:,t] = ff[:,t] 
        for zz in range(z_dim):
            mu[:,t] += phi0.grad[:,zz]*(z_pop[:,zz] - phi_normal[:,zz])
        phi0.grad.zero_()

    LL = 0
    for i in range(N):
        g = res[0]+res[1]*ff[i,:lengths[i]]
        Sigma = torch.eye(lengths[i])#torch.cov(err_big[:,i,:,0].T)
        tmp = torch.matmul(g.diag(), torch.matmul(Sigma, g.diag()))
        #dzf = torch.matmul(df_dphi[i], J_h[i])
        dzf = df_dphi[i, :lengths[i]]
        Gamma = torch.matmul(dzf, torch.matmul(omega_pop.diag(), dzf.T))+ tmp
        LL += lengths[i]/2*torch.log(2*torch.tensor(np.pi))  +0.5*torch.linalg.slogdet(Gamma).logabsdet + 0.5*torch.matmul(data[i,:lengths[i],1] - mu[i, :lengths[i]], torch.linalg.solve(Gamma, data[i,:lengths[i],1] - mu[i,:lengths[i]]))
        
    return LL

def LogLikelihood_sample(M, z_pop_hat, omega_pop, res, data, mu, L, C, h, time, lengths, Decoder):
    nbatch = data.shape[0]
    z_dim = omega_pop.shape[0]
    z_pop = torch.zeros(nbatch, z_dim)
    for i in range(nbatch):
        z_pop[i] = torch.matmul(C[i], z_pop_hat)
    #########################################################
    # Compute true posterior
    #########################################################
    K = 100
    z_samples = torch.zeros(K, nbatch, z_dim)
    log_weights = torch.zeros(K, nbatch)

    for k in range(K):
        eps = torch.randn(nbatch, z_dim)
        z_normal = mu + torch.matmul(L, eps.unsqueeze(-1)).squeeze(-1)
        z_samples[k] = z_normal 
        pred_x = Decoder(z_normal, time, h)    
        p_x_z = p_x_z_compute_lengths_batch(data[:,:,1].view(nbatch,data.shape[1],1) , pred_x, res, lengths)
        p_z  = p_z_compute_batch(z_normal, z_pop, omega_pop)
        q_z = q_z_x_compute_batch(eps, torch.diagonal(L, dim1=1, dim2=2))
        log_weights[k,:] = -(p_x_z + p_z - q_z)
        weights_normalized = log_weights[:k+1,:] - torch.logsumexp(log_weights[:k+1,:], dim=0).view(1,nbatch)
        weights = torch.exp(weights_normalized)

    mu_true = torch.sum(weights.unsqueeze(-1) * z_samples[:k+1], dim=0)  
    diff = z_samples[:k+1] - mu_true
    outer = diff.unsqueeze(-1) @ diff.unsqueeze(-2)  
    weights_ = weights.unsqueeze(-1).unsqueeze(-1)  
    cov_true = torch.sum(weights_ * outer, dim=0)
    L_true = torch.linalg.cholesky(cov_true)
    #########################################################
    # Compute Likelihood
    #########################################################
    log_w = torch.zeros(M,nbatch)
    for m in range(M):      
        eps = torch.randn(nbatch, z_dim)
        z_sample = mu_true + torch.matmul(L_true, eps.unsqueeze(-1)).squeeze(-1)
        pred_x = Decoder(z_sample, time, h)  
        p_x_z = p_x_z_compute_lengths_batch(data[:,:,1].view(nbatch,data.shape[1],1) , pred_x, res, lengths)
        p_z  = p_z_compute_batch(z_sample, z_pop, omega_pop)
        q_z = q_z_x_compute_batch(eps, torch.diagonal(L_true, dim1=1, dim2=2))
        log_w[m] = -(p_x_z + p_z - q_z)
        log_px = torch.logsumexp(log_w[:m+1], dim =0) - torch.log(torch.tensor(m+1))
 
    return -log_px.sum(), - log_px

def initalizeEncoder(iters, L_iter, Encoder, Decoder, data, data_in, z_dim, covariates_in, lengths, h, pop):
    print('#############################################')
    print('BURN IN phase')
  
    optimizer = torch.optim.Adam([{'params': Encoder.parameters()}  ], lr=8e-3)
    b = torch.tensor(0)
    #########################################################
    # Save iterations
    #########################################################
    z_pop_iter_bi = torch.zeros(iters, z_dim)
    omega_pop_iter_bi = torch.zeros(iters,z_dim)
    a_iter_bi = torch.zeros(iters)
    elbo_iter_bi = torch.zeros(iters)
    count = 0
    #########################################################
    for iter in range(iters*L_iter):
        # Encoder
        z_normal, mu, L, log_sigma, eps = Encoder(data_in, covariates_in, lengths)
        # Decoder
        pred_x = Decoder(z_normal, data[:,:,0], h)
        # Update population parameters
        z_pop, omega_pop, a, _ = pop.update_pop(mu.detach(), L.detach(), pred_x.detach(), 0)
        # Compute loss
        p_x_z = p_x_z_compute(data[:,:,1].view(data.shape[0],data.shape[1],1), pred_x, [a,b], lengths)
        p_z  = p_z_compute(z_normal, z_pop, omega_pop)
        q_z = q_z_x_compute(eps, torch.diagonal(L, dim1=1, dim2=2))
        DKL = p_z - q_z 
        elbo = p_x_z + 0.001 * DKL   
        elbo.backward()
        optimizer.step()
        # Reset gradient
        optimizer.zero_grad()
        if iter % L_iter == 0:
            print(f"Iteration {count + 1}/{iters}", end="\r")
            z_pop_iter_bi[count] = h(z_pop)
            omega_pop_iter_bi[count] = omega_pop.sqrt()
            a_iter_bi[count] = a
            elbo_iter_bi[count] = (p_x_z + DKL).detach()
            count += 1

    return Encoder, optimizer, pred_x, mu, L, a.detach(), b.detach(), z_pop_iter_bi, omega_pop_iter_bi, a_iter_bi, elbo_iter_bi


