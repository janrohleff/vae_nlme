"""
Created on 16.07.2025

@author: Jan Rohleff

Functions for visualization of the convergence of the VAE-nlme model.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt

def printOutput_theo(z_pop, omega_pop, a, b, z_dim, nbatch, n_tot, h, names, LL_lin, LL_is):
    ln_N = torch.log(torch.tensor(nbatch))
    ln_n_tot = torch.log(n_tot)
    z_pop_h = h(z_pop)
    print('')
    print('#############################################')
    print('ESTIMATION OF THE POPULATION PARAMETERS')
    print('#############################################')
    print('')
    print('Fixed Effects:')
    print(f'{"ka_pop:":<15} {z_pop_h[0]:>10.2f}')
    print(f'{"ke_pop:":<15} {z_pop_h[1]:>10.2f}')
    print(f'{"V_pop:":<15} {z_pop_h[2]:>10.2f}')
    count = 0
    for k in range(z_dim, len(z_pop)):
        if z_pop[k] != 0:
            print(f'{names[k-z_dim] + ":":<15} {z_pop[k]:>10.2f}')
            count += 1
    print('')
    print('Standard Deviation of the Random Effects:')
    print(f'{"omega_ka:":<15} {omega_pop[0].sqrt():>10.2f}')
    print(f'{"omega_ke:":<15} {omega_pop[1].sqrt():>10.2f}')
    print(f'{"omega_V:":<15} {omega_pop[2].sqrt():>10.2f}')

    print('')
    print('Error Model Parameters:')
    if a != 0:
        print(f'{"a:":<15} {a:>10.2f}')
    if b != 0:
        print(f'{"b:":<15} {b:>10.2f}')

    print('')
    print('#########################################################')
    print('ESTIMATION OF THE LOG LIKELIHOOD and INFORMATION CRITERIA')
    print('#########################################################')

    print(f'{"":<45} {"Linearization:":>30}  {"Importance Sampling:":>30}')
    print('-' * 135)

    # OFV
    print(f'{"-2Log likelihood (OFV):":<45}'f'{2*LL_lin:>30.2f}{2*LL_is:>30.2f}')

    # AIC
    print(f'{"Akaike Information Criteria (AIC):":<45}'
      f'{2*LL_lin + 2*(2*z_dim + 1 + count):>30.2f}'
      f'{2*LL_is + 2*(2*z_dim + 1 + count):>30.2f}')

    # BIC
    print(f'{"Bayesian Information Criteria (BIC):":<45}'
      f'{2*LL_lin + ln_N*(2*z_dim + 1 + count):>30.2f}'
      f'{2*LL_is + ln_N*(2*z_dim + 1 + count):>30.2f}')

    # BICc
    print(f'{"Corrected B.I. Criteria (BICc):":<45}'
      f'{2*LL_lin + ln_N*(z_dim + count) + ln_n_tot*(z_dim + 1):>30.2f}'
      f'{2*LL_is + ln_N*(z_dim + count) + ln_n_tot*(z_dim + 1):>30.2f}')


def plotConvergence_pop_theo(elbo_iter, a_iter, z_pop_iter, omega_pop_iter, iters, kl_iter, gamma_iter, iters_burn_in): 
    elbo_iter = elbo_iter.detach().numpy()
    elbo_iter = np.hstack([np.nan*np.zeros(100), elbo_iter])
    a_iter = a_iter.detach().numpy()
    z_pop_iter = z_pop_iter.detach().numpy()
    omega_pop_iter = omega_pop_iter.detach().numpy()

    fig, axs = plt.subplots(3, 3, sharex=False, sharey=False)
    fig.set_size_inches(10, 6)
    xmax = iters
    burn_in = iters_burn_in
    k_alpha = kl_iter + iters_burn_in
    k_beta = gamma_iter + iters_burn_in
    xticks = [iters_burn_in, kl_iter + iters_burn_in, gamma_iter+ iters_burn_in, iters+ iters_burn_in]
    xtick_labels = ['0', str(kl_iter), str(gamma_iter), str(iters)]
    plot_data = [
    (z_pop_iter[:, 0], r'$k_{a,pop}$'),
    (z_pop_iter[:, 1], r'$k_{e,pop}$'),
    (z_pop_iter[:, 2], r'$V_{pop}$'),
    (omega_pop_iter[:, 0], r'$\omega_{k_a}$'),
    (omega_pop_iter[:, 1], r'$\omega_{k_e}$'),
    (omega_pop_iter[:, 2], r'$\omega_{V}$'),
    (a_iter, r'$a$'),
    (elbo_iter, r'$-\mathcal{L}_\psi(x)$')
    ]

    for i, (data, title) in enumerate(plot_data):
        row, col = divmod(i, 3)
        ax = axs[row, col]

        if data is not None:
            ax.plot(data)
            ax.set_title(title, size = 14)
            ax.set_xlim(0, xmax)

            l1 = ax.axvspan(0, burn_in, facecolor=(0.83, 0.83, 0.83, 0.5), alpha=0.4)
            l2 = ax.axvline(x=k_alpha, linestyle='dashed', color='green')
            l3 = ax.axvline(x=k_beta, linestyle='dashed', color='red')

            ax.set_xticks(xticks)
            ax.set_xticklabels([])

            ymin, ymax = ax.get_ylim()
            offset_text = (ymax - ymin) / 16
            offset_label = (ymax - ymin) / 5
            for x, label in zip(xticks, xtick_labels):
                ax.text(x, ymin - offset_text, label, ha='center', va='top')
            ax.text(210, ymin - 1.2 * offset_label, 'Iterations', ha='center', va='top', fontsize=11)


    plt.tight_layout()
    axs[2,2].axis('off')
    axs[2, 2].legend([l1, l2, l3],['Burn in',r'$K_\alpha$',r'$K_\beta$'], loc='best', fontsize = 'x-large')

    plt.subplots_adjust(hspace = 1)
    plt.savefig('VAE_nlme/Plots/theophylline_convergence_popParam.pdf', dpi=500)
    plt.show()

def plotConvergence_covariate_theo(z_pop_iter, iters, kl_iter, gamma_iter, iters_burn_in): 
    z_pop_iter = z_pop_iter.detach().numpy()

    fig, axs = plt.subplots(2, 3, sharex=False, sharey=False)
    fig.set_size_inches(10, 4)
    xmax = iters
    k_alpha = kl_iter 
    k_beta = gamma_iter 
    xticks = [0, kl_iter, gamma_iter, iters]
    xtick_labels = ['0', str(kl_iter), str(gamma_iter), str(iters)]
    plot_data = [
    (z_pop_iter[iters_burn_in:, 3], r'$\beta_{k_a}^{sex}$'),
    (z_pop_iter[iters_burn_in:, 5], r'$\beta_{k_e}^{sex}$'),
    (z_pop_iter[iters_burn_in:, 7], r'$\beta_{V}^{sex}$'),
    (z_pop_iter[iters_burn_in:, 4], r'$\beta_{k_a}^{w}$'),
    (z_pop_iter[iters_burn_in:, 6], r'$\beta_{k_e}^{w}$'),
    (z_pop_iter[iters_burn_in:, 8], r'$\beta_{V}^{w}$')
    ]

    for i, (data, title) in enumerate(plot_data):
        row, col = divmod(i, 3)
        ax = axs[row, col]

        if data is not None:
            ax.plot(data)
            ax.set_title(title, size = 14)
            ax.set_xlim(0, xmax)
            if data[-1] == 0:
                ymin, ymax = ax.get_ylim()
                ax.axhspan(ymin, ymax, facecolor=(0.83, 0.83, 0.83), alpha=0.4, zorder=0)
                ax.set_ylim(ymin, ymax)

            # vertikale Linien überall
            ax.axvline(x=k_alpha, linestyle='dashed', color='green')
            ax.axvline(x=k_beta, linestyle='dashed', color='red')

            ax.set_xticks(xticks)
            ax.set_xticklabels([])

            ymin, ymax = ax.get_ylim()
            offset_text = (ymax - ymin) / 16
            offset_label = (ymax - ymin) / 5
            for x, label in zip(xticks, xtick_labels):
                ax.text(x, ymin - offset_text, label, ha='center', va='top')
            ax.text(160, ymin - 1.2 * offset_label, 'Iterations', ha='center', va='top', fontsize=11)


    plt.tight_layout()

    plt.subplots_adjust(hspace = 1)
    plt.savefig('Plots/theophylline_convergence_covariate.pdf', dpi=1000)
    plt.show()

def printOutput_neonates(z_pop, omega_pop, a, b, z_dim, nbatch, n_tot, h, names, LL_lin_mu, LL_is):
    ln_N = torch.log(torch.tensor(nbatch))
    ln_n_tot = torch.log(n_tot)
    z_pop_h = h(z_pop)
    print('')
    print('#############################################')
    print('ESTIMATION OF THE POPULATION PARAMETERS')
    print('#############################################')
    print('')
    print('Fixed Effects:')
    print(f'{"W0_pop:":<15} {z_pop_h[0]:>10.2f}')
    print(f'{"kin_pop:":<15} {z_pop_h[1]:>10.2f}')
    print(f'{"Tlag_pop:":<15} {z_pop_h[2]:>10.2f}')
    print(f'{"kout_pop:":<15} {z_pop_h[3]:>10.2f}')
    print(f'{"T50_pop:":<15} {z_pop_h[4]:>10.2f}')
    count = 0
    for k in range(z_dim, len(z_pop)):
        if z_pop[k] != 0:
            print(f'{names[k-z_dim] + ":":<15} {z_pop[k]:>10.2f}')
            count += 1
    print('')
    print('Standard Deviation of the Random Effects:')
    print(f'{"omega_W0:":<15} {omega_pop[0].sqrt():>10.2f}')
    print(f'{"omega_kin:":<15} {omega_pop[1].sqrt():>10.2f}')
    print(f'{"omega_Tlag:":<15} {omega_pop[2].sqrt():>10.2f}')
    print(f'{"omega_kout:":<15} {omega_pop[3].sqrt():>10.2f}')
    print(f'{"omega_T50:":<15} {omega_pop[4].sqrt():>10.2f}')

    print('')
    print('Error Model Parameters:')
    if a != 0:
        print(f'{"a:":<15} {a:>10.2f}')
    if b != 0:
        print(f'{"b:":<15} {b:>10.2f}')

    print('')
    print('#########################################################')
    print('ESTIMATION OF THE LOG LIKELIHOOD and INFORMATION CRITERIA')
    print('#########################################################')

    print(f'{"":<45} {"Linearization:":>30}{"Importance Sampling:":>30}')
    print('-' * 135)

    # OFV
    print(f'{"-2Log likelihood (OFV):":<45}'f'{2*LL_lin_mu:>30.2f}{2*LL_is:>30.2f}')

    # AIC
    print(f'{"Akaike Information Criteria (AIC):":<45}'
      f'{2*LL_lin_mu + 2*(2*z_dim + 1 + count):>30.2f}'
      f'{2*LL_is + 2*(2*z_dim + 1 + count):>30.2f}')

    # BIC
    print(f'{"Bayesian Information Criteria (BIC):":<45}'
      f'{2*LL_lin_mu + ln_N*(2*z_dim + 1 + count):>30.2f}'
      f'{2*LL_is + ln_N*(2*z_dim + 1 + count):>30.2f}')

    # BICc
    print(f'{"Corrected B.I. Criteria (BICc):":<45}'
      f'{2*LL_lin_mu + ln_N*(z_dim + count) + ln_n_tot*(z_dim + 1):>30.2f}'
      f'{2*LL_is + ln_N*(z_dim + count) + ln_n_tot*(z_dim + 1):>30.2f}')
    
def plotConvergence_pop_neonates(elbo_iter, a_iter, z_pop_iter, omega_pop_iter, iters, kl_iter, gamma_iter, iters_burn_in): 
    elbo_iter = elbo_iter.detach().numpy()
    elbo_iter = np.hstack([np.nan*np.zeros(100), elbo_iter])
    a_iter = a_iter.detach().numpy()
    z_pop_iter = z_pop_iter.detach().numpy()
    omega_pop_iter = omega_pop_iter.detach().numpy()

    fig, axs = plt.subplots(3, 4, sharex=False, sharey=False)
    fig.set_size_inches(10, 6)
    xmax = iters
    burn_in = iters_burn_in
    k_alpha = kl_iter + iters_burn_in
    k_beta = gamma_iter + iters_burn_in
    xticks = [iters_burn_in, kl_iter + iters_burn_in, gamma_iter+ iters_burn_in, iters+ iters_burn_in]
    xtick_labels = ['0', str(kl_iter), str(gamma_iter), str(iters)]
    plot_data = [
    (z_pop_iter[:, 0], r'$W_{0,pop}$'),
    (z_pop_iter[:, 1], r'$k_{in,pop}$'),
    (z_pop_iter[:, 2], r'$T_{lag,pop}$'),
    (z_pop_iter[:, 3], r'$k_{out,pop}$'),
    (z_pop_iter[:, 4], r'$T_{50,pop}$'),
    (omega_pop_iter[:, 0], r'$\omega_{W_0}$'),
    (omega_pop_iter[:, 1], r'$\omega_{k_{in}}$'),
    (omega_pop_iter[:, 2], r'$\omega_{T_{lag}}$'),
    (omega_pop_iter[:, 3], r'$\omega_{k_{out}}$'),
    (omega_pop_iter[:, 4], r'$\omega_{T_{50}}$'),
    (a_iter, r'$a$'),
    (elbo_iter, r'$-\mathcal{L}_\psi(x)$')
    ]

    for i, (data, title) in enumerate(plot_data):
        row, col = divmod(i, 4)
        ax = axs[row, col]

        if data is not None:
            ax.plot(data)
            ax.set_title(title, size = 14)
            ax.set_xlim(0, xmax)

            l1 = ax.axvspan(0, burn_in, facecolor=(0.83, 0.83, 0.83, 0.5), alpha=0.4)
            l2 = ax.axvline(x=k_alpha, linestyle='dashed', color='green')
            l3 = ax.axvline(x=k_beta, linestyle='dashed', color='red')

            ax.set_xticks(xticks)
            ax.set_xticklabels([])

            ymin, ymax = ax.get_ylim()
            offset_text = (ymax - ymin) / 16
            offset_label = (ymax - ymin) / 5
            for x, label in zip(xticks, xtick_labels):
                ax.text(x, ymin - offset_text, label, ha='center', va='top')
            ax.text(210, ymin - 1.2 * offset_label, 'Iterations', ha='center', va='top', fontsize=11)


    plt.tight_layout()
    #axs[2,2].axis('off')
    axs[2, 3].legend([l1, l2, l3],['Burn in',r'$K_\alpha$',r'$K_\beta$'], loc='best', fontsize = '12')

    plt.subplots_adjust(hspace = 1)
    plt.savefig('VAE_nlme/Plots/neonates_convergence_popParam.pdf', dpi=500)
    plt.show()

def plotConvergence_covariate_neonates(z_pop_iter, iters, kl_iter, gamma_iter, iters_burn_in): 
    z_pop_iter = z_pop_iter.detach().numpy()

    fig, axs = plt.subplots(5, 5, sharex=False, sharey=False)
    fig.set_size_inches(18, 8)
    xmax = iters
    k_alpha = kl_iter 
    k_beta = gamma_iter 
    xticks = [0, kl_iter, gamma_iter, iters]
    xtick_labels = ['0', str(kl_iter), str(gamma_iter), str(iters)]
    plot_data = [
    (z_pop_iter[iters_burn_in:, 5], r'$\beta_{W_0,sex}$'),
    (z_pop_iter[iters_burn_in:, 6], r'$\beta_{W_0,DelM}$'),
    (z_pop_iter[iters_burn_in:, 7], r'$\beta_{W_0,GA}$'),
    (z_pop_iter[iters_burn_in:, 8], r'$\beta_{W_0,Mage}$'),
    (z_pop_iter[iters_burn_in:, 9], r'$\beta_{W_0,Para_2}$'),
    (z_pop_iter[iters_burn_in:, 10], r'$\beta_{k_{in},sex}$'),
    (z_pop_iter[iters_burn_in:, 11], r'$\beta_{k_{in},DelM}$'),
    (z_pop_iter[iters_burn_in:, 12], r'$\beta_{k_{in},GA}$'),
    (z_pop_iter[iters_burn_in:, 13], r'$\beta_{k_{in},Mage}$'),
    (z_pop_iter[iters_burn_in:, 14], r'$\beta_{k_{in},Para_2}$'),
    (z_pop_iter[iters_burn_in:, 15], r'$\beta_{T_{lag}, sex}$'),
    (z_pop_iter[iters_burn_in:, 16], r'$\beta_{T_{lag}, DelM}$'),
    (z_pop_iter[iters_burn_in:, 17], r'$\beta_{T_{lag}, GA}$'),
    (z_pop_iter[iters_burn_in:, 18], r'$\beta_{T_{lag}, Mage}$'),
    (z_pop_iter[iters_burn_in:, 19], r'$\beta_{T_{lag}, Para_2}$'),
    (z_pop_iter[iters_burn_in:, 20], r'$\beta_{k_{out}, sex}$'),
    (z_pop_iter[iters_burn_in:, 21], r'$\beta_{k_{out}, DelM}$'),
    (z_pop_iter[iters_burn_in:, 22], r'$\beta_{k_{out}, GA}$'),
    (z_pop_iter[iters_burn_in:, 23], r'$\beta_{k_{out}, Mage}$'),
    (z_pop_iter[iters_burn_in:, 24], r'$\beta_{k_{out}, Para_2}$'),
    (z_pop_iter[iters_burn_in:, 25], r'$\beta_{T_{50}, sex}$'),
    (z_pop_iter[iters_burn_in:, 26], r'$\beta_{T_{50}, DelM}$'),
    (z_pop_iter[iters_burn_in:, 27], r'$\beta_{T_{50}, GA}$'),
    (z_pop_iter[iters_burn_in:, 28], r'$\beta_{T_{50}, Mage}$'),
    (z_pop_iter[iters_burn_in:, 29], r'$\beta_{T_{50}, Para_2}$'),

    ]

    for i, (data, title) in enumerate(plot_data):
        row, col = divmod(i, 5)
        ax = axs[row, col]

        if data is not None:
            ax.plot(data)
            ax.set_title(title, size = 14)
            ax.set_xlim(0, xmax)
            if data[-1] == 0:
                ymin, ymax = ax.get_ylim()
                ax.axhspan(ymin, ymax, facecolor=(0.83, 0.83, 0.83), alpha=0.4, zorder=0)
                ax.set_ylim(ymin, ymax)

            # vertikale Linien überall
            ax.axvline(x=k_alpha, linestyle='dashed', color='green')
            ax.axvline(x=k_beta, linestyle='dashed', color='red')

            ax.set_xticks(xticks)
            ax.set_xticklabels([])

            ymin, ymax = ax.get_ylim()
            offset_text = (ymax - ymin) / 16
            offset_label = (ymax - ymin) / 5
            for x, label in zip(xticks, xtick_labels):
                ax.text(x, ymin - offset_text, label, ha='center', va='top')
            ax.text(160, ymin - 1.2 * offset_label, 'Iterations', ha='center', va='top', fontsize=11)


    plt.tight_layout()

    plt.subplots_adjust(hspace = 1)
    plt.savefig('Plots/neonates_convergence_covariate.pdf', dpi=1000)
    plt.show()