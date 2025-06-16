# Variational Autoencoders in Non-Linear Mixed Effects Modeling

This repository contains the Python code used to perform the numerical simulations presented in the paper:

**Redefining Parameter Estimation and Covariate Selection via Variational Autoencoders: One run is all you need**  
*Author(s): Jan Rohleff, Dominic Bräm, Freya Bachmann, Uri Nahum, Britta Steffens, 
Marc Pfister, Gilbert Koch, Johannes Schropp.

*Journal: Name, Year*  
*DOI: [doi-link]*

The entire code was written by Jan Rohleff.

## Overview

The simulations reproduce the results presented in Case Study 1. Due to privacy restrictions, we are not permitted to share the real data from Case Study 2. Instead, we provide simulated data that illustrates the behavior of the VAE in this second case.

### Key Features

- Implementation of a Variational Autoencoder applied to nonlinear mixed-effects modeling
- Simultaneous parameter estimation and covariate selection
- One RUN is all you need
-  Requires the following Python packages: `torch`, `numpy`, `cvxpy`, `matplotlib`, and `scipy`


### Installation

Clone the repository and install the dependencies:

```bash
git clone https://github.com/your-username/vae-nlme.git
cd vae-nlme
pip install torch numpy cvxpy matplotlib scipy
```


## Repository Structure

├── Data/
│   └── neonates_data_lengths.pt    # Data for Case Study 2 (Neonate Weight Progression)
│   └── neonates_data.pt            # Number of observations for every individual for Case Study 2
│   └── theophylline_data.pt        # Data for Case Study 1 (Theophylline Pharmacokinetics)

├── Examples/
│   └── neonates.py                 # Main VAE training + results for Case Study 2
│   └── theophylline.py             # Main VAE training + results for Case Study 1

├── Plots/
│   └── neonates_convergence_covariate.pdf                     # VAE Covariate-Convergence for Case Study 2
│   └── neonates_convergence_popParam.pdf                      # VAE Population-Parameter-Convergence for Case Study 2
│   └── theophylline_convergence_covariate.pdf                 # VAE Covariate-Convergence for Case Study 2
│   └── theophylline_convergence_popParam.pdf                  # VAE Population-Parameter-Convergence for Case Study 2

├── functions.py        # All important functions
├── pop_parameter.py    # Updating Population Parameter during the Training
├── vae.py              # Variational Autoencoder architecture 
├── visualization.py    # Print + Vizualize the training results
├── README.md       

