# Variational Autoencoders in Non-Linear Mixed Effects Modeling

This repository contains the Python code used to perform the numerical simulations presented in the paper:

**Redefining Parameter Estimation and Covariate Selection via Variational Autoencoders: One run is all you need**  

**Authors:**  **Jan Rohleff**, Freya Bachmann, Uri Nahum, Dominic Bräm, Britta Steffens, 
Marc Pfister, Gilbert Koch, Johannes Schropp.

*Journal: CPT: Pharmacometrics & Systems Pharmacology, 2025*
*DOI: https://doi.org/10.1002/psp4.70129*

The entire code was written by Jan Rohleff.

## Overview

The simulations reproduce the results presented in Case Study 1 with a single dose and multiple dosing. Due to privacy restrictions, we are not permitted to share the real data from Case Study 2. Instead, we provide simulated data that illustrates the behavior of the VAE in this second case.

### Key Features

- Implementation of a Variational Autoencoder applied to nonlinear mixed-effects modeling
- Simultaneous parameter estimation and covariate selection
- One RUN is all you need
-  Requires the following Python packages: `torch`, `torchode`, `numpy`, `cvxpy`, `gurobipy`, `matplotlib` and `scipy`


## Installation

Clone the repository and install the dependencies:

```bash
git clone https://github.com/janrohleff/vae_nlme.git
cd vae_nlme
pip install torch torchode numpy cvxpy gurobipy matplotlib scipy
```

## Run
To run the examples, execute one of the following scripts:

**Case study 1 – Theophylline data:**
```bash
python Main/theophylline.py
```

**Case study 1 – Theophylline data with multiple dosing:**
```bash
python Main/theophylline_multiple.py
```

**Case study 2 – Neonate data:**
```bash
python Main/neonates.py
```
Convergence results were stored in `Plots/`.


## Repository Structure

- **`Data/`**
  - `neonates_data.csv` – Weight progression data for neonate (Case Study 2)
  - `neonates_data.pt` – Weight progression data for neonate (Case Study 2)
  - `neonates_data_lengths.pt` – Number of observations per individual (Case Study 2)
  - `theophylline_data.pt` – Pharmacokinetic data (Case Study 1)
  - `theophylline_data.tab` – Pharmacokinetic data (Case Study 1)
  - `theophylline_multiple_data.pt` – Pharmacokinetic data with multiple dosing (Case Study 1)
  - `theophylline_multiple_data.tab` – Pharmacokinetic data with multiple dosing (Case Study 1)

- **`Main/`**
  - `neonates.py` – Main VAE training + evaluation for Case Study 2
  - `theophylline_multiple.py` – Main VAE training + evaluation for Case Study 1 (multiple dosing)
  - `theophylline.py` – Main VAE training + evaluation for Case Study 1
 
- **`ParaUpdate/`**
  - `pop_parameter.c` – Population parameter updates during training phase
  - `pop_parameter.cp313-win_amd64.pyd` - compiled for Windows (Python 3.13)
  - `pop_parameter.cpython-313-darwin.so` – compiled for macOS (Python 3.13)
  - `pop_parameter.cpython-36m-x86_64-linux-gnu.so` – compiled for Linux (Python 3.13)

- **`Plots/`**
  - `neonates_convergence_covariate.pdf` – Covariate convergence (Case Study 2)  
  - `neonates_convergence_popParam.pdf` – Population parameter convergence (Case Study 2)
  - `neonates_convergence_data.pdf` – Data of Case Study 2
  - `theophylline_convergence_covariate.pdf` – Covariate convergence (Case Study 1)  
  - `theophylline_convergence_popParam.pdf` – Population parameter convergence (Case Study 1)
  - `theophylline_data.pdf` – Data of Case Study 1
  - `theophylline_multiple_convergence_covariate.pdf` – Covariate convergence (Case Study 1) (multiple dosing) 
  - `theophylline_multiple_convergence_popParam.pdf` – Population parameter convergence (Case Study 1) (multiple dosing) 
  - `theophylline_multiple_data.pdf` – Data of Case Study 1 (multiple dosing) 
    
- **`VAE/`**
  - `decoder.py` – VAE Decoder
  - `encoder.c` – VAE Encoder
  - `encoder.cp313-win_amd64.pyd` - compiled for Windows (Python 3.13)
  - `encoder.cpython-313-darwin.so` –  compiled for macOS (Python 3.13)
  - `encoder.cpython-36m-x86_64-linux-gnu.so` – compiled for Linux (Python 3.13)

- **`README.md`** – This file
- **`functions.py`** – Core helper functions
- **`functions_neonates.py`** – initialization functions for neonate
- **`functions_theo.py`** – initialization functions for theophylline 
- **`visualization.py`** – generates plots and formatted summaries

