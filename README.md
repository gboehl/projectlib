
# Companion Codes
## Companion codes to replicate key-figures from my papers

The full papers can be found on my website: [gregorboehl.com](gregorboehl.com)

This repo contains the codes for

1. mp_spec: Monetary Policy and Speculative Stock Markets 
2. taxation_wealth_dynamics: Can Taxation Predict US Top-Wealth Share Dynamics? (w. Thomas Fischer)

The folders are each self-contained and the code is modified such that you can run it directly from within a folder that contains all the files, i.e. via command line: 

E.g: `python coef.py`

The simulation code is written in Python. Although I used Python version 3.6.4 the code should be backwards compatible and also run under Python 2.7 (not tested). I did not test any of the code under Windows of Mac OS.

## mp_spec
### Codes for "Monetary Policy and Speculative Stock Markets"

The *.py files are the following:
  
   * base.py: definitions of functions used by other files.
   * oos.py: bifurcation diagrams for phi_s, alpha and gamma (Figure 2 and Figures in Appendix D). Chose in the first paragraph of the code which diagram to plot.
   * coef.py: code for Figure 1.
   * msm.py: code for method of simulated moments. Chose in the first paragraph of the code which model to estimate (with or without exogenous shocks to stock prices, with or without behavioral speculation).
   * vars.py: code for Figure 3.

Additionally I provide the data files (*.xlsx) containing the data as described in the paper.

## taxation_wealth_dynamics
### Codes for "Can Taxation Predict US Top-Wealth Share Dynamics" (w. Thomas Fischer)

The *.py files are the following:
  
   * lib.py: definitions of functions used by other files.
   * oos.py: simulations/out-of-sample forecasts
   * residual.py: plot residuals and show residual analisis
   * projections.py: tax projections

Additionally I provide the data file (datac.npz) containing the data as described in the paper.
