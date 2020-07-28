
# Project Codes

The full papers can be found on my website: [gregorboehl.com](gregorboehl.com)

This repo contains the *.yaml* for several projects and non-projects (in the `yamls` folds) and the codes for

1. `mp_spec`: Monetary Policy and Speculative Asset Markets 
2. `re_chaos`: On the Evolutionary Fitness of Rationality

The simulation code is written in Python. Although I used Python version 3.7+ the code should be backwards compatible and also run under Python 2.7 (not tested). I also didn't do any testing under Windows or Mac OS.

## `mp_spec`
### Codes for "Monetary Policy and Speculative Asset Markets"

The *.py files are the following:
  
   * base.py: definitions of functions used by other files.
   * oos.py: bifurcation diagrams for phi_s, alpha and gamma (Figure 2 and Figures in Appendix D). Chose in the first paragraph of the code which diagram to plot.
   * coef.py: code for Figure 1.
   * msm.py: code for method of simulated moments. Chose in the first paragraph of the code which model to estimate (with or without exogenous shocks to stock prices, with or without behavioral speculation).
   * vars.py: code for Figure 3.

Additionally I provide the data files (*.xlsx) containing the data as described in the paper.

## `re_chaos`
### Codes for "On the Evolutionary Fitnes of Rationality" (with Cars Hommes)

(description coming soon)
