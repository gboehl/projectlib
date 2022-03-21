#!/bin/python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
from pydsge import *
from grgrlib import *
from matplotlib import use
from tqdm import tqdm

plt.rc('text', usetex=True)
plt.rc('legend', fontsize=18)

# set all paths relative to script location
pth = os.path.dirname(os.path.abspath(__file__))
output_path = os.path.join(os.path.split(pth)[0],'output')
dfile0 = os.path.join(output_path, 'bgs_final0_meta.npz')

# load model and reinitialize estimation routines
mod0 = DSGE.load(dfile0, force_parse=False)
mod0.load_estim(l_max=None, k_max=None)

N = 1000

# define function to obtain impulse response given a parameter vector
# find a shock such that the maximum of the exogenous process equals the empirical maximum (see online appendix)
def get_irfs(pars):
    r1 = mod0.get_par('rootk1', pars) + 1e-5
    r2 = mod0.get_par('rootk2', pars)

    smax = (np.log(np.log(r2)/np.log(r1)))/np.log(r1/r2)
    xmax = (r1**smax - r2**smax)/(r1 - r2)
    sig = mod0.data['CB_Loans'].max()/xmax

    return mod0.irfs(('e_qe_k',sig), pars)

# get sample from posterior
pars0 = mod0.get_par('posterior', nsamples=N)

# run simulations
irf0 = []
for p in tqdm(pars0, total=N):
    irf0.append(get_irfs(p)[0].to_numpy())

ir0 = np.array(irf0)

# set alternative parameters 
pars1 = mod0.get_par('post_mean')
pars2 = pars1.copy()
pars1 = mod0.set_par('sig_c', 1.5, pars1)
pars2 = mod0.set_par('psi', .5, pars2)

# run more simulations
ir1 = get_irfs(pars1)[0]

# define variables of interest
v1 = ['Pi','c','l']
vp1 = ['Inflation','Consumption','Labor hours']
f1 = np.array((4,1,1))

plt.rcParams['axes.prop_cycle'] = plt.cycler("color", plt.get_cmap('hot')(np.linspace(0.0,.5,2)))

# plotting...
figs, axs = figurator(1,3,1,figsize=(12,4))
pplot( f1*ir0[...,mod0.vix(v1)], ax=axs, colors='C0', legend='posterior', alpha=.2)
pplot( f1*ir1[v1], labels=vp1, ax=axs, colors='C1', legend='prior $\sigma_c$', styles='--')
axs[-1].legend(framealpha=.8)
figs[0].tight_layout()
figs[0].savefig(output_path+mod0.name+'_defl0_4pub.pdf')

mod0.pool.terminate()

