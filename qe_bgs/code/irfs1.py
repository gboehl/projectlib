#!/bin/python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
from pydsge import *
from grgrlib import *

plt.rc('text', usetex=True)
plt.rc('legend', fontsize=18)

# set all paths relative to script location
pth = os.path.dirname(os.path.abspath(__file__))
output_path = os.path.join(os.path.split(pth)[0],'output')
dfile0 = os.path.join(output_path, 'bgs_final0_meta.npz')

# load model and reinitialize estimation routines
mod0 = DSGE.load(dfile0, force_parse=False)
mod0.load_estim()

N0 = 2000
N1 = 2000

# adjust/normalize priors
mod0.prior['rootk1'][-2:] = [.9, .05]
mod0.prior['rootb1'][-2:] = [.9, .05]
mod0.prior['rootk2'][-2:] = [.9, .05]
mod0.prior['rootb2'][-2:] = [.9, .05]
mod0.prior['rho_cbl'][-2:] = [.75,.05]

# load parameters
pars0 = mod0.get_par('prior', nsamples=N0, verbose=True)
pars1 = mod0.get_par('posterior', nsamples=N1, verbose=True)

# define function to obtain impulse response given a parameter vector
# find a shock such that the maximum of the exogenous process equals the empirical maximum (see online appendix)
def pfunc(par):
    r1 = mod0.get_par('rootk1', par)
    r2 = mod0.get_par('rootk2', par)

    smax = (np.log(np.log(r2)/np.log(r1)))/np.log(r1/r2)
    xmax = (r1**smax - r2**smax)/(r1 - r2)
    sig = mod0.data['CB_Loans'].max()/xmax

    return mod0.irfs(('e_qe_k',sig,0), par, T=40, linear=True)[0]

# run irfs
ir00 = [pfunc(p).to_numpy() for p in pars0]
print('ir00 done')
ir10 = [pfunc(p).to_numpy() for p in pars1]
print('ir10 done')

# define variables of interest
sigma = .05
v = ['y','c','w','i','Pi','excessreturnkb','lev','n','kb'] # 3x3
vp = ['Output', 'Consumption', 'wages', 'Investment', 'Inflation', 'Credit spread', 'Bank leverage', 'Net worth of banks', 'Capital held by banks']

plt.rcParams['axes.prop_cycle'] = plt.cycler("color", plt.get_cmap('hot')(np.linspace(0.0,.5,2)))

vi = mod0.vix(v)
f = np.array((1,1,1,1,4,4,1,1,1))

# plot
figs, axs = figurator(3,3,1, figsize=(10,8))
pplot(f*np.array(ir00)[...,vi], labels=vp, sigma=.05, alpha=.2, ax=axs, colors='C0', legend='prior')
pplot(f*np.array(ir10)[...,vi][:,:], labels=vp, sigma=.05, ax=axs, styles='--', colors='C1', legend='posterior')
figs[0].tight_layout()
axs[6].legend()
[ f.savefig(output_path+mod0.name+'_irfs_prior_qek_'+str(n)+'_4pub.pdf') for n,f in enumerate(figs) ]

