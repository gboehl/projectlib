#!/bin/python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
from pydsge import *
from grgrlib import *
from matplotlib import use
# use('Agg')

from matplotlib import rc
rc('text', usetex=True)
rc('legend', fontsize=18)

dfile0 = '<path to bgs_final0_meta.npz>' # <- edit this! 
# the file can be downloaded from http://gregorboehl.com/data/bgs_final0_meta.npz 
# the files bgs_final0_res.npz and bgs_final0_sampler.h5 must also be downloaded and stored in the same folder

mod0 = DSGE.load(dfile0, force_parse=False)
mod0.load_estim()

N0 = 2000
N1 = 2000

mod0.prior['rootk1'][-2:] = [.9, .05]
mod0.prior['rootb1'][-2:] = [.9, .05]
mod0.prior['rootk2'][-2:] = [.9, .05]
mod0.prior['rootb2'][-2:] = [.9, .05]
mod0.prior['rho_cbl'][-2:] = [.75,.05]

pars0 = mod0.get_par('prior', nsamples=N0, verbose=True)
pars1 = mod0.get_par('posterior', nsamples=N1, verbose=True)

def pfunc(par):
    r1 = mod0.get_par('rootk1', par)
    r2 = mod0.get_par('rootk2', par)

    smax = (np.log(np.log(r2)/np.log(r1)))/np.log(r1/r2)
    xmax = (r1**smax - r2**smax)/(r1 - r2)
    sig = mod0.data['CB_Loans'].max()/xmax

    return mod0.irfs(('e_qe_k',sig,0), par, T=40, linear=True)[0]

ir00 = [pfunc(p).to_numpy() for p in pars0]
print('ir00 done')
ir10 = [pfunc(p).to_numpy() for p in pars1]
print('ir10 done')

def pfunc(par):
    r1 = mod0.get_par('rootb1', par)
    r2 = mod0.get_par('rootb2', par)

    smax = (np.log(np.log(r2)/np.log(r1)))/np.log(r1/r2)
    xmax = (r1**smax - r2**smax)/(r1 - r2)
    sig = mod0.data['CB_Bonds_10Y'].max()/xmax

    return mod0.irfs(('e_qe_b',sig,0), par, T=40, linear=True)[0]

ir01 = [pfunc(p).to_numpy() for p in pars0]
print('ir01 done')
ir11 = [pfunc(p).to_numpy() for p in pars1]
print('ir11 done')

sig = mod0.data['CBL'].max()
ir02 = mod0.irfs(('e_cbl',sig,0), pars0, T=40, linear=True)[0]
print('ir02 done')
ir12 = mod0.irfs(('e_cbl',sig,0), pars1, T=40, linear=True)[0]
print('ir12 done')

ir03 = mod0.irfs(('e_r',-0.25/4), pars0, T=40, linear=True)[0]
print('ir03 done')
ir13 = mod0.irfs(('e_r',-0.25/4), pars1, T=40, linear=True)[0]
print('ir13 done')


sigma = .05
v = ['y','c','w','i','Pi','excessreturnkb','lev','n','kb'] # 3x3
vp = ['Output', 'Consumption', 'wages', 'Investment', 'Inflation', 'Credit spread', 'Bank leverage', 'Net worth of banks', 'Capital held by banks']

vi = mod0.vix(v)
f = np.array((1,1,1,1,4,4,1,1,1))

figs, axs = figurator(3,3,1, figsize=(10,8))
pplot(f*np.array(ir10)[...,vi], labels=vp, sigma=.05, ax=axs, colors='C1', legend='posterior')
figs[0].tight_layout()

figs, axs = figurator(3,3,1, figsize=(10,8))
pplot(f*np.array(ir00)[...,vi], labels=vp, sigma=.05, alpha=.2, ax=axs, colors='C0', legend='prior')
pplot(f*np.array(ir10)[...,vi][:,:], labels=vp, sigma=.05, ax=axs, colors='C1', legend='posterior')
figs[0].tight_layout()
axs[6].legend()

figs, axs = figurator(3,3,1, figsize=(10,8))
pplot(f*np.array(ir11)[...,vi][:,:], labels=vp, sigma=.05, ax=axs, colors='C1')
figs[0].tight_layout()

figs, axs = figurator(3,3,1, figsize=(10,8))
pplot(f*np.array(ir01)[...,vi], labels=vp, sigma=.05, alpha=.2, ax=axs, colors='C0', legend='prior')
pplot(f*np.array(ir11)[...,vi][:,:], labels=vp, sigma=.05, ax=axs, colors='C1', legend='posterior')
figs[0].tight_layout()
axs[6].legend()

figs, axs = figurator(3,3,1, figsize=(10,8))
pplot(f*np.array(ir12)[...,vi][:,:], labels=vp, sigma=.05, ax=axs, colors='C1')
figs[0].tight_layout()

figs, axs = figurator(3,3,1, figsize=(10,8))
pplot(f*np.array(ir02)[...,vi], labels=vp, sigma=.05, alpha=.2, ax=axs, colors='C0', legend='prior')
pplot(f*np.array(ir12)[...,vi][:,:], labels=vp, sigma=.05, ax=axs, colors='C1', legend='posterior')
axs[6].legend()
figs[0].tight_layout()

figs, axs = figurator(3,3,1, figsize=(10,8))
pplot(f*np.array(ir13)[...,vi][:,:], labels=vp, sigma=.05, ax=axs, colors='C1')
figs[0].tight_layout()

figs, axs = figurator(3,3,1, figsize=(10,8))
pplot(f*np.array(ir03)[...,vi], labels=vp, sigma=.05, alpha=.2, ax=axs, colors='C0', legend='prior')
pplot(f*np.array(ir13)[...,vi][:,:], labels=vp, sigma=.05, ax=axs, colors='C1', legend='posterior')
axs[6].legend()
figs[0].tight_layout()

mod0.pool.terminate()
