#!/bin/python# -*- coding: utf-8 -*-

import os
import emcee
import numpy as np
import pandas as pd
from pydsge import *
from grgrlib import *
from os.path import join

from grgrlib.legacy import tmcmc
DSGE.tmcmc = tmcmc

# set all paths in this script relative to script location
pth = '/home/gboehl/repos/projectlib/qe_bgs/'
yaml_path = join(pth, join('code', 'bgs_rnk_exo.yaml'))
data_path = join(pth, join('data', 'BGS_est_data.csv'))
output_path = join(pth, 'output')

# parse and load model
mod = DSGE.read(yaml_path)  

# provide some description
mod.name = 'bgs'
mod.description = 'BGS baseline model with exogenous QE'
mod.path = output_path

# parse data
d0 = pd.read_csv(data_path, sep=';', index_col='date', parse_dates=True).dropna() 

# adjust elb in data
zlb = mod.get_par('elb_level')
rate = d0['FFR']
d0['FFR'] = np.maximum(rate,zlb)

# load data into model
mod.load_data(d0, start='1998Q1')

# prepare estimation routines
mod.prep_estim(N=350, seed=0, verbose=True)

# set measurement errors
mod.filter.R = mod.create_obs_cov(1e-1)
ind = mod.observables.index('FFR')
mod.filter.R[ind,ind] /= 1e1
ind = mod.observables.index('CB_Bonds_10Y')
mod.filter.R[ind,ind] /= 1e1
ind = mod.observables.index('CB_Loans')
mod.filter.R[ind,ind] /= 1e1
ind = mod.observables.index('CBL')
mod.filter.R[ind,ind] /= 1e1

# set target for tempering
fmax = -500

# set ensemble MCMC moves
moves = [(emcee.moves.DEMove(), 0.8), 
         (emcee.moves.DESnookerMove(), 0.2),]

# get a sample of the prior and do tempering 
p0 = mod.tmcmc(199, 200, 11, fmax, moves=moves, update_freq=100)
# save results to disk
mod.save()

# continue sampling without tempering steps
mod.mcmc(p0, moves=moves, nsteps=2500, tune=500, update_freq=250, lprob_seed='set', append=True)
# save results to disk
mod.save()

# get a sample from the posterior distribution, extract shocks and save them to disk
pars = mod.get_par('posterior', nsamples=2000, full=True)
epsd0 = mod.extract(pars, nsamples=1, bound_sigma=6)
mod.save_rdict(epsd0)

# provide a summary of the posterior distribution
mod.mcmc_summary()
