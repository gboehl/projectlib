#!/bin/python# -*- coding: utf-8 -*-

import emcee
import numpy as np
import pandas as pd
from pydsge import *
from grgrlib import *

yaml = '<path to bgs_rnk_exo.yaml>' # <- edit this! The yaml is in the yaml folder of the projectlib: https://github.com/gboehl/projectlib/blob/master/yamls/bgs_rnk_exo.yaml

mod = DSGE.read(yaml)  

mod.name = 'bgs_replicate'
mod.description = 'Replication of BGS'
mod.path = '<path where you want to store the results>' # <- edit this! Its your choice...

# load data and set-up model
d0 = pd.read_csv('<path to bgs_data.csv>') # <- edit this! The data can be downloaded from http://gregorboehl.com/data/bgs_data.csv
mod.load_data(d0)
mod.prep_estim(N=350, seed=0, verbose=True, eval_priors=True)

# specify observation noice covariance
mod.filter.R = mod.create_obs_cov(1e-1)
ind = mod.observables.index('FFR')
mod.filter.R[ind,ind] /= 1e1
ind = mod.observables.index('CB_Bonds_10Y')
mod.filter.R[ind,ind] /= 1e1
ind = mod.observables.index('CB_Loans')
mod.filter.R[ind,ind] /= 1e1
ind = mod.observables.index('CBL')
mod.filter.R[ind,ind] /= 1e1


moves = [(emcee.moves.DEMove(), 0.8), 
         (emcee.moves.DESnookerMove(), 0.2),]

fmax = -500

# do some tempered runs as burn-in
p0 = mod.tmcmc(199, 200, 11, fmax, moves=moves, update_freq=100, lprob_seed='set')
mod.save()

# run the mcmc
mod.mcmc(p0, moves=moves, nsteps=2500, tune=500, update_freq=250, lprob_seed='set', append=True)
mod.save()

# obtain 2000 sets of extracted shocks
pars = mod.get_par('posterior', nsamples=2000, full=True)
epsd0 = mod.extract(pars, nsamples=1, bound_sigma=6)
mod.save_rdict(epsd0)

mod.mcmc_summary()
