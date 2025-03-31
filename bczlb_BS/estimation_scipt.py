#!/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import emcwrap as ew
from pydsge import *
from grgrlib import *

yaml = '<path_to_your_yaml>.yaml'

# load model
mod = DSGE.read(yaml)

# specify some meta data
mod.name = 'my_estimation'
mod.description = 'this is my estimation'
mod.path = '<path_where_you_want_to_store_results>'

# get estimation data
d0 = pd.read_csv('<path_to_your_csv>.csv', sep=';', index_col='date', parse_dates=True)

# adjust elb to model
zlb = mod.get_par('elb_level')
rate = d0['FFR']
d0['FFR'] = np.maximum(rate,zlb)

# load estimation data
mod.load_data(d0, start='1973Q1', end='2019Q4')
mod.save()

# initialize estimation fundamentals
mod.prep_estim(N=350, seed=0, verbose=True, use_prior_transform=True)

# set the data covariance matrix
mod.filter.R = mod.create_obs_cov(1e-1)
ind = mod.observables.index('FFR')
mod.filter.R[ind,ind] /= 1e1

# use DIME to sample from posterior
moves = ew.DIMEMove(aimh_prob=0.1)
nwalks = 200

# get initial sample
p0 = mod.prior_sampler(nwalks, check_likelihood=False, verbose=True)
p1 = mod.bptrans(p0, False)
mod.save()

# do sampling. This may take some time
mod.mcmc(p1, moves=moves, nsteps=2500, tune=500, update_freq=250)
mod.save()

# also extract the shock series for a sample from the posterior
pars = mod.get_par('posterior', nsamples=2000, full=True)
epsd0 = mod.extract(pars, nsamples=1, bound_sigma=6)
mod.save_rdict(epsd0)

# get some stats
mod.mcmc_summary()
