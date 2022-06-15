#!/bin/python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
from pydsge import *
from grgrlib import *

from matplotlib import rc
rc('text', usetex=True)
rc('legend', fontsize=18)

dfile0 = '<path to bgs_final0_meta.npz>' # <- edit this! 
# the file can be downloaded from http://gregorboehl.com/data/bgs_final0_meta.npz 
# the files bgs_final0_res.npz and bgs_final0_sampler.h5 must also be downloaded and stored in the same folder

mod0 = DSGE.load(dfile0, force_parse=False)
mod0.load_estim()

epsd0 = mod0.load_rdict()

# historical decomposition
v0 = ['y','c','i','Pi','w','rn','excessreturnkb','cap'] # 2x4
v1 = ['y','c','i','Pi','w','r','excessreturnkb','cap'] # 2x4
vp = ['Output', 'Consumption', 'Investment', 'Inflation', 'Wages', '(Notational) Interest rate', 'Credit spread', 'Capital utilization']
sp = [ '$\epsilon_z$', '$\epsilon_u$', '$\epsilon_g$', '$\epsilon_i$', '$\epsilon_r$', '$\epsilon_p$', '$\epsilon_w$', '$\epsilon_{lk}$', '$\epsilon_{QEB}$', '$\epsilon_{QEK}$', '$\epsilon_{CBL}$']

# adjust quarterly measures
fac = np.array((1,1,1,4,1,4,4,1))

hdfull, means = mod0.nhd(epsd0)
hd = [h[v0] for h in hdfull]
hmin, hmax = sort_nhd(hd)
hmax = tuple(fac*h for h in hmax)
hmin = tuple(fac*h for h in hmin)

figs, axs = figurator(4,1,2, figsize=(10,9))
pplot(hmax, mod0.data.index, labels=vp, alpha=.5, ax=axs)
pplot(hmin, mod0.data.index, labels=vp, alpha=.5, ax=axs, legend=sp)
pplot(fac*means[v0], mod0.data.index, labels=vp, ax=axs, styles='--')
pplot(fac*means[v1], mod0.data.index, labels=vp, ax=axs, styles='-')

[f.tight_layout() for f in figs]
axs[2].legend(loc=3)
axs[5].legend(loc=3)

mod0.pool.terminate()
