#!/bin/python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
from pydsge import *
from grgrlib import *

from matplotlib import rc
rc('text', usetex=True)
rc('legend', fontsize=18)

# set all paths relative to script location
pth = os.path.dirname(os.path.abspath(__file__))
output_path = os.path.join(os.path.split(pth)[0],'output')
dfile0 = os.path.join(output_path, 'bgs_final0_meta.npz')

# load estimated model from metadata
mod0 = DSGE.load(dfile0, force_parse=False)
mod0.load_estim()

# load stored shocks
epsd0 = mod0.load_rdict()

# set parameters for all simulations
pars0 = epsd0['pars']
pars1 = pars0
pars2 = pars0

s = mod0.data.index.slice_indexer('2004Q4','2020')

# set a color cycler that has nice black/white contrasts
plt.rcParams['axes.prop_cycle'] = plt.cycler("color", plt.get_cmap('hot')(np.linspace(0.0,.6,3)))

# swich off some shocks and simulate 
mod0.shocks
msk = mod0.mask
sim0, (l0,k0), _ = mod0.simulate(epsd0, verbose=True)
msk['e_r']['2007Q4':] = 0
sim1, (l1,k1), _ = mod0.simulate(epsd0, msk, verbose=True)
msk = mod0.mask
msk['e_cbl'][:] = 0
sim2, (l2,k2), _ = mod0.simulate(epsd0, msk, verbose=True)
msk['e_qe_b'][:] = 0
sim3, (l3,k3), _ = mod0.simulate(source=epsd0, mask=msk, pars=pars1, verbose=True)
msk['e_qe_k'][:] = 0
sim4, (l4,k4), _ = mod0.simulate(source=epsd0, mask=msk, pars=pars2, verbose=True)

# get observables based on simulations
oim0 = mod0.obs(sim0, pars=pars0)
oim1 = mod0.obs(sim1, pars=pars0)
oim2 = mod0.obs(sim2, pars=pars0)
oim3 = mod0.obs(sim3, pars=pars1)
oim4 = mod0.obs(sim4, pars=pars2)

# let software know what we are interested in
v0 = ['y', 'c', 'i', 'excess_return_b']
v1 = ['y', 'c', 'i', 'excess_return_b', 'Pi','excessreturnkb']
vp0 = ['Output', 'Consumption', 'Investment', 'Term premium']
o = ['Infl', 'GZSpread']
vp1 = ['Inflation', 'Credit spread']

fac = np.array([1,1,1,4,4,4])

vmean = sim0.mean(axis=0)[...,mod0.vix(v0)]
omean = oim0.mean(axis=0)[...,mod0.oix(o)]
y2 = (sim2-sim0)[:,s,mod0.vix(v0)]+vmean[s]
y3 = (sim3-sim0)[:,s,mod0.vix(v0)]+vmean[s]
y4 = (sim4-sim0)[:,s,mod0.vix(v0)]+vmean[s]

# provide some information
print('mean max net cum CBL:', list(zip(v1, fac*np.max(np.mean(sim0-sim2,0),0)[mod0.vix(v1)])))
print('mean max net cum QEB:', list(zip(v1, fac*np.max(np.mean(sim2-sim3,0),0)[mod0.vix(v1)])))
print('mean max net cum QEK:', list(zip(v1, fac*np.max(np.mean(sim3-sim4,0),0)[mod0.vix(v1)])))
print('mean max net cum all:', list(zip(v1, fac*np.max(np.mean(sim0-sim4,0),0)[mod0.vix(v1)])))

print('mean min net cum CBL:', list(zip(v1, fac*np.min(np.mean(sim0-sim2,0),0)[mod0.vix(v1)])))
print('mean min net cum QEB:', list(zip(v1, fac*np.min(np.mean(sim2-sim3,0),0)[mod0.vix(v1)])))
print('mean min net cum QEK:', list(zip(v1, fac*np.min(np.mean(sim3-sim4,0),0)[mod0.vix(v1)])))
print('mean min net cum all:', list(zip(v1, fac*np.min(np.mean(sim0-sim4,0),0)[mod0.vix(v1)])))

# plotting ...
figs, axs = figurator(3,2,2,figsize=(10,7))
ax0y = axs[[0,2,4,6]]
ax0o = axs[[8,10]]
ax1 = axs[1::2]
[fig.autofmt_xdate() for fig in figs]
pplot(y2, mod0.data.index[s], ax=ax0y, alpha=.01, bulk_plot=True, colors='C0')
pplot(y3, mod0.data.index[s], ax=ax0y, alpha=.01, bulk_plot=True, colors='C1')
pplot(y4, mod0.data.index[s], ax=ax0y, alpha=.01, bulk_plot=True, colors='C2')
pplot(vmean[s], mod0.data.index[s], labels=vp0, ax=ax0y, alpha=.9, colors='maroon', styles='--')
pplot(4*oim2[:,s,mod0.oix(o)], mod0.data.index[s], ax=ax0o, alpha=.01, bulk_plot=True, colors='C0')
pplot(4*oim3[:,s,mod0.oix(o)], mod0.data.index[s], ax=ax0o, alpha=.01, bulk_plot=True, colors='C1')
pplot(4*oim4[:,s,mod0.oix(o)], mod0.data.index[s], ax=ax0o, alpha=.01, bulk_plot=True, colors='C2')
pplot(4*omean[s], mod0.data.index[s], labels=vp1, ax=ax0o, alpha=.9, colors='maroon', styles='--')
_, _, p0 = pplot(fac*np.mean(sim0-sim2,0)[s,mod0.vix(v1)], mod0.data.index[s], labels=vp0+vp1, ax=ax1, sigma=.05, alpha=.99, colors='C0', styles='--')
_, _, p1 = pplot(fac*np.mean(sim0-sim3,0)[s,mod0.vix(v1)], mod0.data.index[s], labels=vp0+vp1, ax=ax1, sigma=.05, alpha=.99, colors='C1', styles='-.')
_, _, p2 = pplot(fac*(sim0-sim4)[:,s,mod0.vix(v1)], mod0.data.index[s], labels=vp0+vp1, ax=ax1, sigma=.05, alpha=.3, colors='C2')
ps = [ p0[0][0][0], p1[0][0][0], p2[0][0][0] ]
axs[4].legend(ps,['CBL', '+ QEB', '+ QEK'], framealpha=.7)
axs[10].legend(ps,['CBL', '+ QEB', '+ QEK'], framealpha=.7)
[ f.tight_layout() for f in figs ]
[ f.savefig(output_path+mod0.name+'_cfs_qe'+str(n)+'.pdf') for n,f in enumerate(figs) ]


