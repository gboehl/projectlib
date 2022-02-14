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
mod0.data.keys()

pars0 = epsd0['pars']
pars1 = pars0
pars2 = pars0

s = mod0.data.index.slice_indexer('2004Q4','2020')

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

oim0 = mod0.obs(sim0, pars=pars0)
oim1 = mod0.obs(sim1, pars=pars0)
oim2 = mod0.obs(sim2, pars=pars0)
oim3 = mod0.obs(sim3, pars=pars1)
oim4 = mod0.obs(sim4, pars=pars2)

## MP
v0 = ['y']
v1 = ['y','Pi','r']
o = ['Infl','FFR']
vp0 = 'Output',
vp1 = 'Inflation', 'Interest rate'

fac = np.array((1,4,1))
vmean = sim0.mean(axis=0)[...,mod0.vix(v0)]
omean = oim0.mean(axis=0)[...,mod0.oix(o)]
y1 = (sim1-sim0)[:,s,mod0.vix(v0)]+vmean[s]

figs, axs = figurator(3,2,1, figsize=(10,8))
ax0y = axs[0]
ax0o = axs[[2,4]]
ax1 = axs[1::2]
[fig.autofmt_xdate() for fig in figs]
pplot(y1, mod0.data.index[s], ax=ax0y, alpha=.05, colors='C0', bulk_plot=True)
pplot(vmean[s], mod0.data.index[s], labels=vp0, ax=ax0y, alpha=.9, colors='maroon', styles='--')
pplot(4*oim1[:,s,mod0.oix(o)], mod0.data.index[s], ax=ax0o, alpha=.05, colors='C0', bulk_plot=True)
pplot(4*omean[s], mod0.data.index[s], labels=vp1, ax=ax0o, alpha=.9, colors='maroon', styles='--')
pplot(fac*(sim0-sim1)[:,s,mod0.vix(v1)], mod0.data.index[s], labels=vp0+vp1, ax=ax1, sigma=.05, alpha=.3, colors='C0', legend='pre-2009 rule')
[ f.tight_layout() for f in figs ]

## QE
v0 = ['y', 'c', 'i', 'excess_return_b']
v1 = ['y', 'c', 'i', 'excess_return_b', 'Pi','excessreturnkb']
vp0 = ['Output', 'Consumption', 'Investment', 'Term premium']
o = ['Infl', 'GZSpread']
vp1 = ['Inflation', 'Credit spread']

# orig
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

## durations
s = mod0.data.index.slice_indexer('2008Q1','2016Q4')

i0 = mod0.data.index.get_loc('2009Q1')
i1 = mod0.data.index.get_loc('2011Q1')
i2 = mod0.data.index.get_loc('2012Q1')
i3 = mod0.data.index.get_loc('2013Q1')

sigma = .1
interval = np.nanpercentile(k0, [sigma*100/2, (1 - sigma/2)*100], axis=0)
median = np.nanmedian(k0, axis=0)

fig = plt.figure(figsize=(10,8))
ax0 = plt.subplot(311)
ax1 = plt.subplot(323)
ax2 = plt.subplot(324)
ax3 = plt.subplot(325)
ax4 = plt.subplot(326)
axformater(ax0)

ax0.bar(mod0.data.index[s], interval[0][s], width=70, alpha=0.5, color='C2')
ax0.bar(mod0.data.index[s], interval[1][s], width=40, alpha=0.5, color='C2')
ax0.bar(mod0.data.index[s], median[s], width=70, alpha=0.5, color='C0')
ax0.set_title('Expected durations')
ax1.hist(k0[:,i0], np.arange(2,12).astype(int), alpha=.5, density=True)
ax1.set_title('distribution in 2009:Q1')
ax2.hist(k0[:,i1], np.arange(2,12).astype(int), alpha=.5, density=True)
ax2.set_title('distribution in 2011:Q1')
ax3.hist(k0[:,i2], np.arange(2,12).astype(int), alpha=.5, density=True)
ax3.set_title('distribution in 2012:Q1')
ax4.hist(k0[:,i2], np.arange(2,12).astype(int), alpha=.5, density=True)
ax4.set_title('distribution in 2013:Q1')

fig.tight_layout()

