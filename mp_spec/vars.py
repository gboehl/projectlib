#!/bin/python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from scipy import interpolate
from base import *
inv = np.linalg.inv
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
import warnings
warnings.simplefilter("always")

## parameters:
g      	= 1.006
b      	= 1.229
std_p  	= .001 
std_y  	= .004

det 	= 100
batch_no 	= 100

x = np.linspace(-.11, .25, det)

pars        = pars_raw.copy()

sim 		= np.array(get_ts(pars, g = g, b = b, no_iters=159, transition = 100, std_p=std_p, std_y=std_y)).reshape(3,158)
moms_sim 	= np.hstack((np.std(sim, axis=1), np.corrcoef(sim).flat[[1,2,5]]))

SP 	= []
SY 	= []
SS 	= []
for i in range(batch_no):
	P, Y, S     = get_ts(pars, pval = x, g = g, b = b, no_iters=159, transition = 100, std_p = std_p, std_y = std_y, rnd_seed=i)
	SP.append(np.std(P,axis=1))
	SY.append(np.std(Y,axis=1))
	SS.append(np.std(S,axis=1))

SP 	= np.mean(SP, axis=0)
SY 	= np.mean(SY, axis=0)
SS 	= np.mean(SS, axis=0)

SP_min 	= x[np.argmin(SP)]
SY_min 	= x[np.argmin(SY)]

print('y_min =', SY_min)
print('pi_min =', SP_min)

## plotting
alp     = 0.7
txs     = 22
f, ax 	= plt.subplots(1, 1)
ax2 	= ax.twinx()
ax.plot([SP_min, SP_min], [min(SP),.05], ':', c='darkgreen', lw=3, alpha=alp)
ax.plot([SY_min, SY_min], [min(SY),.04], ':', c='b', lw=3, alpha=alp)
ax.annotate('$\pi_{min}$', xy=(SP_min-.01, .05), xycoords='data', fontsize=txs-2)
ax.annotate('$y_{min}$', xy=(SY_min-.01, .04), xycoords='data', fontsize=txs-2)
ax.plot(x, SP, color = 'darkgreen', linewidth=3, label='$\pi_t$')
ax.plot(x, SY, '--', color = 'b', linewidth=3, label='$y_t$')
ax2.plot(x, SS, '-.', color = 'maroon', linewidth=3, label='$s_t$')
ax.spines['top'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax.tick_params(axis='both', which='both', top='off', right='off', labelsize=txs)
ax2.tick_params(axis='both', which='both', top='off', right='off', labelsize=txs)
ax.set_xticks((-.1, 0, .1, .2))
ax2.set_ylabel('Stock prices', fontsize=txs-2)
ax.set_ylabel('Inflation \& Output', fontsize=txs-2)
ax.set_xlabel('Asset price targeting $\phi_s$', fontsize=txs-2)
h1, l1 = ax.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1+h2, l1+l2, frameon=False, fontsize=txs-2)
ax.set_xticks([0], minor=True)
ax.xaxis.grid(True, which='minor')
plt.tight_layout()
plt.savefig('/home/gboehl/rsh/dfi/var.pdf', dpi=500)
plt.show()

