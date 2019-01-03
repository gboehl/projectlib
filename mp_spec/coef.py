#!/bin/python2
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from scipy import interpolate
from base import *
inv = np.linalg.inv
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

pars = pars_raw.copy()
eigs = 0

ptype = 'a_s'

print("nu: ", pars_raw['nut']/(1- pars_raw['nut']))

det = 500

if ptype == 'nut': 	x 	= np.linspace(-1, .9, det)
if ptype == 'a_p': 	x 	= np.linspace(.95, 1.8, det)
if ptype == 'a_s': 	x 	= np.linspace(-0.25, 0.35, det) 	
if eigs: 	x 	= np.linspace(-1, 0.9, det) 	

X 	= []
for i in tqdm(x):
	pars[ptype] = i
	if eigs: 	
		val       	= np.sort(np.abs(get_psi_inv(pars, return_eigvs=1)))[2:]
	else:
		Psi_inv 	= get_psi_inv(pars)
		States 		= np.array([[0],[0],[pars['beta']]])
		val       	= Psi_inv.dot(States)
	X.append(val)
X 	= np.array(X)

if eigs: print(x[np.where(np.min(X, axis=1) < 1)])

P_max 		= max(x[np.where(X[:,0] > 0)[0]])
Y_max 		= max(x[np.where(X[:,1] > 0)[0]])
S_max 		= max(x[np.where(X[:,2] > 1)[0]])

print('$I_\pi =$', P_max)
print('$I_y =$', Y_max)
print('$\lambda_s =$', S_max)

txs=22
alp     = 0.7
f, ax = plt.subplots(1, 1)
ax.plot([P_max, P_max], [0.3,0], ':', c='darkgreen', lw=3, alpha=alp)
ax.plot([Y_max, Y_max], [0.5,0], ':', c='b', lw=3, alpha=alp)
ax.plot([S_max, S_max], [1.2,1], ':', c='maroon', lw=3, alpha=alp)
ax.plot([P_max, P_max], [0,0], 'o', c='darkgreen', lw=3, alpha=alp)
ax.plot([Y_max, Y_max], [0,0], 'o', c='b', lw=3, alpha=alp)
ax.plot([S_max, S_max], [1,1], 'o', c='maroon', lw=3, alpha=alp)
ax.annotate('$I_{\pi}$', xy=(P_max-.01, 0.3), xycoords='data', fontsize=txs-2)
ax.annotate('$I_{y}$', xy=(Y_max-.01, 0.5), xycoords='data', fontsize=txs-2)
ax.annotate('$\lambda_{s}$', xy=(S_max-.01, 1.2), xycoords='data', fontsize=txs-2)
ax.plot(x,X[:,0], linewidth=3, color='darkgreen', label='$\pi_t$')
ax.plot(x,X[:,1], '--', linewidth=3, color='b', label='$y_t$')
ax.plot(x,X[:,2], '-.', linewidth=3, color='maroon', label='$s_t$')
ax.tick_params(axis='both', which='both', top='off', right='off', labelsize=txs)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_ylabel('Feedback to stock prices', size=txs)
ax.set_xlabel('${\phi_s}$ Asset Price Targeting', size=txs+2)
ax.legend(frameon=False, fontsize=txs-2)
ax.set_yticks([0, 1], minor=True)
ax.yaxis.grid(True, which='minor')
ax.set_xticks([0], minor=True)
ax.xaxis.grid(True, which='minor')
plt.tight_layout()
plt.show()

