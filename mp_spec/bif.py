#!/bin/python2
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from tqdm import tqdm
from base import *
from scipy import interpolate
import warnings
warnings.simplefilter("always")
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
pars            = pars_raw

## chose parameter for bifurcation diagram
bif_type 	= 0 	# alpha
bif_type 	= 1 	# phi_s
# bif_type 	= 2 	# gamma

aufl 	= 1000

if bif_type == 1: 		# phi_s
	st 		= -.21
	ending  = .08
	x 		= np.linspace(st,ending,aufl)
	pval    = x
	b       = 1.229
	g     	= 1.006
elif bif_type == 2: 	# gamma
	st 		= 0.95
	ending 	= 1.4
	x 		= np.linspace(st,ending,aufl)
	pval    = 0
	g     	= x
	b       = 1.229
else:					# alpha
	st 		= 1.18 
	ending 	= 1.47 	
	x 		= np.linspace(st,ending,aufl)
	pval    = 0
	b     	= x
	g     	= 1.006

P, Y, S     = get_ts(pars, pval = pval, g = g, b = b, no_iters=1000, transition = 10000)

if bif_type == 1:
	X2 = []
	for i in tqdm(x):
		pars = pars_raw.copy()
		pars['a_s'] = i
		Psi_inv = get_psi_inv(pars)
		States = np.array([[0],[0],[pars['beta']]])
		X       = Psi_inv.dot(States)
		X2.append(X)
	X2  = np.array(X2)
	P2_zero = interpolate.sproot(interpolate.splrep(x, X2[:,0], s=0))
	Y2_zero = interpolate.sproot(interpolate.splrep(x, X2[:,1], s=0))
	S2_zero = interpolate.sproot(interpolate.splrep(x, X2[:,2]-1, s=0))

S2_zero 	= 0.019

alp     = 0.7
txs     = 22
f, ax  	= plt.subplots(1, 1)
if bif_type == 1:
	if np.size(P2_zero) == 1: 
		ax.plot([P2_zero, P2_zero], [0.1,0], ':', c='darkgreen', lw=3, alpha=alp)
		ax.plot([P2_zero, P2_zero], [0,0], 'o', c='darkgreen', lw=3, alpha=alp)
		ax.annotate('$I_{\pi}$', xy=(P2_zero-.01, 0.1), xycoords='data', fontsize=txs-2)
	if np.size(Y2_zero) == 1:
		ax.plot([Y2_zero, Y2_zero], [0.05,0], ':', c='b', lw=3, alpha=alp)
		ax.plot([Y2_zero, Y2_zero], [0,0], 'o', c='b', lw=3, alpha=alp)
		ax.annotate('$I_{y}$', xy=(Y2_zero-.01, 0.05), xycoords='data', fontsize=txs-2)
	if np.size(S2_zero) == 1:
		ax.plot([S2_zero, S2_zero], [0.12,0], ':', c='maroon', lw=3, alpha=alp)
		ax.plot([S2_zero, S2_zero], [0,0], 'o', c='maroon', lw=3, alpha=alp)
		ax.annotate('$B(\lambda_{s})$', xy=(S2_zero-.01, .12), xycoords='data', fontsize=txs-2)
	ax.plot(x, np.max(S,axis=1)/5, '--', color = 'maroon', lw=3, alpha=.4)
	ax.plot(x, np.min(S,axis=1)/5, '--', color = 'maroon', lw=3, alpha=.4)
elif not bif_type:
	ax.plot([1.226, 1.224],[-.28,.28], color ='gray', linewidth=1.5, linestyle="--", alpha=.5)
	ax.plot([1.357, 1.357],[-.28,.28], color ='gray', linewidth=1.5, linestyle="--", alpha=.5)
	ax.plot([1.368, 1.368],[-.28,.28], color ='gray', linewidth=1.5, linestyle="--", alpha=.5)
	ax.plot([1.402, 1.402],[-.28,.28], color ='gray', linewidth=1.5, linestyle="--", alpha=.5)
ax.plot(x,Y, '.', color = 'b',markersize = 0.01)
ax.plot(x,P, '.', color = 'darkgreen',markersize = 0.01)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(axis='both', which='both', top='off', right='off', labelsize=txs)
# ax.set_xlabel(r'Intensity of Choice $\delta$', size=txs)
if bif_type == 1: 	ax.set_xlabel('Asset Price Targeting $\phi_s$', size=txs)
elif not bif_type:  ax.set_xlabel(r'Behavioral Bias $\alpha$', size=txs)
else: 				ax.set_xlabel(r'Degree of Trend Extrapolation $\gamma$', size=txs)
ax.set_ylabel(r'\% deviation', size=txs+2)
plt.tight_layout()
if bif_type == 1: 	
	text1 = ax.annotate(r'Stock prices $s$ ($\times 0.2$)', xy=(-.07,-0.14), xytext=(0, 0),
					   textcoords='offset points', alpha=.4,
					   size=16, color='maroon',
					   horizontalalignment='left',
					   verticalalignment='bottom')
	text1.set_rotation(35)
	text2 = ax.annotate(r'Inflation $\pi$', xy=(-.15,0), xytext=(0, 0),
					   textcoords='offset points', alpha=.4,
					   size=16, color='darkgreen',
					   horizontalalignment='left',
					   verticalalignment='bottom')
	text2.set_rotation(-35)
	text3 = ax.annotate(r'Output $y$', xy=(-.13,0.05), xytext=(0, 0),
					   textcoords='offset points', alpha=.4,
					   size=16, color='blue',
					   horizontalalignment='left',
					   verticalalignment='bottom')
	text3.set_rotation(-35)
elif not bif_type:
	text = ax.annotate('explosive dynamics', xy=(1.404,-0.07), xytext=(0, 0),
					   textcoords='offset points',
						alpha=.6,
					   size=13, color='maroon',
					   horizontalalignment='left',
					   verticalalignment='bottom')
	text.set_rotation(29)
	plt.xlim([1.19,1.49])
else:
	text = ax.annotate('explosive dynamics', xy=(1.23,-0.03), xytext=(0, 0),
					   textcoords='offset points',
						alpha=.6,
					   size=13, color='maroon',
					   horizontalalignment='left',
					   verticalalignment='bottom')
	text.set_rotation(29)
	plt.xlim([min(x),1.34])
ax.set_xticks([0], minor=True)
ax.xaxis.grid(True, which='minor')
plt.tight_layout()
plt.show()

