#!/bin/python2
# -*- coding: utf-8 -*-

from lib import *

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as ss

year 		= 1922

tax     = np.load('datac.npz')['tax_data']
pps     = np.load('datac.npz')['iiq_data']
years   = np.load('datac.npz')['years']

gamma0 	= .35
std0 	= .01
share 	= 1

if share == 1:
    data 	= pps[:,0]
if share == .1:
    data 	= pps[:,1]
if share == .01:
    data 	= pps[:,2]

## tax projections
horizon     = 50
forerun     = 40
tax1    = tax[-1]
tax2    = .23
tax3    = .28

arg     = 1945 - year
x                       = estim(gamma0, std0, data[arg:], tax[arg:], share/100)
med, iv0, iv1, _        = simul(x, (data[-1], np.repeat(tax1, horizon), share/100))
med_v2, iv0_v2, iv1_v2, _   = simul(x, (data[-1], np.repeat(tax2, horizon), share/100))
med_v3, iv0_v3, iv1_v3, _   = simul(x, (data[-1], np.repeat(tax3, horizon), share/100))

xx_fc    = np.arange(years[-1] - forerun, years[-1] + horizon)

f, ax 	= plt.subplots()
ax.plot(xx_fc[:forerun], data[-forerun:], 'o', color='C0', lw=2, label='data {}\%'.format(share))
ax.plot(xx_fc[forerun:],med, color='C0', lw=2, label='{}\% tax rate'.format(int(tax1*100)))
ax.plot(xx_fc[forerun:],med_v2, color='C1', lw=2, label='{}\% tax rate'.format(int(tax2*100)))
ax.plot(xx_fc[forerun:],med_v3, color='C2', lw=2, label='{}\% tax rate'.format(int(tax3*100)))
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(axis='both', which='both', top=False, right=False, labelsize=16)
ax.legend(frameon=False)
plt.tight_layout()
plt.show()
