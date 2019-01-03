#!/bin/python2
# -*- coding: utf-8 -*-

from lib import *

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

year 		= 1922

tax     = np.load('datac.npz')['tax_data']
pps     = np.load('datac.npz')['iiq_data']
years      = np.load('datac.npz')['years']

gamma0 	= .35
std0 	= .01
share 	= 1

year_oos    = year
year_oos    = 1940
# year_oos    = 1975
# year_oos    = 1990

if share == 1:
    data 	= pps[:,0]
if share == .1:
    data 	= pps[:,1]
if share == .01:
    data 	= pps[:,2]

oos_arg     = year_oos - year
x_oos   = estim(gamma0, std0, data[:oos_arg], tax[:oos_arg], share/100)
med, iv0, iv1, stat     = simul(x_oos, (data[oos_arg], tax[oos_arg:], share/100))

f, ax 	= plt.subplots()
ax.plot(years,data, 'o', color='C0', lw=2, label='data {}\%'.format(share))
ax.plot(years[oos_arg:], med, color='C1', lw=2, label='median {}\%'.format(share))
ax.plot(years[oos_arg:], stat, '--', color='C5', lw=2, label='stationary {}\%'.format(share), alpha=.7)
ax.fill_between(years[oos_arg:], iv0, iv1, lw=0, alpha=.3, color='C1')
ax.plot(years,tax, ':', color='C7', lw=2, label='tax')
ax.tick_params(axis='both', which='both', top=False, right=False)
ax.legend(frameon=False)
plt.tight_layout()
plt.show()
