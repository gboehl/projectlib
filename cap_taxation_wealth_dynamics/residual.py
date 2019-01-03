#!/bin/python2
# -*- coding: utf-8 -*-

from lib import *

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as ss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.diagnostic import acorr_ljungbox

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

## residual analyses
x   = estim(gamma0, std0, data, tax, share/100)
pe  = resid(x, (data, tax, share/100))[0]

spl_year    = 1945
pe0     = pe[years[1:] < spl_year]
pe1     = pe[years[1:] > spl_year]
# from pandas import Series
print('D-W test statistic:', durbin_watson(pe).round(4))
print('Ljung-Box test statistic:', acorr_ljungbox(pe, 1)[1])
print('Test on normality full sample:', ss.normaltest(pe)[1].round(4))
print('Test on normality pre 1945:', ss.normaltest(pe0)[1].round(4))
print('Test on normality post 1945:', ss.normaltest(pe1)[1].round(4))
print('Anderson-Darling full sample:', ss.anderson(pe))
print('Anderson-Darling pre 1945:', ss.anderson(pe0))
print('Anderson-Darling post 1945:', ss.anderson(pe1))

f, ax 	= plt.subplots()
plt.plot(years[1:][years[1:] < spl_year],pe[years[1:] < spl_year], 'o')
plt.plot(years[1:][years[1:] > spl_year],pe[years[1:] > spl_year], 'o')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(axis='both', which='both', top=False, right=False)
plt.show()


