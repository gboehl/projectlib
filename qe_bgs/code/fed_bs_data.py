#!/bin/python
# -*- coding: utf-8 -*-

import os
import matplotlib.pyplot as plt
import numpy as np
from pydsge import *
from grgrlib import *

plt.rc('text', usetex=True)
plt.rc('legend', fontsize=18)

# set all paths relative to script location
pth = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(os.path.split(pth)[0],'data','BGS_est_data.csv')
output_path = os.path.join(os.path.split(pth)[0],'output')

#load data
d0 = pd.read_csv(data_path, sep=',', index_col='date', parse_dates=True).dropna() 
cbl = d0['CBL']
qeb = d0['CB_Bonds_10Y'] + 5
qek = d0['CB_Loans']
s = d0.index.slice_indexer('2004Q4','2020')

# use a cycler that has nice black/white contrasts
plt.rcParams['axes.prop_cycle'] = plt.cycler("color", plt.get_cmap('hot')(np.linspace(0.0,.6,3)))

# plot...
figs, axs = figurator(1,1,figsize=(10,6))
axs[0].fill_between(d0.index[s], qeb[s], (qeb+cbl)[s], hatch='X', alpha=.45, color='C0', lw=0, label='Liquidity injection')
axs[0].fill_between(d0.index[s], 0, qeb[s], hatch='\\', alpha=.45, color='C1', lw=0, label='Gov. bonds')
axs[0].fill_between(d0.index[s], (qeb+cbl)[s], (qeb+cbl+qek)[s], hatch='o', alpha=.45, color='C2', lw=0, label='Private securities')
axs[0].legend(loc='upper left')
figs[0].tight_layout()

# save plot
figs[0].savefig(os.path.join(output_path, 'fig1'))

