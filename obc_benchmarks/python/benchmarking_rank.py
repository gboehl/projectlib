#!/bin/python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import tqdm
from pydsge import *
from grgrlib import *
from timeit import default_timer as timer
from pydsge.engine import *

yamlf = '/home/gboehl/repos/projectlib/obc_benchmarks/python/rank.yaml'

mod = DSGE.read(yamlf, verbose=True)
p = mod.set_par('calib', k_max = 30)

np.random.seed(0)
nsamples = int(1e6)

cov = np.diag([10]*len(mod.svv))
X = np.random.multivariate_normal(mean=np.zeros(len(mod.svv)), cov=cov, size=nsamples)

res = np.empty((nsamples, 4))


omg, lam, x_bar = mod.sys
pmat, qmat, pterm, qterm, bmat, bterm = mod.precalc_mat

dimp, dimq = omg.shape
dimeps = mod.neps

shocks = np.zeros(dimeps)

set_k, set_l = -1, -1

for i,x in enumerate(tqdm.tqdm(X)):
    st = timer()
    pobs, q, l, k, flag = t_func_jit(pmat, pterm, qmat[:, :, :-dimeps], qterm[..., :-dimeps], bmat, bterm, x_bar, *mod.hx, x[-dimq+dimeps:], shocks, set_l, set_k, False)
    lapsed = timer() - st

    res[i,:] = l, k, flag, lapsed


columns = ['l', 'k', 'flag', 'time']
df = pd.DataFrame(res, columns=columns)

out = df['flag'] > 0

table = pd.DataFrame(columns=['mean', 'std', '\% of samples'])

samples = ()
samples += (df['k'] == 0) & ~out,
samples += (df['k'] > 0) & (df['k'] <= 5) & ~out,
samples += (df['k'] > 5) & (df['k'] <= 10) & ~out,
samples += (df['k'] > 10) & (df['k'] <= 15) & ~out,
samples += (df['k'] > 15) & (df['k'] <= 20) & ~out,
samples += (df['k'] > 20) & ~out,
samples += (df['l'] > 0) & (df['k'] > 0) & ~out,
samples += out,
samples += [True]*nsamples,

labels = ('$k=0$', '$k \in (1,5)$', '$k \in (6,10)$', '$k \in (11,15)$', '$k \in (16,20)$', '$k > 20$', '$l>0 | k>0$', 'No ZLB solution', '\textbf{Total}')

for i,s in enumerate(samples):
    table.loc[i] = (df[s]['time'].mean(), df[s]['time'].std(), "%.2f" %(sum(s)/nsamples*100) + "\%")

# table[['mean','std']] *= 1000
table.index = labels
# print(table.to_latex(float_format='%.3e', escape=False))
print(table.to_latex(escape=False))
print(df['time'].sum(), nsamples/df['time'].sum())

