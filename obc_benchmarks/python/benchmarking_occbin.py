#!/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import scipy.io as si

mat = si.loadmat('../matlab/occbin_benchmark30.mat')

ls = mat['Ls']
ks = mat['Ks']
flags = mat['flags']
ts = mat['Ts']

res = np.vstack((ls, ks, flags, ts)).T
nsamples = res.shape[0]

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

table.index = labels
# print(table.to_latex(float_format='%.3e', escape=False))
print(table.to_latex(escape=False))
print(df['time'].sum(), nsamples/df['time'].sum())

1/6.42e-6 * 0.010449
