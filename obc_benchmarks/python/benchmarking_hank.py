#!/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from grgrlib import *
from pydsge import *
import tqdm
from timeit import default_timer as timer
from pydsge.engine import *

from matplotlib import rc
rc('legend', fontsize=18)


# load stuff

AA = np.loadtxt('/home/gboehl/rsh/blh/matrices/HANK2_BAYERETAL/AA.txt', delimiter=',')
BB = np.loadtxt('/home/gboehl/rsh/blh/matrices/HANK2_BAYERETAL/BB.txt', delimiter=',')
CC = np.loadtxt('/home/gboehl/rsh/blh/matrices/HANK2_BAYERETAL/CC.txt', delimiter=',')
vv = np.loadtxt('/home/gboehl/rsh/blh/matrices/HANK2_BAYERETAL/list_of_vars.txt', delimiter=',', dtype=str)
vv = np.array([v[2:-1] for v in vv])


# lets stick with UTF-8
if 'π' in vv:
    vv[list(vv).index('π')] = 'pi'
if 'πw' in vv:
    vv[list(vv).index('πw')] = 'piw'

# DD is the mapping from shocks to states
shock_states = [s for s in ['Z','ZI','mu','muw','Gshock','A','Tlevshock','Tprogshock','Rshock','Sshock', 'HDrop', 'Bcbshock', 'Kcbshock'] if s in vv]
shocks = ['e_' + s for s in shock_states]

DD = np.zeros((len(vv), len(shocks)))
for i,v in enumerate(shock_states):
    DD[:,shock_states.index(v)] = CC[:,list(vv).index(v)]

# add constraint. RN is the shadow/notational rate 
# the constraint acts on RB: rb_t = max(rn_t, x_bar)
const_var = 'RB'
rix = list(vv).index('RB') 
vv = np.hstack((vv, 'RN'))
dimy = len(vv)

AA = np.pad(AA, ((0,0),(0,1)))
BB = np.pad(BB, ((0,0),(0,1)))
CC = np.pad(CC, ((0,0),(0,1)))

BB[rix, [rix,-1]] = BB[rix, [-1,rix]] 
CC[rix, [rix,-1]] = CC[rix, [-1,rix]] 
fb0 = np.zeros(dimy)
fc0 = np.zeros(dimy)
fb0[rix] = 1
fb0[-1] = -1

# populate dictionary with model matrices
mdict = {}

# misc
mdict['vars'] = vv
mdict['shocks'] = shocks
mdict['const_var'] = const_var
mdict['x_bar'] = -1 # lower bound (relative to st.st.)

# sys matrices w/o constraint equation
mdict['AA'] = AA
mdict['BB'] = BB
mdict['CC'] = CC
mdict['DD'] = DD

# constraint equation (MP rule)
mdict['fb'] = -fb0
mdict['fc'] = fc0

# load the dict and initialize a model instance. 
# (l_max, k_max) are the maximum values of (l,k) to be considered
mod = gen_sys_from_dict(mdict, l_max=3, k_max=30, parallel=False, verbose=1)

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

table.index = labels
print(table.to_latex(float_format='%.3e', escape=False))
print(df['time'].sum(), nsamples/df['time'].sum())
