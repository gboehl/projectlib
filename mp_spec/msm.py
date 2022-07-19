#!/bin/python
# -*- coding: utf-8 -*-

import numpy as np
from dfi_base import *


pars = pars_raw.copy()

# chose whether or not stock markets are speculative
re_sim = False

# obtain the data moments
PQ, YQ, SQ = get_data('data.csv', end='2012-Q3')
moms_data = np.hstack((np.std([PQ, YQ, SQ], axis=1), np.corrcoef([PQ, YQ, SQ]).flat[[1, 2, 5]]))
iter_no = np.shape(PQ)[0] # sample lenght is the lenght of the data

if re_sim:
    # RE model:
    nu0 = .2
    rho_x0 = .95
    std_x0 = .001
    std_p0 = .001
    std_y0 = .004
    x0 = (nu0, rho_x0, std_x0, std_p0, std_y0)
    bnd = ((-.4, .99), (0.5, .999), (0, 0.2), (0, 0.2), (0, 0.2))
else:
    # speculative model:
    nu0 = .2
    g0 = 1.
    b0 = 1.22
    std_p0 = .004
    std_y0 = .004
    x0 = (nu0, g0, b0, std_p0, std_y0)
    bnd = ((-.4, .99), (0, 2), (0, None), (0, None), (0, None))

res = sm_run(x0, bnd, pars, moms_data, nruns=100, batch_no=100, iter_no=iter_no, re_sim=re_sim)

_ = [print(' %s: %s (%s)' % (var, round(val, 5), round(std, 5))) for var, val, std in zip(res[1], np.mean(res[0], axis=0), np.std(res[0], axis=0))]

err = sm_errors(np.mean(res[0], 0), pars, None, 'moments', re_sim, iter_no, 100, moms_data, 0)

_ = [print(' %s: %s (%s)' % (var, round(val, 5), round(std, 5))) for var, val, std in zip(np.arange(6), np.mean(err, 0), np.std(err, 0))]



