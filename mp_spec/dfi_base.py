#!/bin/python
# -*- coding: utf-8 -*-

from tqdm import tqdm
import numpy as np
import numpy.linalg as nl
import statsmodels.api as sm
import pandas as pd
import scipy.optimize as so


pars_raw = {
    # Define default parameter values here
    'beta': 0.99,
    'sigma': 1.0,
    'gamma': 0.3,
    'omega': 0.66,
    'nut': 0.076,
    'a_y': 0.0,
    'a_p': 1.5,
    'a_s': 0.0,
    'rhop': 0.9,
    'rhoy': 0.7,
    'rhox': 0.998,
    'std_p': 0.,
    'std_y': 0.,
    'std_s': 0.
}


def reduc(A, dims):

    dims = np.sort(dims)[::-1]
    for i in dims:
        if A[i, i] == 0:
            A = np.delete(np.delete(A, i, 0), i, 1)
        else:
            A = np.delete(np.delete(A, i, 0), i, 1) - np.outer(np.delete(A,
                                                                         i, 0)[:, i]/A[i, i], np.delete(A, i, 1)[i, :])

    return A


def M_full(args):
    kappa = (1-args['omega'])*(1 - args['beta']*args['omega'])/args['omega']
    eta = (args['sigma'] + args['gamma'] + args['nut'])/(1-args['nut'])
    M = np.array([
        # pi,   y,      s,      mc,     r,  a,  d
        [1, 	0, 	0, 	-kappa, 0, -1, 0],
        [0, 	1, 	0, 	0, 	1/args['sigma'], 0, -1],
        [0, 	0, 	1, 	0, 	1, 	0, 0],
        [0, 	-eta, args['nut']/(1-args['nut']), 1, -1, 0, 0],
        [-args['a_p'], -args['a_y'], -args['a_s'], 0, 1, args['a_y'], 0],
        [0, 	0, 	0, 	0, 	0, 	args['rhop'], 0],
        [0, 	0, 	0, 	0, 	0, 	0, 	args['rhoy']]])
    return reduc(M, (3, 4))


def P_full(args):
    kappa = (1-args['omega'])*(1 - args['beta']*args['omega'])/args['omega']
    P = np.array([
        # `-kappa` adjusts for real rate effect in mc
        [args['beta']-kappa, 0, 0, 0, 0, 0, 0],
        [1/args['sigma'], 1, 0, 0, 0, 0, 0],
        [1, 1-args['beta'], args['beta'], 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 1]
    ])
    return reduc(P, (3, 4))


def Rho(args):
    return np.diag([args['rhop'], args['rhoy']])


def get_psi_inv(parameters, return_eigvs=0, debug=0):

    M = M_full(parameters)
    P = P_full(parameters)

    try:
        N_inv = nl.inv(P).dot(M)
        eigvs, Lambda = np.linalg.eig(N_inv)
        idx = np.abs(eigvs).argsort()
        Lambda_inv = nl.inv(Lambda[:, idx])
        Omega = np.real(-nl.inv(Lambda_inv[2:, :3]).dot(Lambda_inv[2:, 3:]))
    except np.linalg.linalg.LinAlgError:
        Omega = np.empty((2, 2))
        Omega[:] = np.NAN
        if debug:
            print('...there is singularity!')

    if return_eigvs:
        return eigvs

    Epsilon = Omega[:2].dot(Rho(parameters)).dot(nl.inv(Omega[:2]))
    Epsilon_zero = np.array([
        [Epsilon[0, 0],  Epsilon[0, 1],   0],
        [Epsilon[1, 0],  Epsilon[1, 1],   0],
        [0,             0,              0]
    ])
    Psi = M[:3, :3] - P[:3, :3].dot(Epsilon_zero)

    return nl.inv(Psi)


def get_psi_inv_rhox(parameters, return_eigvs=0, debug=0):
    M = M_full_with_rhox(parameters)
    P = P_full_with_rhox(parameters)
    try:
        N_inv = nl.inv(P) @ M
        eigvs, Lambda = np.linalg.eig(N_inv)
        idx = np.abs(eigvs).argsort()
        Lambda_inv = nl.inv(Lambda[:, idx])
        Omega = np.real(-nl.inv(Lambda_inv[3:, :3]).dot(Lambda_inv[3:, 3:]))
        return Omega
    except np.linalg.linalg.LinAlgError:
        Omega = np.empty((3, 3))
        Omega[:] = np.NAN
        if debug:
            print('...there is singularity!')
            print('Eigenvalues: ', eigvs)
        return Omega


def M_full_with_rhox(args):
    kappa = (1-args['omega'])*(1 - args['beta']*args['omega'])/args['omega']
    eta = (args['sigma'] + args['gamma'] + args['nut'])/(1-args['nut'])
    M = np.array([
        # pi,   y,      s,      mc,     r,      a,  d, x
        [1, 	0, 	0, 	-kappa, 0,      -1, 0, 0],
        [0, 	1, 	0, 	0, 	1/args['sigma'], 0, -1, 0],
        [0, 	0, 	1, 	0, 	1, 	0, 0, -args['beta']],
        [0, 	-eta, args['nut']/(1-args['nut']), 1, -1, 0, 0, 0],
        [-args['a_p'], -args['a_y'], -args['a_s'], 0, 1, args['a_y'], 0, 0],
        [0, 	0, 	0, 	0, 	0, 	args['rhop'], 0, 0],
        [0, 	0, 	0, 	0, 	0, 	0, 	args['rhoy'], 0],
        [0, 	0, 	0, 	0, 	0, 	0, 	0, 	args['rhox']]])
    return reduc(M, (3, 4))


def P_full_with_rhox(args):
    kappa = (1-args['omega'])*(1 - args['beta']*args['omega'])/args['omega']
    P = np.array([
        [args['beta']-kappa, 0, 0, 0, 0, 0, 0, 0],
        [1/args['sigma'], 1, 0, 0, 0, 0, 0, 0],
        [1, 0, args['beta'], 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 1]
    ])
    return reduc(P, (3, 4))


def Rho_with_rhox(args):
    return np.diag([args['rhop'], args['rhoy'], args['rhox']])


def get_ts_re(pars, ptype='a_s', pval=0, no_iters=1000, transition=100, rnd_seed=0):

    np.random.seed(rnd_seed)

    size_variation_nadj = np.max((np.size(pars['std_s']), np.size(
        pars['std_y']), np.size(pars['std_p']), np.size(pval)))

    if np.size(pval) == 1:
        pval = np.ones(size_variation_nadj)*pval

    Psi_inv = []
    for j in pval:
        pars[ptype] = j
        Psi_inv.append(get_psi_inv_rhox(pars))

    Psi_inv = np.array(Psi_inv)
    if np.isnan(Psi_inv).all():
        return np.ones((3, 1, no_iters-1))*np.nan

    di = np.where(np.invert(np.isnan(Psi_inv[:, 0, 0])))
    Psi_inv = Psi_inv[di]
    pval_adj = np.shape(Psi_inv)[0]
    size_variation = np.max((np.size(pars['std_s']), np.size(
        pars['std_y']), np.size(pars['std_p']), pval_adj))
    pars[ptype] = np.min(pval)
    noise = np.random.normal(0, 1, (3, no_iters + transition))
    eps_p = pars['std_p']*noise[0]
    eps_y = pars['std_y']*noise[1]
    eps_s = pars['std_s']*noise[2]
    iters = range(no_iters + transition)
    z_p, z_y, z_s = 3*[np.zeros(size_variation)]

    P, Y, S = [], [], []
    for t in iters:
        z_p = pars['rhop']*z_p + eps_p[t]
        z_y = pars['rhoy']*z_y + eps_y[t]
        z_s = pars['rhox']*z_s + eps_s[t]
        States = np.vstack([[z_p], [z_y], [z_s]])
        X = np.einsum('ijk,ki->ij', Psi_inv, States)
        p = X[:, 0]
        y = X[:, 1]
        s = X[:, 2]
        if t > transition:
            praw = np.empty(size_variation_nadj)*np.nan
            yraw = np.empty(size_variation_nadj)*np.nan
            sraw = np.empty(size_variation_nadj)*np.nan
            praw[di] = p
            yraw[di] = y
            sraw[di] = s
            P.append(praw)
            Y.append(yraw)
            S.append(sraw)

    P = np.swapaxes(np.array(P), 0, 1)
    Y = np.swapaxes(np.array(Y), 0, 1)
    S = np.swapaxes(np.array(S), 0, 1)

    return P, Y, S


def get_ts(pars, ptype='a_s', pval=0, g=0.9, b=1.85, ioc=1, no_iters=1000, transition=100, rnd_seed=0):

    np.random.seed(rnd_seed)
    size_variation_nadj = np.max((np.size(g), np.size(b), np.size(
        ioc), np.size(pars['std_y']), np.size(pars['std_p']), np.size(pval)))

    if np.size(pval) == 1:
        pval = np.ones(size_variation_nadj)*pval

    Psi_inv = []
    for j in pval:
        pars[ptype] = j
        Psi_inv.append(get_psi_inv(pars))

    Psi_inv = np.array(Psi_inv)
    if np.isnan(Psi_inv).all():
        return np.nan, np.nan, np.nan

    di = np.where(np.invert(np.isnan(Psi_inv[:, 0, 0])))
    Psi_inv = Psi_inv[di]
    pval_adj = np.shape(Psi_inv)[0]
    size_variation = np.max((np.size(g), np.size(b), np.size(
        ioc), np.size(pars['std_y']), np.size(pars['std_p']), pval_adj))

    pars[ptype] = np.min(pval)
    eps_p = pars['std_p'] * \
        np.random.normal(0, 1, no_iters + transition).reshape(-1, 1)
    eps_y = pars['std_y'] * \
        np.random.normal(0, 1, no_iters + transition).reshape(-1, 1)
    iters = range(no_iters + transition)
    s, n1, n2 = np.ones(
        size_variation)*.1, np.ones(size_variation)*.2, np.ones(size_variation)*.2
    z_p, z_y = np.zeros(size_variation), np.zeros(size_variation)
    Es = s
    P, Y, S = [], [], []
    N1, N2, U0, U1, U2 = [], [], [], [], []

    for t in iters:
        s_1 = s
        Es = (n1+n2)*g*s_1 + (n1-n2)*b

        z_p = pars['rhop']*z_p + eps_p[t]
        z_y = pars['rhoy']*z_y + eps_y[t]
        States = np.vstack([[z_p], [z_y], [Es]])
        X = np.einsum('ijk,ki->ij', Psi_inv, States)
        p = X[:, 0]
        y = X[:, 1]
        s = X[:, 2]
        s_buff = s_1
        s = np.where(np.isinf(s_1), 0, s)
        s_1 = np.where(np.isinf(s_1), 0, s_1)
        u0raw = ioc*-(pars['beta']*s - s_1)*s_1
        di_u0 = np.where(u0raw < 1e2)
        u0 = (u0raw+1)*np.inf
        u0[di_u0] = np.exp(u0raw[di_u0])
        u1raw = ioc*(pars['beta']*s-s_1)*(pars['beta']*(g*s_1 + b) - s_1)
        di_u1 = np.where(u1raw < 1e2)
        u1 = (u1raw+1)*np.inf
        u1[di_u1] = np.exp(u1raw[di_u1])
        u2raw = ioc*(pars['beta']*s-s_1)*(pars['beta']*(g*s_1 - b) - s_1)
        di_u2 = np.where(u2raw < 1e2)
        u2 = (u2raw+1)*np.inf
        u2[di_u2] = np.exp(u2raw[di_u2])
        norm = u0 + u1 + u2
        s_1 = s_buff
        di_norm = np.where((norm != 0) & (np.invert(np.isinf(norm))))
        n1[di_norm] = np.divide(u1[di_norm], norm[di_norm])
        n2[di_norm] = np.divide(u2[di_norm], norm[di_norm])
        di_norm = np.where(np.isinf(u1) & (u1 > u0))
        n1[di_norm] = 1
        n2[di_norm] = 0
        di_norm = np.where(np.isinf(u2) & (u2 > u0))
        n1[di_norm] = 0
        n2[di_norm] = 1
        s = np.where(np.isinf(s_1), s_1, s)
        p = np.where(np.isinf(s_1), s_1, p)
        y = np.where(np.isinf(s_1), s_1, y)
        di_s = np.where(np.abs(s) > 1e10)
        s[di_s] = s[di_s]*np.inf
        y[di_s] = y[di_s]*np.inf
        p[di_s] = p[di_s]*np.inf

        if t > transition:
            praw = np.empty(size_variation_nadj)*np.nan
            yraw = np.empty(size_variation_nadj)*np.nan
            sraw = np.empty(size_variation_nadj)*np.nan
            praw[di] = p
            yraw[di] = y
            sraw[di] = s
            P.append(praw)
            Y.append(yraw)
            S.append(sraw)
            N1.append(list(n1))
            N2.append(list(n2))
            U0.append(list(u0raw/ioc))
            U1.append(list(u1raw/ioc))
            U2.append(list(u2raw/ioc))

    P = np.swapaxes(np.array(P), 0, 1)
    Y = np.swapaxes(np.array(Y), 0, 1)
    S = np.swapaxes(np.array(S), 0, 1)

    N1 = np.swapaxes(np.array(N1), 0, 1)
    N2 = np.swapaxes(np.array(N2), 0, 1)
    U0 = np.swapaxes(np.array(U0), 0, 1)
    U1 = np.swapaxes(np.array(U1), 0, 1)
    U2 = np.swapaxes(np.array(U2), 0, 1)

    return P, Y, S, (N1, N2, U0, U1, U2)


def get_data(path, lambda1=1600, start=None, end=None):

    df = pd.read_csv(path, index_col=0, parse_dates=True)

    y_raw = df['GDP'][start:end]
    p_raw = df['CPI'][start:end]
    s_raw = df['MSCI'][start:end]

    s_log = np.log(s_raw)
    y_log = np.log(y_raw)
    p_log = np.log(p_raw)

    s = sm.tsa.filters.hpfilter(s_log, lambda1)[0]
    y = sm.tsa.filters.hpfilter(y_log, lambda1)[0]
    p = sm.tsa.filters.hpfilter(p_log, lambda1)[0]

    return p, y, s


def sm_errors(x, *args):

    pars, w_mat, return_value, re_sim, iter_no, batch_no, moms_data, j = args

    if re_sim:
        pars['nut'], pars['rhox'], pars['std_s'], pars['std_p'], pars['std_y'] = x
    else:
        pars['nut'], g, b, pars['std_p'], pars['std_y'] = x

    if return_value == 'errors':
        batch_no_loc = 1
    else:
        batch_no_loc = batch_no

    moms_batch = np.empty((batch_no_loc, 6))
    for i in range(batch_no_loc):
        if re_sim:
            sim = np.array(get_ts_re(pars, no_iters=iter_no+1,
                                     transition=0, rnd_seed=i+batch_no*j)).reshape(3, iter_no)
        else:
            sim = np.array(get_ts(pars, g=g, b=b, no_iters=iter_no+1,
                                  transition=0, rnd_seed=i+batch_no*j)[:3]).reshape(3, iter_no)

        moms_sim = np.hstack(
            (np.std(sim, axis=1), np.corrcoef(sim).flat[[1, 2, 5]]))

        if np.isfinite(moms_sim).all():
            moms_batch[i, :] = moms_sim

    if return_value == 'moments':
        return moms_batch

    errors = (moms_batch - moms_data)/moms_data

    if return_value == 'w':
        return np.cov(errors.T)

    errors = np.mean(errors, axis=0)

    return errors.T @ w_mat @ errors


def sm_run(x0, bnd, pars, moms_data, nruns, batch_no, iter_no, re_sim=False, solver='L-BFGS-B', warnings='ignore', verbose=False):

    # alternative solver: 'TNC'
    np.warnings.filterwarnings(warnings)

    if re_sim:
        pnames = ('nu', 'rho_x', 'sig_x', 'sig_pi', 'sig_y')
    else:
        pnames = ('nu', 'gamma', 'alpha', 'sig_pi', 'sig_y')

    j = 0
    RES = []
    pbar = tqdm(total=nruns)

    while True:

        res = so.minimize(sm_errors, x0, args=(pars, np.identity(6), 'errors', re_sim, iter_no,
                                               batch_no, moms_data, j), method=solver, options={'disp': verbose}, bounds=bnd)
        vcvar = sm_errors(res['x'], pars, np.identity(
            6), 'w', re_sim, iter_no, batch_no, moms_data, j)
        res = so.minimize(sm_errors, res['x'], args=(pars, nl.inv(
            vcvar), 'errors', re_sim, iter_no, batch_no, moms_data, j), method=solver, options={'disp': verbose}, bounds=bnd)

        j += 1

        if res['success']:
            RES.append(res['x'])
            pbar.update(1)
            if j == nruns:
                break
        else:
            nruns += 1

        if verbose:
            print('')
            if re_sim:
                print('** RE Model with exogenous stock price fluctuations **')
            else:
                print('** Speculative Model **')

            [print(' %s: %s' % (var, round(val, 5)))
             for var, val in zip(pnames, res['x'])]

    return RES, pnames
