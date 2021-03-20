#!/bin/python
# -*- coding: utf-8 -*-

import numpy as np
from numba import njit
from grgrlib.njitted import numba_rand_norm
from interpolation.splines import UCGrid, nodes, eval_linear
from interpolation.splines import extrap_options as xto


class model(object):

    def __init__(self, func, par_names, par_values, arg_names, arg_values, xfromv=None):

        self.func = func
        self.par_names = par_names
        self.pars = par_values.copy()
        self.init_pars = par_values.copy()
        self.arg_names = arg_names
        self.args = arg_values.copy()
        self.init_args = arg_values.copy()

        if xfromv is None:

            @njit
            def xfromv(v):
                return v

        self.xfromv = xfromv

    def __repr__(self):
        return "A generic representation of a model"

    def reset(self):
        self.pars = self.init_pars.copy()
        self.args = self.init_args.copy()

    def get_args(self):

        arg_dict = dict(zip(self.arg_names, self.args))

        return arg_dict

    def set_args(self, **args):
        for a in zip(args, args.values()):
            self.args[self.arg_names.index(a[0])] = a[1]


@njit(nogil=True, cache=True)
def bh_func(pars, state, expect, args):

    rational = args[0]
    dis, dlt, bet, gam, cos, gam2 = pars

    type4 = gam2 != 123456789

    if not type4:
        gam2 = gam

    xe = expect

    xm1 = np.ascontiguousarray(state[:, 0])
    xm2 = np.ascontiguousarray(state[:, 1])

    if state.shape[1] < 3:
        xm3 = np.zeros_like(xm1)
    else:
        xm3 = np.ascontiguousarray(state[:, 2])

    x_shp = np.shape(xm1)

    if rational:
        prof0 = (xm1 - dis*xm2)**2 - cos
    else:
        prof0 = -(xm1 - dis*xm2)*xm2 - cos

    prof1 = (xm1 - dis*xm2) * (gam*xm3 + bet - dis*xm2)

    if bet == 0 and not gam:
        prof2 = np.zeros_like(prof1)
    else:
        prof2 = (xm1 - dis*xm2) * (gam2*xm3 - bet - dis*xm2)

    frac0 = 1/(1 + np.exp(dlt*(prof1-prof0)) +
               (bool(bet) | type4) * np.exp(dlt*(prof2-prof0)))
    frac1 = 1/(1 + np.exp(dlt*(prof0-prof1)) +
               (bool(bet) | type4) * np.exp(dlt*(prof2-prof1)))
    frac2 = (bool(bet) | type4) / \
        (1 + np.exp(dlt*(prof0-prof2)) + np.exp(dlt*(prof1-prof2)))

    x = (frac0*xe + (frac1*gam + frac2*gam2)*xm1 + (frac1-frac2)*bet)/dis

    if state.shape[1] < 3:
        ts = np.concatenate((
            x.reshape(x_shp+(1,)),
            xm1.reshape(x_shp+(1,))),
            axis=1)
    else:
        ts = np.concatenate((
            x.reshape(x_shp+(1,)),
            xm1.reshape(x_shp+(1,)),
            xm2.reshape(x_shp+(1,))),
            axis=1)

    # return ts, (frac0, frac1, frac2)
    return ts, np.vstack((frac0, frac1, frac2))


@njit(nogil=True, cache=True)
def bh_xfromv(v):
    return v[:, 0]


def simulate_raw(t_func, T, transition_phase, initial_state, noise):

    x = initial_state
    res = np.empty((T,)+x.shape)

    for t in range(transition_phase):
        x = t_func(x)

    for t in range(T):

        x = t_func(x)
        x[0] += np.random.randn()*noise
        res[t] = x

    return res


def simulate(t_func, T=None, transition_phase=0, initial_state=None, eps=None, noise=False, numba_jit=True, show_warnings=True):
    """Generic simulation command
    """

    if T is None:
        if eps is not None:
            T = eps.shape[0]
        else:
            UnboundLocalError("Either `T` or `eps` must be given.")

    if initial_state is None:
        if ndim is not None:
            initial_state = np.zeros(ndim)
        else:
            UnboundLocalError(
                "Either `initial_state` or `ndim` must be given.")

    if numba_jit:
        res = simulate_jit(t_func, int(T), int(
            transition_phase), initial_state, noise)
    else:
        res = simulate_raw(t_func, int(T), int(
            transition_phase), initial_state, noise)

    return res


def pfi_t_func(pfunc, grid, numba_jit=True):
    """Wrapper to return a jitted transition function based on the policy function and the grid
    """

    def pfi_t_func_wrap(state):

        newstate = eval_linear(grid, pfunc, state, xto.LINEAR)

        return newstate

    if numba_jit:
        return njit(pfi_t_func_wrap, nogil=True)
    else:
        return pfi_t_func_wrap


def pfi_raw(func, xfromv, pars, args, grid_shape, grid, gp, eps_max, it_max, init_pfunc, x0=None, use_x0=True, verbose=False):

    ndim = len(grid)
    eps = 1e9
    it_cnt = 0

    xe = xfromv(eval_linear(grid, init_pfunc,
                            init_pfunc.reshape(-1, ndim), xto.LINEAR))
    values = func(pars, gp, xe, args=args)[0]
    svalues = values.reshape(grid_shape)

    if use_x0:
        z_old = eval_linear(grid, svalues, x0, xto.LINEAR)[0]

    while eps > eps_max or eps_max < 0:

        it_cnt += 1
        values_old = values.copy()
        values = svalues.reshape(-1, 3)
        xe = xfromv(eval_linear(grid, svalues, values, xto.LINEAR))
        values = func(pars, gp, xe, args=args)[0]
        # values = np.maximum(values, -1e2)
        # values = np.minimum(values, 1e2)
        svalues = values.reshape(grid_shape)

        if use_x0:
            z = eval_linear(grid, svalues, x0, xto.LINEAR)[0]
            eps = np.abs(z-z_old)
            z_old = z
        else:
            eps = np.linalg.norm(values - values_old)

        if verbose:
            print(it_cnt, '- eps:', eps)

        if it_cnt == it_max:
            break

    return values.reshape(grid_shape), it_cnt, eps


def pfi(grid, model, init_pfunc=None, eps_max=1e-8, it_max=100, numba_jit=True, **pfiargs):
    """Somewhat generic policy function iteration

    This assumes the form 

        v_t = f(E_t x_{t+1}, v_{t-1})

    where f(.) is `func` and x_t = h(v_t) (where h(.) is `xfromv` can be directly retrieved from v_t. The returning function `p_func` is then the array repesentation of the solution v_t = g(v_{t-1}).

    In the future this should also allow for 

        y_t = f(E_t x_{t+1}, w_t, v_{t-1})

    with implied existing functions x_t = h_1(y_t), v_t = h_2(y_t) and w_t = h_3(y_t).

    Parameters
    ----------
    func : dynamic system as described above. Takes the argument `pars', `state`,  and `expect` (must be jitted)
    xfrom : see above (must be jitted)
    pars: list of parameters to func
    grid: for now only UCGrid from interpolate.py are supported and tested
    eps_max: float of maximum error tolerance

    Returns
    -------
    numpy array 
    """

    if numba_jit:
        pfi_func = pfi_jit
    else:
        pfi_func = pfi_raw

    flag = 0

    func = model.func
    xfromv = model.xfromv
    pars = np.ascontiguousarray(model.pars)
    args = np.ascontiguousarray(model.args)

    gp = nodes(grid)
    grid_shape = tuple(g_spec[2] for g_spec in grid)
    grid_shape = grid_shape + (len(grid),)

    if init_pfunc is None:
        init_pfunc = 0.

    if isinstance(init_pfunc, (float, int)):
        init_pfunc = np.ones(grid_shape)*init_pfunc

    if init_pfunc.shape != grid_shape:
        init_pfunc = init_pfunc.reshape(grid_shape)

    p_func, it_cnt, eps = pfi_func(
        func, xfromv, pars, args, grid_shape, grid, gp, eps_max, it_max, init_pfunc, **pfiargs)
    if np.isnan(p_func).any():
        flag += 1
    if np.isnan(p_func).all():
        flag += 2
    if it_cnt >= it_max:
        flag += 4

    return p_func, it_cnt, flag


@njit(cache=True)
def race_njit(func, pars, args, x0, xss, n, neff, eps, max_iter):

    x = np.ones(n)*xss
    x[0:3] = x0[::-1]

    cond = False
    cnt = 1

    while not cond:
        x_old = x[:neff].copy()

        for i in range(0, min(cnt+1, n-4)):
            y = np.ascontiguousarray(x[i:i+3][::-1])
            x[3+i] = func(pars, y.reshape(1, -1), x[4+i], args)[0][0, 0]

        cond = np.max(np.abs(x_old - x[:neff])) < eps

        if cnt == max_iter:
            break

        if np.any(np.isnan(x)):
            break

        cnt += 1

    return x, cnt


def race(mod, x0, xss=0, n=500, eps=1e-8, max_iter=5000, neff=None, verbose=True):

    if neff is None:
        neff = n

    x, cnt = race_njit(mod.func, mod.pars, mod.args, x0, xss, n, neff, eps, max_iter)

    flag = 0
    mess = ''

    if cnt == max_iter:
        flag += 1
        mess += 'max_iter reached'

    if np.any(np.isnan(x)):
        flag += 2
        mess += 'contains NaNs'

    if np.any(np.isinf(x)):
        flag += 4
        mess += 'contains infs'

    if verbose and mess:
        print('race done. ', mess)

    return x, (flag, cnt)


@njit(cache=True)
def rocket_njit(func, pars, args, x0, xss, n, eps, max_iter, max_horizon):

    x_fin = np.empty(n)
    x_fin[0:3] = x0[::-1]

    fracs = np.empty((n,3))

    x = np.ones(max_horizon)*xss
    x[0:3] = x0[::-1]

    flag = np.zeros(3)

    for i in range(n):

        cond = False
        cnt = 2

        while True:

            x_old = x[4]

            for t in range(min(cnt,max_horizon-4)): 
                X = np.ascontiguousarray(x[t:t+3][::-1])
                x[3+t] = func(pars, X.reshape(1,-1), x[4+t], args)[0][0,0]

            if np.abs(x_old - x[4]) < eps and cnt > 2:
                break

            if cnt == max_iter:
                flag[0] = 1
                break

            if np.any(np.isnan(x)):
                flag[1] = 1
                break

            if np.any(np.isinf(x)):
                flag[2] = 1
                break

            cnt += 1

        X = np.ascontiguousarray(x[0:3][::-1])
        xt, frac = func(pars, X.reshape(1,-1), x[4], args)

        x_fin[i] = xt[0,0]
        fracs[i] = frac[:,0]

        x = np.roll(x, -1)
        x[-1] = xss

    return x_fin, fracs, flag


def rocket(mod, x0, xss=0, n=500, eps=1e-16, max_iter=None, max_horizon=1000, verbose=True):

    if max_iter is None:
        max_iter = max_horizon

    x, fracs, flag = rocket_njit(mod.func, mod.pars, mod.args, x0, xss, n, eps, max_iter, max_horizon)

    mess = ''

    fin_flag = 0

    if flag[0]:
        fin_flag += 1
        mess += 'max_iter reached'

    if flag[1]:
        fin_flag += 2
        mess += 'contains NaNs'

    if flag[2]:
        fin_flag += 4
        mess += 'contains infs'

    if verbose and mess:
        print('rocket done. ', mess)

    return x, fracs, fin_flag

def xstar(mod):

    z = (2*np.arctan(1-2*(mod.pars[0]-1)/(mod.pars[3]-1))/mod.pars[1] + mod.pars[4])/(mod.pars[0]-1)/(mod.pars[3]-1)

    return np.sqrt(max(0,z))

bh_par_names = ['discount_factor', 'intensity_of_choice', 'bias',
                'degree_trend_extrapolation', 'costs', 'degree_trend_extrapolation_type2']
bh_pars = np.array([1/.99, 1., 1., 0., 0., 123456789])
bh_arg_names = ['rational']
bh_args = np.array([0, 0])

bh1998 = model(bh_func, bh_par_names, bh_pars,
               bh_arg_names, bh_args, bh_xfromv)

simulate_jit = njit(simulate_raw, nogil=True, fastmath=True)
pfi_jit = njit(pfi_raw, nogil=True, fastmath=True)
