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

    frac0 = 1/(1 + np.exp(dlt*(prof1-prof0)) + (bool(bet) | type4) * np.exp(dlt*(prof2-prof0)))
    frac1 = 1/(1 + np.exp(dlt*(prof0-prof1)) + (bool(bet) | type4) * np.exp(dlt*(prof2-prof1)))
    frac2 = (bool(bet) | type4) / (1 + np.exp(dlt*(prof0-prof2)) + np.exp(dlt*(prof1-prof2)))

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

    return ts, (frac0, frac1, frac2)


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
        res = simulate_jit(t_func, int(T), int(transition_phase), initial_state, noise)
    else:
        res = simulate_raw(t_func, int(T), int(transition_phase), initial_state, noise)

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


    xe = xfromv(eval_linear(grid, init_pfunc, init_pfunc.reshape(-1,ndim), xto.LINEAR))
    values = func(pars, gp, xe, args=args)[0]
    svalues = values.reshape(grid_shape)
    
    if use_x0:
        z_old = eval_linear(grid, svalues, x0, xto.LINEAR)[0]

    while eps > eps_max or eps_max < 0:

        it_cnt += 1
        values_old = values.copy()
        values = svalues.reshape(-1,3)
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

    if isinstance(init_pfunc, (float,int)):
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


bh_par_names = ['discount_factor', 'intensity_of_choice', 'bias', 'degree_trend_extrapolation', 'costs', 'degree_trend_extrapolation_type2']
bh_pars = np.array([1/.99, 1., 1., 0., 0., 123456789])
bh_arg_names = ['rational']
bh_args = np.array([0, 0])

bh1998 = model(bh_func, bh_par_names, bh_pars, bh_arg_names, bh_args, bh_xfromv)

simulate_jit = njit(simulate_raw, nogil=True, fastmath=True)
pfi_jit = njit(pfi_raw, nogil=True, fastmath=True)
