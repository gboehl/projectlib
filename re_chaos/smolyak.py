#!/bin/python
# -*- coding: utf-8 -*-

import numpy as np
from numba import njit
from grgrlib.njitted import numba_rand_norm
from interpolation.splines import UCGrid, nodes, eval_linear
from interpolation.splines import extrap_options as xto

from interpolation.smolyak.interp import SmolyakInterp


def pfi_t_func_smolyak(pfunc, sgrid):

    si = SmolyakInterp(sgrid, pfunc)

    def pfi_t_func_wrap(state):

        newstate = si.interpolate(np.expand_dims(state,1))

        return newstate

    return pfi_t_func_wrap


def pfi_smolyak_raw(func, xfromv, pars, args, sgrid, eps_max, it_max, use_norm=True, initval=None, verbose=False):

    eps = 1e9
    it_cnt = 0

    values = func(pars, sgrid.grid, 0., args=args)[0]
    # svalues = values.reshape(sgrid.grid.shape)
    si = SmolyakInterp(sgrid, values)
    # z_old = eval_linear(grid, svalues, initval, xto.LINEAR)[0]
    z_old = si.interpolate(np.expand_dims(initval,1))[0][0]

    while eps > eps_max:

        it_cnt += 1
        values_old = values.copy()
        # xe = xfromv(eval_linear(grid, svalues, values, xto.LINEAR))
        xe = xfromv(si.interpolate(values))
        values = func(pars, sgrid.grid, xe, args=args)[0]
        # svalues = values.reshape(grid_shape)

        if it_cnt == it_max:
            break

        si = SmolyakInterp(sgrid, values)
        if initval is not None:
            z = si.interpolate(np.expand_dims(initval,1))[0][0]
            eps = np.abs(z-z_old)
            z_old = z
        elif use_norm:
            eps = np.linalg.norm(values - values_old)
        else:
            eps = np.nanmax(np.abs(values - values_old))

        if verbose:
            print(it_cnt, '- eps:', eps)


    return values, it_cnt, eps


def pfi_smolyak(sgrid, model, eps_max=1e-8, it_max=100, numba_jit=True, **pfiargs):
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
        pfi_func = pfi_smolyak_jit
    else:
        pfi_func = pfi_smolyak_raw

    flag = 0

    func = model.func
    xfromv = model.xfromv
    pars = np.ascontiguousarray(model.pars)
    args = np.ascontiguousarray(model.args)

    p_func, it_cnt, eps = pfi_func(
        func, xfromv, pars, args, sgrid, eps_max, it_max, **pfiargs)
    if np.isnan(p_func).any():
        flag += 1
    if np.isnan(p_func).all():
        flag += 2
    if it_cnt >= it_max:
        flag += 4

    return p_func, it_cnt, flag


pfi_smolyak_jit = njit(pfi_smolyak_raw, nogil=True, fastmath=True)
