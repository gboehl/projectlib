#!/bin/python2
# -*- coding: utf-8 -*-

import numpy as np
import scipy.optimize as so
import scipy.stats as ss
from numba import jit, njit

@njit(cache=True)
def t_func(gamma, tax, share):

    tau 		= 0.5*gamma*np.sqrt(.5)*tax
    alpha 		= (1 + np.sqrt(2)*tau/gamma**2/(1-tau)**2)
    s_bar 		= share**(1-1/alpha)
    phi 		= (gamma*(1-tau)*alpha)**2
    rho 		= np.exp(-phi)

    return rho, s_bar


@jit(cache=True)
def resid(x, *args):

    sample, taxes, share    = args[0]
    gamma, std              = x
    Y   = []
    ll  = 0

    for (t, tx) in enumerate(taxes[:-1]):

        rho, s_bar  = t_func(gamma, tx, share)
        y       = sample[t+1] - rho*sample[t] - (1-rho)*s_bar
        Y.append(y)
        ll      += ss.norm.logpdf(y, scale=std)

    return np.array(Y), ll


def estim(gamma0, std0, sample, tax_rate, share):

    llh             = lambda x, args: -resid(x,args)[1]
    gamma, std	    = so.fmin(llh, (gamma0, std0), args=((sample, tax_rate, share), ))

    print('gamma:', gamma.round(5), 'std:', np.abs(std).round(5))

    return gamma, np.abs(std)


# @jit(cache=True)
def simul(x, args, shape = 1e3):

    # np.random.seed(0)
    s, taxes, share = args
    gamma, std      = x

    S   = []
    SS  = []

    for tx in taxes:
        rho, s_bar  = t_func(gamma, tx, share)
        s       = rho*s + (1-rho)*s_bar + np.random.normal(scale=std, size=int(shape))
        S.append(s)
        SS.append(s_bar)

    S  = np.array(S)
    SS  = np.array(SS)

    iv0     = np.percentile(S, 2.5, 1)
    iv1     = np.percentile(S, 97.5, 1)
    med 	= np.median(S, 1)

    return med, iv0, iv1, SS

