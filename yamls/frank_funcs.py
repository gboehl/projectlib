#!/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import scipy.stats as ss
import scipy.optimize as so


def calc_sigW(zW, sprd, zeta_spb):

    def func(sigW):

        G = ss.norm.cdf(zW-sigW, 0, 1)
        F = ss.norm.cdf(zW, 0, 1) 
        Fprime = ss.norm.pdf(zW, 0, 1)

        omegabar = np.exp(sigW*zW-1/2*sigW**2)
        GAMMA = omegabar*(1-F)+G
        dGdomega = Fprime/sigW                         
        d2Gdomega2 = -zW*Fprime/omegabar/sigW**2
        dGammadomega = 1-F
        d2Gammadomega2 = -Fprime/omegabar/sigW

        # dGdsigma = -zW*Fprime/sigW 
        dGdsigma = -zW*ss.norm.pdf(zW-sigW, 0, 1)/sigW	
        d2Gdomegadsigma = -Fprime*(1-zW*(zW-sigW))/sigW**2
        dGammadsigma = -ss.norm.pdf(zW-sigW, 0, 1)
        d2Gammadomegadsigma : (zW/sigW-1)*Fprime

        MU = (1-1/sprd)/(dGdomega/dGammadomega*(1-GAMMA)+G) 
        nK = 1-(GAMMA-MU*G)*sprd                                    # 
        Rhostar = 1/nK-1
        GammamuG = GAMMA-MU*G
        GammamuGprime = dGammadomega-MU*dGdomega

        zeta_bw = omegabar*MU*nK*(d2Gammadomega2*dGdomega-d2Gdomega2*dGammadomega)/(dGammadomega-MU*dGdomega)**2/sprd/(1-GAMMA+dGammadomega*(GAMMA-MU*G)/(dGammadomega-MU*dGdomega))
        zeta_zw = omegabar*(dGammadomega-MU*dGdomega)/(GAMMA-MU*G)
        zeta_bw_zw = zeta_bw/zeta_zw

        return -zeta_bw_zw/(1-zeta_bw_zw)*nK/(1-nK) - zeta_spb

    res = so.root_scalar(func, x0=.2, x1=.5)

    if res.converged:
        return res.root
    else:
        raise ValueError('No value found! Results from root finding:\n\n'+str(res))


