#!/bin/python2
# -*- coding: utf-8 -*-

from tqdm import tqdm
import numpy as np
import statsmodels.api as sm
import pandas
import warnings
inv = np.linalg.inv
warnings.simplefilter("always")
import matplotlib.pyplot as plt

pars_raw = {
    #Define default parameter values here    
    'beta'  : 0.99,
    'sigma' : 1.0,
    'gamma' : 0.3, 
    'omega' : 0.66,
	'nut'   : 0.09,
    'a_y'   : 0.0,
    'a_p'   : 1.5,
    'a_s'   : 0.0,
    'rhop'  : 0.9,
    'rhoy'  : 0.7,
    'rhox'  : 0.998
}

def reduc(mtrx,dims):
    mtrx = np.matrix(mtrx)
    if type(dims) != int: 
        dims = np.sort(dims)[::-1]
        for i in dims:
            if mtrx[i,i] == 0:
                mtrx = np.delete(np.delete(mtrx,i,0),i,1)
            else:    
                mtrx = np.delete(np.delete(mtrx,i,0),i,1) - \
                    np.delete(mtrx,i,0)[:,i]/mtrx[i,i]*np.delete(mtrx,i,1)[i,:]
    else:         
        i = dims
        if mtrx[i,i] == 0:
            mtrx = np.delete(np.delete(mtrx,i,0),i,1)
        else:    
            mtrx = np.delete(np.delete(mtrx,i,0),i,1) - \
                np.delete(mtrx,i,0)[:,i]/mtrx[i,i]*np.delete(mtrx,i,1)[i,:]
    return np.array(mtrx)

def M_full(args):
    kappa = (1-args['omega'])*( 1 - args['beta']*args['omega'] )/args['omega']
    eta = (args['sigma'] + args['gamma'] + args['nut'])/(1-args['nut'])
    M = np.array([
        [1, 	0, 	0, 	-kappa,0,-1,0],
        [0, 	1, 	0, 	0, 	1/args['sigma'],0,-1],
        [0, 	0, 	1, 	0, 	1, 	0,0],
        [0, 	-eta,args['nut']/(1-args['nut']),1,-1,0,0],
        [-args['a_p'],-args['a_y'],-args['a_s'],0,1,0,0],
        [0, 	0, 	0, 	0, 	0, 	args['rhop'],0],
        [0, 	0, 	0, 	0, 	0, 	0, 	args['rhoy']]])
    return reduc(M,(3,4))

def P_full(args):
	kappa = (1-args['omega'])*( 1 - args['beta']*args['omega'] )/args['omega']
	P = np.array([
		[args['beta']-kappa,0,0,0,0,0,0],
        [1/args['sigma'],1,0,0,0,0,0],
        [1, 1-args['beta'], args['beta'],0,0,0,0],
        [0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0],
        [0,0,0,0,0,1,0],
        [0,0,0,0,0,0,1]
    ])
	return reduc(P,(3,4))


def Rho(args):
    return np.diag([args['rhop'],args['rhoy']])

def get_psi_inv(parameters, return_eigvs=0, return_omega=0, debug=0):
    M               = M_full(parameters)
    P               = P_full(parameters)
    try:
        N_inv           = inv(P).dot(M)
        eigvs, Lambda   = np.linalg.eig(N_inv)
        idx             = np.abs(eigvs).argsort()
        Lambda_inv      = inv(Lambda[:,idx])     
        Omega       = np.real(-inv(Lambda_inv[2:,:3]).dot(Lambda_inv[2:,3:]))
    except np.linalg.linalg.LinAlgError:
        Omega       = np.empty((2,2))
        Omega[:]    = np.NAN
        if debug:
            print('...there is singularity!')
    if return_omega:
        return Omega
    Epsilon         = Omega[:2].dot(Rho(parameters)).dot(inv(Omega[:2]))
    Epsilon_zero    = np.array([
        [Epsilon[0,0],  Epsilon[0,1],   0],
        [Epsilon[1,0],  Epsilon[1,1],   0],
        [0,             0,              0]
    ])
    Psi             = M[:3,:3] - P[:3,:3].dot(Epsilon_zero)
    if return_eigvs:
        return eigvs
    return inv(Psi)

def get_psi_inv_rhox(parameters, return_eigvs=0, debug=0):
	M               = M_full_with_rhox(parameters)
	P               = P_full_with_rhox(parameters)
	try:
		N_inv           = inv(P).dot(M)
		eigvs, Lambda   = np.linalg.eig(N_inv)
		idx             = np.abs(eigvs).argsort()
		Lambda_inv      = inv(Lambda[:,idx])     
		Omega       = np.real(-inv(Lambda_inv[3:,:3]).dot(Lambda_inv[3:,3:]))
		return Omega
	except np.linalg.linalg.LinAlgError:
		Omega       = np.empty((3,3))
		Omega[:]    = np.NAN
		if debug:
			print('...there is singularity!')
			print('Eigenvalues: ', eigvs)
		return Omega

def M_full_with_rhox(args):
    kappa = (1-args['omega'])*( 1 - args['beta']*args['omega'] )/args['omega']
    eta = (args['sigma'] + args['gamma'] + args['nut'])/(1-args['nut'])
    M = np.array([
        [1, 	0, 	0, 	-kappa,0,-1,0, 0],
        [0, 	1, 	0, 	0, 	1/args['sigma'],0,-1, 0],
        [0, 	0, 	1, 	0, 	1, 	0,0, 0],
        [0, 	-eta,args['nut']/(1-args['nut']),1,-1,0,0, 0],
        [-args['a_p'],-args['a_y'],-args['a_s'],0,1,0,0, 0],
        [0, 	0, 	0, 	0, 	0, 	args['rhop'],0, 0],
        [0, 	0, 	0, 	0, 	0, 	0, 	args['rhoy'], 0],
        [0, 	0, 	0, 	0, 	0, 	0, 	0, 	args['rhox']]])
    return reduc(M,(3,4))


def P_full_with_rhox(args):
    kappa = (1-args['omega'])*( 1 - args['beta']*args['omega'] )/args['omega']
    P = np.array([
        [args['beta']-kappa,0,0,0,0,0,0,0],
        [1/args['sigma'],1,0,0,0,0,0,0],
        [1, 0, args['beta'],0,0,0,0,0],
        [0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0],
        [0,0,0,0,0,1,0,0],
        [0,0,0,0,0,0,1,0],
        [0,0,0,0,0,0,0,1]
    ])
    return reduc(P,(3,4))


def Rho_with_rhox(args):
    return np.diag([args['rhop'],args['rhoy'],args['rhox']])

def get_ts_re(pars, ptype = 'a_s', pval = 0, no_iters = 1000, transition = 100, std_p = 0, std_y = 0, std_s=0, rnd_seed=0):
    np.random.seed(rnd_seed)
    size_variation_nadj     = np.max((np.size(std_s), np.size(std_y), np.size(std_p), np.size(pval)))
    if np.size(pval) == 1:  pval    = np.ones(size_variation_nadj)*pval
    Psi_inv         = []
    for j in pval:
        pars[ptype] = j
        Psi_inv.append(get_psi_inv_rhox(pars))
    Psi_inv         = np.array(Psi_inv)
    if np.isnan(Psi_inv).all():   return np.ones((3,1,no_iters-1))*np.nan
    di              = np.where(np.invert(np.isnan(Psi_inv[:,0,0])))
    Psi_inv         = Psi_inv[di]
    pval_adj        = np.shape(Psi_inv)[0]
    size_variation  = np.max((np.size(std_s), np.size(std_y), np.size(std_p), pval_adj))
    pars[ptype]     = np.min(pval)
    eps_p   = std_p*np.random.normal(0, 1, no_iters + transition).reshape(-1,1)
    eps_y   = std_y*np.random.normal(0, 1, no_iters + transition).reshape(-1,1)
    eps_s   = std_s*np.random.normal(0, 1, no_iters + transition).reshape(-1,1)
    iters       = range(no_iters + transition) 
    z_p, z_y, z_s 	= np.zeros(size_variation), np.zeros(size_variation), np.zeros(size_variation)
    P, Y, S     = [], [], []
    for t in iters:
        z_p     = pars['rhop']*z_p + eps_p[t]
        z_y     = pars['rhoy']*z_y + eps_y[t]
        z_s     = pars['rhox']*z_s + eps_s[t]
        States  = np.vstack([[z_p],[z_y],[z_s]])
        X       = np.einsum('ijk,ki->ij',Psi_inv,States) 
        p       = X[:,0]
        y       = X[:,1]
        s       = X[:,2]
        if t > transition: 
            praw = np.empty(size_variation_nadj)*np.nan
            yraw = np.empty(size_variation_nadj)*np.nan
            sraw = np.empty(size_variation_nadj)*np.nan
            praw[di]   = p
            yraw[di]   = y
            sraw[di]   = s
            P.append(praw)
            Y.append(yraw)
            S.append(sraw)
    P = np.swapaxes(np.array(P), 0, 1)
    Y = np.swapaxes(np.array(Y), 0, 1)
    S = np.swapaxes(np.array(S), 0, 1)
    return P, Y, S

def get_ts(pars, ptype = 'a_s', pval = 0, g = 0.9, b = 1.85, ioc = 1, no_iters = 1000, transition = 100, std_p = 0, std_y = 0, rnd_seed=0, std_Es=0, rho_Es=0):
    np.random.seed(rnd_seed)
    size_variation_nadj     = np.max((np.size(g), np.size(b), np.size(ioc), np.size(std_y), np.size(std_p), np.size(pval)))
    if np.size(pval) == 1:  pval    = np.ones(size_variation_nadj)*pval
    Psi_inv         = []
    for j in pval:
        pars[ptype] = j
        Psi_inv.append(get_psi_inv(pars))
    Psi_inv         = np.array(Psi_inv)
    if np.isnan(Psi_inv).all():   return np.nan, np.nan, np.nan
    di              = np.where(np.invert(np.isnan(Psi_inv[:,0,0])))
    Psi_inv         = Psi_inv[di]
    pval_adj        = np.shape(Psi_inv)[0]
    size_variation  = np.max((np.size(g), np.size(b), np.size(ioc), np.size(std_y), np.size(std_p), pval_adj))
    pars[ptype]     = np.min(pval)
    eps_p   = std_p*np.random.normal(0, 1, no_iters + transition).reshape(-1,1)
    eps_y   = std_y*np.random.normal(0, 1, no_iters + transition).reshape(-1,1)
    eps_Es  = std_Es*np.random.normal(0, 1, no_iters + transition).reshape(-1,1)
    iters       = range(no_iters + transition) 
    s, n1, n2   = np.ones(size_variation)*.1, np.ones(size_variation)*.2, np.ones(size_variation)*.2
    z_p, z_y    = np.zeros(size_variation), np.zeros(size_variation)
    Es          = s
    P, Y, S     = [], [], []
    for t in iters:
        s_1     = s
        Es      = (n1+n2)*g*s_1 + (n1-n2)*b + eps_Es[t]
        if rho_Es:  Es  = eps_Es[t] + rho_Es*Es
        z_p     = pars['rhop']*z_p + eps_p[t]
        z_y     = pars['rhoy']*z_y + eps_y[t]
        States  = np.vstack([[z_p],[z_y],[Es]])
        X       = np.einsum('ijk,ki->ij',Psi_inv,States) 
        p       = X[:,0]
        y       = X[:,1]
        s       = X[:,2]
        s_buff  = s_1
        s       = np.where(np.isinf(s_1), 0, s)
        s_1     = np.where(np.isinf(s_1), 0, s_1)
        u0raw   = ioc*-(pars['beta']*s - s_1)*s_1
        di_u0   = np.where(u0raw < 1e2)
        u0      = (u0raw+1)*np.inf
        u0[di_u0]    = np.exp(u0raw[di_u0])
        u1raw   = ioc*(pars['beta']*s-s_1)*(pars['beta']*(g*s_1 + b) - s_1)
        di_u1   = np.where(u1raw < 1e2)
        u1      = (u1raw+1)*np.inf
        u1[di_u1]    = np.exp(u1raw[di_u1])
        u2raw   = ioc*(pars['beta']*s-s_1)*(pars['beta']*(g*s_1 - b) - s_1)
        di_u2   = np.where(u2raw < 1e2)
        u2      = (u2raw+1)*np.inf
        u2[di_u2]    = np.exp(u2raw[di_u2])
        norm    = u0 + u1 + u2
        s_1     = s_buff
        di_norm         = np.where((norm!=0) & (np.invert(np.isinf(norm))))
        n1[di_norm]     = np.divide(u1[di_norm],norm[di_norm])
        n2[di_norm]     = np.divide(u2[di_norm],norm[di_norm])
        di_norm         = np.where(np.isinf(u1) & (u1>u0))
        n1[di_norm]     = 1
        n2[di_norm]     = 0
        di_norm         = np.where(np.isinf(u2) & (u2>u0))
        n1[di_norm]     = 0
        n2[di_norm]     = 1
        s = np.where(np.isinf(s_1), s_1, s)
        p = np.where(np.isinf(s_1), s_1, p)
        y = np.where(np.isinf(s_1), s_1, y)
        di_s    = np.where(np.abs(s)>1e10)
        s[di_s] = s[di_s]*np.inf
        y[di_s] = y[di_s]*np.inf
        p[di_s] = p[di_s]*np.inf
        if t > transition: 
            praw = np.empty(size_variation_nadj)*np.nan
            yraw = np.empty(size_variation_nadj)*np.nan
            sraw = np.empty(size_variation_nadj)*np.nan
            praw[di]   = p
            yraw[di]   = y
            sraw[di]   = s
            P.append(praw)
            Y.append(yraw)
            S.append(sraw)
    P = np.swapaxes(np.array(P), 0, 1)
    Y = np.swapaxes(np.array(Y), 0, 1)
    S = np.swapaxes(np.array(S), 0, 1)
    return P, Y, S

def get_data(lambda1=1600):
	## Import series
	gdp = np.genfromtxt('oecd-eu-gdp.csv', delimiter=',')
	cpi = np.genfromtxt('oecd-eu-cpi.csv', delimiter=',')
	msci = pandas.read_excel('eu-msci.xls')
	## Get labels
	ys = []
	ys.append(np.arange(77, 100,4))
	for i in range(1,10,4):
		ys.append('0' + str(i))
	for i in range(13,17,4):
		ys.append(str(i))
	for i in range(0,len(ys)):
		ys[i] = '\'' + str(ys[i])
	y 	= gdp[:,21]
	p 	= cpi[1:-3:3,16]
	s 	= np.array(msci[u'eu'])[0:-3:3]/p*100
	s   = np.log(s)
	y   = np.log(y)
	p   = np.log(p)
	S, s2 	= sm.tsa.filters.hpfilter(s, lambda1)
	Y, y2  	= sm.tsa.filters.hpfilter(y, lambda1)
	P, p2  	= sm.tsa.filters.hpfilter(p, lambda1)
	return P, Y, S, ys
