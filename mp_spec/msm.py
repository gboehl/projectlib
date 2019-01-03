#!/bin/python2
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from tqdm import tqdm
from base import *
import scipy.optimize as so
import scipy.stats as ss
import warnings

## chose model
s_shock		= 1
s_shock		= 0 	# whether or not stock price fluctuations are exogenous
re_sim 		= 1 	# whether or not stock markets are speculative
# re_sim 		= 0

tran_no 	= 50
batch_no 	= 100

## obtain the data moments
lam             = 1600
PQ, YQ, SQ, ys  = get_data(lambda1=lam)
moms_data 		= np.hstack((np.std([PQ,YQ,SQ], axis=1), np.corrcoef([PQ,YQ,SQ]).flat[[1,2,5]]))
iter_no 		= np.shape(PQ)[0]

def sm_errors(x, *args):
	w_mat 					= args[0]
	return_w				= args[1]
	re_sim					= args[2]
	if re_sim: 	
		nu, var1, var2, std_p, std_y 	= x
		if s_shock:
			pars['rhox'] 		= var1
			std_s 				= var2
		else:
			std_s 	= 0
			pars['sigma'] 		= var1
			pars['omega'] 		= var2
	else: 		nu, g, b, std_p, std_y 	= x
	pars['nut']  			= nu
	MOMS 		= []
	for i in range(batch_no):
		# print(np.shape(get_ts_re(pars, no_iters=iter_no+1, transition = tran_no, std_p=std_p, std_y=std_y, std_s=std_s, rnd_seed=i)))
		if re_sim: 	sim 	= np.array(get_ts_re(pars, no_iters=iter_no+1, transition = tran_no, std_p=std_p, std_y=std_y, std_s=std_s, rnd_seed=i)).reshape(3,iter_no)
		else: 		sim 	= np.array(get_ts(pars, g = g, b = b, no_iters=iter_no+1, transition = tran_no, std_p=std_p, std_y=std_y, rnd_seed=i)).reshape(3,iter_no)
		moms_sim 	= np.hstack((np.std(sim, axis=1), np.corrcoef(sim).flat[[1,2,5]]))
		MOMS.append(moms_sim)
	moms_sim 	= np.array(MOMS)
	moms_sim[np.isinf(moms_sim)]  = 1
	moms_sim[np.isnan(moms_sim)]  = 1
	errors 		= (moms_sim - moms_data)/moms_data
	if return_w:
		return moms_sim, np.mean(errors, axis=0), np.linalg.pinv(np.dot(errors.T,errors)/batch_no)
	errors 	= np.mean(errors, axis=0)
	return 	errors.T.dot(w_mat).dot(errors)

pars		= pars_raw.copy()
if re_sim:
	## RE model:
	nu0 	= 0.0
	if s_shock:
		var1_0 	= .85 	
		var2_0  = .001
	else:
		var1_0 	= 1 
		var2_0  = .66
	std_p0  = .001
	std_y0  = .001
	x0 		= (nu0, var1_0, var2_0, std_p0, std_y0)
	bnd 	= ((-.4,.99),(0,.99),(0,None),(0,None),(0,None))
	bnd 	= ((-.4,.99),(0,4),(0,1),(0,None),(0,None))
	bnd 	= ((None,None),(0.1,None),(0,None),(0,None),(0,None))
else:
	## speculative model:
	nu0 			= .09
	g0       		= 0.99
	g0       		= 1.006
	b0       		= 1.22
	std_p0   		= .001
	std_y0   		= .004
	x0 		= (nu0, g0, b0, std_p0, std_y0)
	bnd 	= ((-.4,1),(0,None),(0,None),(0,None),(0,None))

w_mat 	= np.identity(6)

res 	= so.minimize(sm_errors, x0, args=(w_mat, 0, re_sim), options={'disp': True}, bounds=bnd)
moms_sim, errors, w_mat2 	= sm_errors(res['x'], w_mat, 1, re_sim)
res2 	= so.minimize(sm_errors, res['x'], args=(w_mat2, 0, re_sim), options={'disp': True}, bounds=bnd)
moms_sim2, errors2, wmat2 	= sm_errors(res2['x'], w_mat, 1, re_sim)

if not re_sim: 	
	print('    ** Speculative Model **')
	print('Step 2, estimated values: \n nu: ', res2['x'][0], '\n gamma: ', res2['x'][1], '\n alpha: ', res2['x'][2], '\n sigma_pi: ', res2['x'][3],'\n sigma_y: ', res2['x'][4])
if re_sim: 	
	if s_shock: print('    ** RE Model WITH exogenous stock price fluctuations **')
	else: 		print('    ** RE Model WITHOUT exogenous stock price fluctuations **')
	print('Step 2, estimated values: \n nu: ', res2['x'][0], '\n var1: ', res2['x'][1], '\n var2: ', res2['x'][2], '\n sigma_pi: ', res2['x'][3],'\n sigma_y: ', res2['x'][4])
print('Step 2, estimated values:\n', res2['x'])
print('Step 2, moment errors:\n', errors2)
print('Data moments:\n', moms_data)
print('Step 2, simulated moments:\n', np.mean(moms_sim2, axis=0))
print('Step 2, StDev of simulated moments:\n', np.std(moms_sim2, axis=0))

