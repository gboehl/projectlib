#!/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from grgrlib import *
from pydsge import *
import tqdm
from timeit import default_timer as timer
import time
import scipy.linalg as sl
import cloudpickle as cpickle
from grgrlib import fast0, ouc, klein, speed_kills
from pydsge.engine import preprocess
from pydsge.clsmethods import DSGE_RAW
from pydsge.parser import DSGE

aca = np.ascontiguousarray



from matplotlib import rc
rc('legend', fontsize=18)


# load stuff

# AA = np.loadtxt('/home/gboehl/rsh/blh/matrices/HANK2_BAYERETAL/AA.txt', delimiter=',')
# BB = np.loadtxt('/home/gboehl/rsh/blh/matrices/HANK2_BAYERETAL/BB.txt', delimiter=',')
# CC = np.loadtxt('/home/gboehl/rsh/blh/matrices/HANK2_BAYERETAL/CC.txt', delimiter=',')
# vv = np.loadtxt('/home/gboehl/rsh/blh/matrices/HANK2_BAYERETAL/list_of_vars.txt', delimiter=',', dtype=str)
AA = np.loadtxt('/home/gboehl/rsh/stuff/blh_local/archiv/matrices/HANK2_BAYERETAL/AA.txt', delimiter=',')
BB = np.loadtxt('/home/gboehl/rsh/stuff/blh_local/archiv/matrices/HANK2_BAYERETAL/BB.txt', delimiter=',')
CC = np.loadtxt('/home/gboehl/rsh/stuff/blh_local/archiv/matrices/HANK2_BAYERETAL/CC.txt', delimiter=',')
vv = np.loadtxt('/home/gboehl/rsh/stuff/blh_local/archiv/matrices/HANK2_BAYERETAL/list_of_vars.txt', delimiter=',', dtype=str)
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


# functions must be adjusted for this exercise...

def gen_sys_from_dict(mdict, l_max=None, k_max=None, parallel=True, force_processing=False, verbose=True):

    global processed_mdicts

    # add l/k max to dict
    mdict['l_max'] = l_max
    mdict['k_max'] = k_max

    # check if dict has already been processed. If so, load it
    if not force_processing and 'processed_mdicts' in globals():
        mdict_dump = cpickle.dumps(mdict)
        if mdict_dump in processed_mdicts:
            if verbose:
                print('[get_sys:]'.ljust(15, ' ') + ' Model dict was already processed. Loading from cache.')
            return processed_mdicts[mdict_dump]
    else:
        processed_mdicts = {}

    from pydsge.tools import t_func, irfs, traj, k_map, shock2state

    # create a dummy DSGE instance
    class DSGE_DUMMY(DSGE_RAW):
        pass

    self = DSGE_DUMMY()
    self.debug = True
    self.set_par = None
    self.fdict = mdict

    # fix value of x_bar
    if 'x_bar' in mdict:
        self.x_bar = mdict['x_bar']
    else:
        print("Parameter `x_bar` (maximum value of the constraint) not specified. Assuming x_bar = -1 for now.")
        self.x_bar = -1

    self.vv = mdict['vars']
    self.shocks = mdict['shocks']
    self.neps = len(mdict['shocks'])
    self.const_var = mdict['const_var']

    self.observables = mdict.get('observables')
    if self.observables:
        self.nobs = len(mdict.get('observables'))

    ZZ0 = mdict.get('ZZ0')
    ZZ1 = mdict.get('ZZ1')
    fd = mdict.get('fd')

    res_sys = gen_sys(self, mdict['AA'], mdict['BB'], mdict['CC'], mdict['DD'], mdict['fb'], mdict['fc'], fd, ZZ0, ZZ1, l_max, k_max, False, parallel, verbose)

    processed_mdicts[cpickle.dumps(mdict)] = res_sys

    return res_sys


def gen_sys_from_yaml(self, par=None, l_max=None, k_max=None, get_hx_only=False, parallel=False, verbose=True):

    self.par = self.p0() if par is None else list(par)
    try:
        self.ppar = self.pcompile(self.par)  # parsed par
    except TypeError:
        raise SyntaxError(
            "At least one parameter is a function of other parameters, and should be declared in `parafunc`.")

    if not self.const_var:
        raise NotImplementedError('Package is only meant to work with OBCs')

    # fix value of x_bar
    if 'x_bar' in [p.name for p in self.parameters]:
        self.x_bar = self.par[[p.name for p in self.parameters].index('x_bar')]
    elif 'x_bar' in self.parafunc[0]:
        pf = self.parafunc
        self.x_bar = pf[1](self.par)[pf[0].index('x_bar')]
    else:
        print("Parameter `x_bar` (maximum value of the constraint) not specified. Assuming x_bar = -1 for now.")
        self.x_bar = -1

    self.vv = np.array([v.name for v in self.variables])

    AA0 = self.AA(self.ppar)              # forward
    BB0 = self.BB(self.ppar)              # contemp
    CC0 = self.CC(self.ppar)              # backward
    DD0 = -self.PSI(self.ppar).astype(float)

    fbc = self.bb(self.ppar).flatten().astype(float)  # constraint
    fd0 = -self.bb_PSI(self.ppar).flatten().astype(float)  # constraint
    fb0 = -fbc[:len(self.vv)]
    fc0 = -fbc[len(self.vv):]

    # observables from z
    ZZ0 = self.ZZ0(self.ppar).astype(float)
    ZZ1 = self.ZZ1(self.ppar).squeeze().astype(float)

    return gen_sys(self, AA0, BB0, CC0, DD0, fb0, fc0, fd0, ZZ0, ZZ1, l_max, k_max, get_hx_only, parallel, verbose)


def gen_sys(self, AA0, BB0, CC0, DD0, fb0, fc0, fd0, ZZ0, ZZ1, l_max, k_max, get_hx_only, parallel, verbose):
    """Generate system matrices expressed in the one-sided, first-order compressed dimensionality reduction given a set of parameters. 

    Details can be found in "Efficient Solution of Models with Occasionally Binding Constraints" (Gregor Boehl).
    If no parameters are given this will default to the calibration in the `yaml` file.

    Parameters
    ----------
    par : array or list, optional
        The parameters to parse into the transition function. (defaults to calibration in `yaml`)
    l_max : int, optional
        The expected number of periods *until* the constraint binds (defaults to 3).
    k_max : int, optional
        The expected number of periods for which the constraint binds (defaults to 17).
    verbose : bool or int, optional
        Level of verbosity
    """

    st = time.time()

    # set default values of l_max & k_max
    if l_max is not None:
        if l_max < 2 and k_max > 0:
            print('[get_sys:]'.ljust(15, ' ') +
                  ' `l_max` must be at least 2 (is %s). Correcting...' % l_max)
            l_max = 2
        # effective l_max is one lower because algorithm exists on l_max
        l_max += 1
        # TODO: test if true

    elif hasattr(self, 'lks'):
        l_max = self.lks[0]
    else:
        l_max = 3

    if k_max is not None:
        pass
    elif hasattr(self, 'lks'):
        k_max = self.lks[1]
    else:
        k_max = 17

    self.lks = np.array([l_max, k_max])

    # start
    vv0 = self.vv

    # z-space is the space of the original variables
    dimx = len(vv0)
    dimeps = len(self.shocks)

    # y-space is the space of the original variables augmented by the shocks
    c_arg = list(vv0).index(str(self.const_var))
    fc0 = -fc0/fb0[c_arg]
    fb0 = -fb0/fb0[c_arg]

    # create auxiliry vars for those both in A & C
    inall = ~fast0(AA0, 0) & ~fast0(CC0, 0)
    if np.any(inall):
        vv0 = np.hstack((vv0, [v + '_lag' for v in vv0[inall]]))
        AA0 = np.pad(AA0, ((0, sum(inall)), (0, sum(inall))))
        BB0 = np.pad(BB0, ((0, sum(inall)), (0, sum(inall))))
        CC0 = np.pad(CC0, ((0, sum(inall)), (0, sum(inall))))
        DD0 = np.pad(DD0, ((0, sum(inall)), (0, 0)))
        fb0 = np.pad(fb0, (0, sum(inall)))
        fc0 = np.pad(fc0, (0, sum(inall)))

        if ZZ0 is not None:
            ZZ0 = np.pad(ZZ0, ((0, 0), (0, sum(inall))))

        BB0[-sum(inall):, -sum(inall):] = np.eye(sum(inall))
        BB0[-sum(inall):, :-sum(inall)][:, inall] = -np.eye(sum(inall))
        CC0[:, -sum(inall):] = CC0[:, :-sum(inall)][:, inall]
        CC0[:, :-sum(inall)][:, inall] = 0

    # create representation in y-space
    AA0 = np.pad(AA0, ((0, dimeps), (0, dimeps)))
    BB0 = sl.block_diag(BB0, np.eye(dimeps))
    CC0 = np.block([[CC0, DD0], [np.zeros((dimeps, AA0.shape[1]))]])
    fb0 = np.pad(fb0, (0, dimeps))
    if fd0 is not None:
        fc0 = -np.hstack((fc0, fd0))
    else:
        fc0 = np.pad(fc0, (0, dimeps))

    inq = ~fast0(CC0, 0) | ~fast0(fc0)
    inp = (~fast0(AA0, 0) | ~fast0(BB0, 0)) & ~inq

    # check dimensionality
    dimq = sum(inq)
    dimp = sum(inp)

    # create hx. Do this early so that the procedure can be stopped if get_hx_only
    if ZZ0 is None:
        # must create dummies
        zp = np.empty(dimp)
        zq = np.empty(dimq)
        zc = np.empty(1)
    else:
        zp = ZZ0[:, inp[:-dimeps]]
        zq = ZZ0[:, inq[:-dimeps]]
        zc = ZZ1

    AA = np.pad(AA0, ((0, 1), (0, 0)))
    BBU = np.vstack((BB0, fb0))
    CCU = np.vstack((CC0, fc0))
    BBR = np.pad(BB0, ((0, 1), (0, 0)))
    CCR = np.pad(CC0, ((0, 1), (0, 0)))
    BBR[-1, list(vv0).index(str(self.const_var))] = -1

    fb0[list(vv0).index(str(self.const_var))] = 0

    self.svv = vv0[inq[:-dimeps]]
    self.cvv = vv0[inp[:-dimeps]]
    self.vv = np.hstack((self.cvv, self.svv))

    self.dimx = len(self.vv)
    self.dimq = dimq
    self.dimp = dimp
    self.dimy = dimp+dimq
    self.dimeps = dimeps
    self.hx = zp, zq, zc

    if get_hx_only:
        return self

    PU = -np.hstack((BBU[:, inq], AA[:, inp]))
    MU = np.hstack((CCU[:, inq], BBU[:, inp]))

    PR = -np.hstack((BBR[:, inq], AA[:, inp]))
    MR = np.hstack((CCR[:, inq], BBR[:, inp]))
    gg = np.pad([float(self.x_bar)], (dimp+dimq-1, 0))

    # avoid QL in jitted funcs
    R, Q = sl.rq(MU.T)
    MU = R.T
    PU = Q @ aca(PU)

    R, Q = sl.rq(MR.T)
    MR = R.T
    PR = Q @ aca(PR)
    gg = Q @ gg

    solver = 'klein' # to be implemented
    # solver = 'speed_kills' # to be implemented

    stx = timer()
    if solver == 'speed_kills':
        omg, lam = speed_kills(PU, MU, dimp, dimq, tol=1e-4)
    else:
        omg, lam = klein(PU, MU, nstates=dimq, verbose=verbose, force=False)
    time_lin = timer() - stx

    # finally add relevant stuff to the class

    fq0 = fc0[inq]
    fp1 = fb0[inp]
    fq1 = fb0[inq]

    self.sys = omg, lam, self.x_bar
    self.ff = fq1, fp1, fq0

    # preprocess all system matrices until (l_max, k_max)
    stx = timer()
    preprocess(self, PU, MU, PR, MR, gg, fq1, fp1, fq0, parallel, verbose)
    time_preprocess = timer() - stx

    if verbose:
        print('[get_sys:]'.ljust(15, ' ')+' Creation of system matrices finished in %ss.' %
              np.round(time.time() - st, 3))

    return time_lin, time_preprocess


times_hank = []
for _ in tqdm.tqdm(range(20)):
    times = gen_sys_from_dict(mdict, l_max=3, k_max=30, parallel=False, force_processing=True, verbose=False)
    times_hank.append(times)

yaml = '/home/gboehl/rsh/yamls_pydsge/rank.yaml'

mod = DSGE.read(yaml, verbose=True)
p = mod.set_par('calib', verbose=True)

times_rank = []
for _ in tqdm.tqdm(range(20)):
    times = gen_sys_from_yaml(mod, l_max=3, k_max=30, parallel=False, verbose=False)
    times_rank.append(times)


print('RANK')
print('Solving linear model & %1.3e & %1.3e & -- \\\\' %(np.array(times_rank)[:,0].mean(), np.array(times_rank)[:,0].std()))
print('Preprocessing OBC & %1.3e & %1.3e & -- \\\\' %(np.array(times_rank)[:,1].mean(), np.array(times_rank)[:,1].std()))

print('HANK')
print('Solving linear model & %1.3e & %1.3e & -- \\\\' %(np.array(times_hank)[:,0].mean(), np.array(times_hank)[:,0].std()))
print('Preprocessing OBC & %1.3e & %1.3e & -- \\\\' %(np.array(times_hank)[:,1].mean(), np.array(times_hank)[:,1].std()))

