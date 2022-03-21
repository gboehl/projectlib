#!/bin/python
# -*- coding: utf-8 -*-

from pydsge import *

# set all paths relative to script location
pth = os.path.dirname(os.path.abspath(__file__))
dfile0 = os.path.join(os.path.split(pth)[0],'output', 'bgs_final0_meta.npz')

# load model
mod0 = DSGE.load(dfile0, force_parse=False)

# get pandas.DataFrame of estimation results and print them as latex table
ms = mod0.mcmc_summary()
print(ms.round(3).to_latex())

