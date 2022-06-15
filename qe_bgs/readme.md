
# replication files for "A Structural Investigation of Quantitative Easing" (Boehl, Goy, Strobel)

## Files included
- readme.md
- folder `data`:
    - folder `sources`:
	    - ACMTermPremium.xlsx
	    - ebp_csv.csv
	    - SOMA_data.xlsx
	- load_data.m
	- BGS_est_data.csv
- folder `code`:
	- bgs_rnk_exo.yaml (model specification)
    - estimation.py (script to run the estimation)
	- estimation_summary.py (obtain latex tables)
    - fed_bs_data.py (script to reproduce Figure 1)
    - counterfactuals.py (script to reproduce Figure 2)
    - irfs1.py (script to reproduce Figure 3)
    - irfs2.py (script to reproduce Figure 4)
    - shock_decomposition.py (script to reproduce the historic shock decompositions from the appendix)
- folder `output`:
    - <empty>

    
## Installation of required packages

The code is based on three packages that are freely and publicly available: `pydsge`, `grgrlib` and `econsieve`. The versions conserved for _exact_ replication are tagged as "bgs_version" and can be installed using `pip` via:

```
pip install git+https://github.com/gboehl/pydsge.git@bgs_version
pip install git+https://github.com/gboehl/grgrlib.git@bgs_version
pip install git+https://github.com/gboehl/econsieve.git@bgs_version
```

If you are unfamiliar with `pip`, we provide a short guide here:

    https://pydsge.readthedocs.io/en/latest/installation_guide.html


## Code and description
- load_data.m - downloads data from FRED St Louis Fed and combines it with manually downloaded data to the BGS_est_data.csv file used in the estimation. The data in BGS_est_data.csv is downloaded on January 18, 2022. Note that running load_data.m will replace this file. This may imply minor changes to the data due to data revisions past our download date. 
- to replicate the estimation results _using our estimation_, download the following files and place them in the `output` folder (they are too large for github):
    - http://gregorboehl.com/data/bgs_final0_meta.npz
    - http://gregorboehl.com/data/bgs_final0_res.npz
    - http://gregorboehl.com/data/bgs_final0_sampler.h5
- bgs_rnk_exo.yaml - model specification, similar to dynares `mod`-files
- all Python code is intended to be run as `python <path/to/script.py>`:
    - estimation.py - run to estimate model with the given data
    - estimation_summary.py - run to obtain the latex content for tables 1 and 2
    - the other scripts replicate the respective figures as given above.
