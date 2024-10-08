import numpy as np

from sequence_jacobian import simple, solved, combine, create_model, grids, hetblocks

import grgrlib.hanktools as gsj

model = gsj.load_model('/home/gboehl/rsh/emc/yamls/hank2_hh.py')
hh = model.hh


'''Part 1: Blocks'''

@simple
def pricing(pi, mc, zeta_p, mup, eps_p, pistar, beta, iota_p):

    kappap = (1-zeta_p*beta)*(1-zeta_p) / ((1 + iota_p*beta)*zeta_p)
    pistarlog = (1+pistar/100).apply(np.log) 

    nkpc = 1/(1 + beta*iota_p)*(beta*((1+pi(+1)).apply(np.log) - pistarlog) + iota_p*((1+pi(-1)).apply(np.log) - pistarlog)) \
            + kappap*(mc - 1/mup) \
            + eps_p \
            - ((1 + pi).apply(np.log) - pistarlog)

    return nkpc


@simple
def arbitrage(div, p, r):
    equity = div(+1) + p(+1) - p * (1 + r(+1))
    return equity


@simple
def labor(Y, w, K, Z, eps_z, alpha):
    N = (Y / Z / eps_z.apply(np.exp) / K(-1) ** alpha) ** (1 / (1 - alpha))
    mc = w * N / (1 - alpha) / Y
    return N, mc


@simple
def investment(Q, K, I, r, N, mc, Z, delta, phiss, alpha, eps_i, eps_z):
    x = I/I(-1)
    xPrime = I(+1)/I
    inv = eps_i.apply(np.exp)*Q*(1 - 1/(2*phiss)*(x - 1)**2 - 1/phiss*(x - 1)*x) + eps_i(+1).apply(np.exp)*Q(+1)/((1+r(+1))*phiss)*(xPrime-1)*xPrime**2 - 1
    val = (1+r(+1))*Q - (1-delta)*Q(+1) - alpha * Z * eps_z(+1).apply(np.exp)* (N(+1) / K) ** (1 - alpha) * mc(+1)
    return inv, val


@simple
def dividend(Y, I, w, N, K, delta, phiss):
    k_adjust = 1/(2*phiss)*(I/I(-1) - 1)**2
    I_res = K - (1 - delta) * K(-1) - (1 - k_adjust)*I
    div = Y - w * N - I
    return I_res, div


@simple
def monetary(i, pi, Y, rstar, pistar, eps_r, phi_pi, phi_y, rho):
    taylor = rho*i(-1) + (1-rho)*(rstar/100 + phi_pi*(pi-pistar/100)  + phi_y*(Y.apply(np.log) - Y(-1).apply(np.log))) + eps_r - i
    return taylor


@simple
def fiscal(r, w, N, G, eps_g, Bg):
    tax = (r * Bg + G*eps_g.apply(np.exp)) / w / N
    return tax


@simple
def finance(i, p, pi, r, div, omega, eps_omega, pshare):
    rb = r - omega*eps_omega.apply(np.exp)
    ra = pshare(-1) * (div + p) / p(-1) + (1 - pshare(-1)) * (1 + r) - 1
    fisher = 1 + i(-1) - (1 + r) * (1 + pi)
    return rb, ra, fisher


@simple
def wage(pi, w):
    piw = (1 + pi) * w / w(-1) - 1
    return piw


@simple
def union(piw, N, tax, w, UCE, zeta_w, muw, vphi, sig_l, beta, eps_w, iota_w, pistar):

    kappaw = (1-zeta_w*beta)*(1-zeta_w) / ((1 + iota_w*beta)*zeta_w)
    pistarlog = (1+pistar/100).apply(np.log) 

    wnkpc = 1/(1 + beta*iota_w)*(beta*((1 + piw(+1)).apply(np.log) - pistarlog) + iota_w*((1 + piw(-1)).apply(np.log) - pistarlog)) \
            + kappaw*(vphi * N ** (1 + sig_l) - (1 - tax)*w*N*UCE/muw) \
            + eps_w \
            - ((1 + piw).apply(np.log) - pistarlog)

    return wnkpc


@simple
def mkt_clearing(p, A, B, Bg, C, I, G, eps_g, CHI, omega, eps_omega, Y):
    wealth = A + B
    asset_mkt = p + Bg - wealth
    goods_mkt = C + I + G*eps_g.apply(np.exp) + CHI + omega * eps_omega.apply(np.exp) * B - Y
    return asset_mkt, wealth, goods_mkt


@simple
def share_value(p, tot_wealth, Bh):
    pshare = p / (tot_wealth - Bh)
    return pshare


@simple
def partial_ss(Y, N, K, r, tot_wealth, Bg, delta):
    """Solves for (mup, alpha, Z, w) to hit (tot_wealth, Y, K, pi)."""
    # 1. Solve for markup to hit total wealth
    p = tot_wealth - Bg
    mc = 1 - r * (p - K) / Y
    mup = 1 / mc

    # 2. Solve for capital share to hit K
    alpha = (r + delta) * K / Y / mc

    # 3. Solve for TFP to hit Y
    Z = Y * K ** (-alpha) * N ** (alpha - 1)

    # 4. Solve for w such that piw = 0
    w = mc * (1 - alpha) * Y / N

    return p, mc, mup, alpha, Z, w


@simple
def union_ss(tax, w, UCE, N, muw, sig_l):
    """Solves for (vphi) to hit (wnkpc)."""
    vphi = (1 - tax) * w * UCE / muw / N ** (1 + sig_l)
    wnkpc = vphi * N ** (1 + sig_l) - (1 - tax) * w * UCE / muw
    return vphi, wnkpc


@simple
def measurement(Y, C, I, w, N, TOP10Y, TOP10W):
    # trivially true since ln(x) - ln(xSS) ~= (x-xSS)/xSS
    dY = Y.apply(np.log) - Y(-1).apply(np.log) 
    dC = C.apply(np.log) - C(-1).apply(np.log)
    dI = I.apply(np.log) - I(-1).apply(np.log)
    dW = w.apply(np.log) - w(-1).apply(np.log)
    # true since nSS = 1 and hence log(nSS) = 0
    n_obs = N.apply(np.log)
    # this is the deviation from SS _in levels_, same as the data
    top10y_obs = TOP10Y
    top10w_obs = TOP10W
    return dY, dC, dI, dW, n_obs, top10y_obs, top10w_obs


'''Part 2: Embed HA block'''

def make_grids(bmax, amax, kmax, nB, nA, nK, nZ, rho_z, sigma_z, eps_sigma):
    b_grid = grids.agrid(amax=bmax, n=nB)
    a_grid = grids.agrid(amax=amax, n=nA)
    k_grid = grids.agrid(amax=kmax, n=nK)[::-1].copy()
    e_grid, pi_e, Pi = grids.markov_rouwenhorst(rho=rho_z, sigma=sigma_z*np.exp(eps_sigma), N=nZ)
    return b_grid, a_grid, k_grid, e_grid, Pi, pi_e


def income(e_grid, tax, w, N, pi_e, tau, eps_tau):
    # redistribute, but keep total amount of income fix
    y_grid = (1 - tax) * w * N * e_grid
    z_grid = y_grid**(1-tau*np.exp(eps_tau)) + np.sum(pi_e*(y_grid - y_grid**(1-tau*np.exp(eps_tau))))
    return z_grid

# a measurement function
def dist_measures(a, b, a_grid, b_grid, z_grid, rb, ra, D):

    # each measure must have the shape of the distribution and will be dot-multiplied by D
    inc = z_grid[:,np.newaxis,np.newaxis] + (1 + rb)*b_grid[:, np.newaxis] + (1 + ra) * a_grid
    inc_sortinds = np.argsort(inc.flatten())
    D_sorted = D.flatten()[inc_sortinds]
    inc_sorted = inc.flatten()[inc_sortinds]
    top10inds = np.cumsum(D_sorted) > .9
    top10y = np.vdot(D_sorted[top10inds],inc_sorted[top10inds])/np.vdot(D,inc) * np.ones_like(D)

    wealth = b + a
    wealth_sortinds = np.argsort(wealth.flatten())
    D_sorted = D.flatten()[wealth_sortinds]
    wealth_sorted = wealth.flatten()[wealth_sortinds]
    top10inds = np.cumsum(D_sorted) > .9
    top10w = np.vdot(D_sorted[top10inds],wealth_sorted[top10inds])/np.vdot(D,wealth) * np.ones_like(D)

    return top10y, top10w


'''Part 3: DAG'''

def dag():
    # # Combine Blocks
    household = hh.add_hetinputs([income, make_grids])
    household = household.add_hetoutputs([dist_measures])
    production = combine([labor, investment])

    blocks = [household, pricing, arbitrage, production, dividend, monetary, fiscal, share_value, finance, wage, union, mkt_clearing, measurement]

    two_asset_model = create_model(blocks, name='Two-Asset HANK')

    # # Steadt state DAG
    blocks_ss = [household, partial_ss, dividend, monetary, fiscal, share_value, finance, union_ss, mkt_clearing]
    two_asset_model_ss = create_model(blocks_ss, name='Two-Asset HANK SS')

    # # Steady State
    calibration = {'Y': 1., 'N': 1.0, 'K': 10., 'Q': 1, 'I': 0.2,
                   # steady state values
                   'pistar': 0.625, 
                   'rstar': 1.25, 
                   'ybar': 0.4,
                   'nbar': 2,
                   'tot_wealth': 14, 
                   'delta': 0.02, 'muw': 1.1, 'Bh': 1.04, 'Bg': 2.8,
                   'G': 0.2, 
                   # exogenous shocks
                   'eps_r': 0, 'eps_z': 0, 'eps_g': 0, 'eps_w': 0, 'eps_p': 0, 'eps_i': 0., 'eps_beta': 0., 'eps_omega': 0., 'eps_sigma': 0., 'eps_tau': 0.,
                   # model parameters
                   'sig_c': 1.5, 'sig_l': 2., 'sigma_z': 1.5, 'tau': 0.2, 
                   'chi0': 0.25, 'chi2': 2, 'phiss': 4,
                   'omega': 0.005, 'phi_pi': 1.5, 'phi_y': 0.125, 'rho': 0.75,
                   'zeta_p': 0.5, 'zeta_w': 0.5,
                   'iota_p': 0.3, 'iota_w': 0.3, 
                   # numerical parameters
                   'nZ': 3, 'nB': 10, 'nA': 16, 'nK': 4,
                   'bmax': 50, 'amax': 4000, 'kmax': 1,
                   'rho_z': 0.966,
                  }

    calibration['pi'] = calibration['pistar']/100
    calibration['i'] = calibration['rstar']/100 # from taylor
    calibration['r'] = (1 + calibration['i']) / (1 + calibration['pi']) - 1 # from fisher

    load_inits = np.load('ressources/init_vals_small_grid.npz')

    unknowns_ss = {
        'beta': float(load_inits['beta']),
        'chi1': float(load_inits['chi1']),
    }
    calibration['Va_init'] = load_inits['Va']
    calibration['Vb_init'] = load_inits['Vb']

    targets_ss = {'asset_mkt': 0., 'B': 'Bh'}
    cali = two_asset_model_ss.solve_steady_state(calibration, unknowns_ss, targets_ss, solver='broyden_custom')
    ss = two_asset_model.steady_state(cali)

    # Transitional Dynamics/Jacobian Calculation
    unknowns = ['r', 'w', 'Y', 'i', 'pi', 'Q', 'K', 'I']
    targets = ['asset_mkt', 'fisher', 'wnkpc', 'taylor', 'nkpc', 'inv', 'val', 'I_res']
    exogenous = ['eps_r', 'eps_z', 'eps_g', 'eps_w', 'eps_p', 'eps_i', 'eps_beta', 'eps_omega', 'eps_sigma', 'eps_tau']

    return two_asset_model_ss, ss, unknowns_ss, targets_ss, two_asset_model, unknowns, targets, exogenous
