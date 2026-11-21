"""Mechanistic equations: cell growth/death balances, metabolite balances with feed boluses, numerical simulation and parameter fitting."""

import numpy as np # numerics
import pandas as pd # tables
from scipy.integrate import solve_ivp # ODE integrator
from scipy.optimize import least_squares # nonlinear least squares

CELL_PARAMS = ["mu_max", "vcd_max", "ivcd_max", "kd0", "kd1"] # cell-model parameters, fixed order
TOL = {"rtol": 1e-6, "atol": 1e-8} # integrator tolerances
FEED_SUBSTRATE = {"feed_substrate": "c_fs", "feed_b": "c_fb"} # feeds carrying substrate -> concentration parameter


# exponential growth: dN/dt = mu N
def rhs_exponential(t, x, p, t_vol, vol):
    return [p["mu"] * x[0]] # constant specific growth rate


# logistic growth: growth slows linearly with cell density
def rhs_logistic(t, x, p, t_vol, vol):
    vcd = x[0] / np.interp(t, t_vol, vol) # viable cell density
    return [p["mu_max"] * (1 - vcd / p["vcd_max"]) * x[0]] # dN/dt


# cell model: growth braked by density and cumulative load, death rising with load
def rhs_cells(t, x, p, t_vol, vol):
    n, dead, ivcd = x # viable cells, dead cells, cumulative load
    vcd = n / np.interp(t, t_vol, vol) # viable cell density
    mu = p["mu_max"] * (1 - vcd / p["vcd_max"] - ivcd / p["ivcd_max"]) # specific growth rate
    kd = p["kd0"] + p["kd1"] * ivcd # specific death rate
    return [(mu - kd) * n, kd * n, vcd] # dN/dt, dD/dt, dIVCD/dt


RHS = {"exponential": rhs_exponential, "logistic": rhs_logistic, "cells": rhs_cells} # model name -> equations


# simulate one batch from day 0 (cells only), return VCD and TCD at t_eval
def simulate(model, batch, p, t_eval):
    t_vol = batch["volume"]["elapsed_time"].to_numpy() # sample times
    vol = batch["volume"]["volume"].to_numpy() # measured volume
    x0 = batch["x0"] # day-0 sample
    n0 = x0["vcd"] * x0["volume"] # total viable cells
    state0 = [n0, x0["dead"] * x0["volume"], 0.0] if model == "cells" else [n0] # initial state
    sol = solve_ivp(RHS[model], (0, t_eval[-1]), state0, t_eval=t_eval, args=(p, t_vol, vol), **TOL) # integrate
    v = np.interp(t_eval, t_vol, vol) # volume at output times
    out = pd.DataFrame({"batch_id": batch["batch_id"], "elapsed_time": t_eval, "vcd": sol.y[0] / v}) # viable density
    if model == "cells": # dead cells simulated
        out["tcd"] = (sol.y[0] + sol.y[1]) / v # total density
    else: # no death: constant viability of day 0
        out["tcd"] = out["vcd"] * x0["tcd"] / x0["vcd"] # total density
    return out


# observed rates per sampling interval: growth, death, cumulative load (adds columns to a samples table)
def observed_rates(samples):
    s = samples.sort_values(["batch_num", "elapsed_time"]).copy() # ordered copy
    g = s.groupby("batch_id") # per batch
    s["cells"] = s["vcd"] * s["volume"] # total viable cells N
    s["dead"] = (s["tcd"] - s["vcd"]) * s["volume"] # total dead cells D
    dt_next = g["elapsed_time"].diff().shift(-1) # time to next sample
    s["mu_obs"] = np.log(s["cells"]).groupby(s["batch_id"]).diff().shift(-1) / dt_next # net growth rate
    n_mean = 0.5 * (s["cells"] + g["cells"].shift(-1)) # mean viable cells in the interval
    s["kd_obs"] = s.groupby("batch_id")["dead"].diff().shift(-1) / (n_mean * dt_next) # death rate
    s["mu_g"] = s["mu_obs"] + s["kd_obs"] # gross growth rate (births only)
    vcd_mid = 0.5 * (s["vcd"] + g["vcd"].shift(1)) # mean VCD in previous interval
    s["ivcd"] = (vcd_mid * g["elapsed_time"].diff()).fillna(0).groupby(s["batch_id"]).cumsum() # trapezoid integral of VCD
    return s


# logistic parameters from observed rates: line mu_obs = mu_max (1 - VCD/VCD_max), growth phase only
def logistic_from_rates(rates, t_max=6.5):
    gr = rates[rates["elapsed_time"] < t_max].dropna(subset=["mu_obs"]) # growth phase
    slope, intercept = np.polyfit(gr["vcd"], gr["mu_obs"], 1) # straight line
    return {"mu_max": float(intercept), "vcd_max": float(-intercept / slope)} # intercept and zero crossing


# cell-model parameters from observed rates: linear fits of gross growth and death
def cells_from_rates(rates):
    fit = rates.dropna(subset=["mu_g", "kd_obs"]) # valid intervals
    A = np.c_[np.ones(len(fit)), fit["vcd"], fit["ivcd"]] # design matrix: 1, VCD, IVCD
    c0, c1, c2 = np.linalg.lstsq(A, fit["mu_g"], rcond=None)[0] # mu_g = c0 + c1 VCD + c2 IVCD
    B = np.c_[np.ones(len(fit)), fit["ivcd"]] # design matrix: 1, IVCD
    kd0, kd1 = np.linalg.lstsq(B, fit["kd_obs"], rcond=None)[0] # kd = kd0 + kd1 IVCD
    return {"mu_max": c0, "vcd_max": -c0 / c1, "ivcd_max": -c0 / c2, "kd0": max(kd0, 1e-4), "kd1": max(kd1, 1e-7)} # rewritten as model parameters


# relative errors of VCD and TCD for many batches (vector for least squares)
def residuals_cells(theta, packages):
    p = dict(zip(CELL_PARAMS, np.exp(theta))) # log-parameters -> parameters
    res = [] # residual pieces
    for b in packages.values(): # each batch
        sim = simulate("cells", b, p, b["t_pred"]) # simulate at sample times
        for c in ["vcd", "tcd"]: # both cell targets
            y = b["y_meas"][c].to_numpy() # measured values
            res.append((sim[c].to_numpy() - y) / y) # relative errors
    return np.concatenate(res)


# joint least-squares fit of the 5 cell parameters on full curves, starting from p0
def fit_cells(packages, p0):
    theta0 = np.log([p0[k] for k in CELL_PARAMS]) # start in log scale
    sol = least_squares(residuals_cells, theta0, args=(packages,), diff_step=1e-3) # optimize
    return dict(zip(CELL_PARAMS, np.exp(sol.x))) # back to parameters


# VCD_max of one batch (other parameters fixed), bounded within a factor e of the global value
def fit_vcd_max(batch, p, sd):
    def residuals(theta): # normalized VCD and TCD errors for ln(VCD_max)
        sim = simulate("cells", batch, {**p, "vcd_max": float(np.exp(theta[0]))}, batch["t_pred"]) # simulate
        return np.r_[(sim["vcd"].to_numpy() - batch["y_meas"]["vcd"].to_numpy()) / sd["vcd"], # VCD errors
                     (sim["tcd"].to_numpy() - batch["y_meas"]["tcd"].to_numpy()) / sd["tcd"]] # TCD errors
    theta0 = np.log([p["vcd_max"]]) # start from the global value
    sol = least_squares(residuals, theta0, bounds=(theta0 - 1, theta0 + 1), diff_step=1e-3, max_nfev=60) # bounded fit
    return float(np.exp(sol.x[0])) # back from log scale


# full model: cells + substrate + byproduct amounts (between feed boluses)
def rhs_full(t, x, p, t_vol, vol):
    n, dead, ivcd, m_s, m_p = x # viable, dead, load, substrate amount, byproduct amount
    v = np.interp(t, t_vol, vol) # volume
    vcd, s, pc = n / v, max(m_s / v, 0.0), max(m_p / v, 0.0) # concentrations
    mu = p["mu_max"] * (1 - vcd / p["vcd_max"] - ivcd / p["ivcd_max"]) # growth rate
    kd = p["kd0"] + p["kd1"] * ivcd # death rate
    qs = (max(mu, 0.0) / p["yxs"] + p["ms"]) * s / (p["ks"] + s) # substrate uptake per cell
    qp = p["alpha"] * max(mu, 0.0) - p["beta"] * pc # byproduct production - uptake per cell
    return [(mu - kd) * n, kd * n, vcd, -qs * n, qp * n] # derivatives


# simulate cells + metabolites, integrating between feed boluses that add substrate
def simulate_full(batch, p, t_eval):
    t_vol = batch["volume"]["elapsed_time"].to_numpy() # sample times
    vol = batch["volume"]["volume"].to_numpy() # measured volume
    x0, v0 = batch["x0"], batch["x0"]["volume"] # day-0 sample and volume
    x = np.array([x0["vcd"] * v0, x0["dead"] * v0, 0.0, x0["substrate"] * v0, x0["byproduct"] * v0]) # initial amounts
    ev = batch["feed_events"] # feeding plan
    ev = ev[ev["feed"].isin(FEED_SUBSTRATE) & (ev["elapsed_time"] < t_eval[-1])] # substrate-carrying feeds
    bounds = np.r_[0.0, ev["elapsed_time"].to_numpy(), t_eval[-1]] # integration segments
    out = np.zeros((5, len(t_eval))) # results
    for k in range(len(bounds) - 1): # each segment
        a, b = bounds[k], bounds[k + 1] # segment limits
        if b > a: # non-empty segment
            sol = solve_ivp(rhs_full, (a, b), x, args=(p, t_vol, vol), dense_output=True, **TOL) # integrate
            idx = (t_eval >= a) & ((t_eval <= b) if k == len(bounds) - 2 else (t_eval < b)) # outputs in segment
            if idx.any(): # outputs present
                out[:, idx] = sol.sol(t_eval[idx]) # interpolated solution
            x = sol.y[:, -1].copy() # state at segment end
        if k < len(ev): # feed bolus at segment end
            e = ev.iloc[k] # the addition
            x[3] += p[FEED_SUBSTRATE[e["feed"]]] * e["amount"] # substrate added = concentration x amount
    v = np.interp(t_eval, t_vol, vol) # volume at outputs
    return pd.DataFrame({"batch_id": batch["batch_id"], "elapsed_time": t_eval, "vcd": out[0] / v, "tcd": (out[0] + out[1]) / v,
                         "substrate": out[3] / v, "byproduct": out[4] / v}) # concentrations


# fit selected metabolite parameters on one target (others fixed), normalized errors
def fit_metabolite(packages, p_fixed, p0, names, target, sd):
    def residuals(theta): # normalized errors of the target
        p = {**p_fixed, **dict(zip(names, np.exp(theta)))} # parameters
        return np.concatenate([(simulate_full(b, p, b["t_pred"])[target].to_numpy() - b["y_meas"][target].to_numpy()) / sd for b in packages.values()]) # all batches
    sol = least_squares(residuals, np.log([p0[k] for k in names]), diff_step=1e-3, max_nfev=40) # optimize
    return dict(zip(names, np.exp(sol.x))) # fitted parameters
