"""Learned-rate hybrid (Plan C): mass balances of cells, substrate and byproduct, with specific rates learned from data as functions of state, feeds and batch features."""

import numpy as np # numerics
import pandas as pd # tables
from scipy.optimize import nnls # non-negative least squares
from sklearn.preprocessing import StandardScaler # features to mean 0, std 1

from .data_handling import TARGET_COLS # target names
from .models import BATCH_FEATS, ridge # batch features, ridge regression

FEEDS = ["feed_substrate", "feed_b", "feed_c", "feed_d"] # feed columns
CARRIERS = ["feed_substrate", "feed_b", "feed_d"] # feeds that may contain substrate
STATE_INPUTS = ["vcd", "ivcd", "S", "P", "viab", "t"] # state at the start of an interval
FEED_INPUTS = [f + "_v" for f in FEEDS] # feeds added in the interval, per volume
RATES = ["mu", "kd", "rP", "rS"] # learned rates


# amount of each feed added in every interval [t_k, t_k+1) between consecutive times
def interval_feeds(feed_events, t):
    ev = feed_events # feeding plan (time, feed, amount)
    out = np.zeros((len(t) - 1, len(FEEDS))) # intervals x feeds
    for j, f in enumerate(FEEDS): # each feed
        e = ev[ev["feed"] == f] # its additions
        k = np.searchsorted(t, e["elapsed_time"].to_numpy(), side="right") - 1 # interval index of each addition
        ok = (k >= 0) & (k < len(t) - 1) # inside the time range
        np.add.at(out[:, j], k[ok], e["amount"].to_numpy()[ok]) # sum per interval
    return out


# one row per interval between two samples: state at the start, feeds, observed rates
def rate_table(data, ids):
    rows = [] # one dict per interval
    for b in ids: # each batch
        m = data.samples([b]).sort_values("elapsed_time") # its samples
        t, V = m["elapsed_time"].to_numpy(), m["volume"].to_numpy() # times and volumes
        vcd, tcd, S, P = (m[c].to_numpy() for c in TARGET_COLS) # measured concentrations
        N, D = vcd * V, (tcd - vcd) * V # viable and dead cells (amounts)
        ivcd = np.r_[0.0, np.cumsum(0.5 * (vcd[1:] + vcd[:-1]) * np.diff(t))] # integral of VCD (trapezoid)
        F = interval_feeds(data.pk[b]["feed_events"], t) # feeds per interval
        for k in range(len(t) - 1): # each interval
            dt, Vm, Nm = t[k + 1] - t[k], 0.5 * (V[k] + V[k + 1]), 0.5 * (N[k] + N[k + 1]) # length, mean volume, mean cells
            rows.append({"batch_id": b, "t": t[k], "dt": dt, "Vm": Vm, # identifiers and sizes
                         "vcd": vcd[k], "ivcd": ivcd[k], "S": S[k], "P": P[k], "viab": vcd[k] / tcd[k], # state at the start
                         **{f: F[k, j] for j, f in enumerate(FEEDS)}, # feeds added (amounts)
                         **{f + "_v": F[k, j] / V[k] for j, f in enumerate(FEEDS)}, # feeds added per volume
                         "mu": np.log(N[k + 1] / N[k]) / dt, # net growth rate [1/day]
                         "kd": (D[k + 1] - D[k]) / (Nm * dt), # death rate [1/day]
                         "rP": (P[k + 1] * V[k + 1] - P[k] * V[k]) / (Vm * dt), # byproduct rate per volume [g/L/day]
                         "dAS": S[k + 1] * V[k + 1] - S[k] * V[k]}) # change of substrate amount
    return pd.DataFrame(rows)


# substrate content of each carrier feed (>= 0) from the substrate balance of all intervals
def feed_content(table):
    phi = StandardScaler().fit_transform(table[STATE_INPUTS]) # standardized state
    B = -(table["Vm"] * table["dt"]).to_numpy()[:, None] * np.c_[np.ones(len(table)), phi] # consumption = Vm dt (a0 + a.state)
    A = np.c_[table[CARRIERS].to_numpy(), B, -B] # carriers (>= 0) + consumption terms (free sign as +/- pair)
    sol = nnls(A, table["dAS"].to_numpy(), maxiter=5000)[0] # dAS = sum c_j F_j - consumption
    return {**dict(zip(CARRIERS, sol[:len(CARRIERS)])), "feed_c": 0.0} # content per feed (feed C carries none)


# learned-rate model: rates = ridge(state, feeds [, batch features]), simulated interval by interval from day 0
class LearnedRates:
    # batch: add batch features to the rate inputs; name: display name
    def __init__(self, batch=False, name=None):
        self.batch = batch # use batch features?
        self.name = name or ("C2 Learned rates + batch features" if batch else "C1 Learned rates") # display name

    # input matrix of the rate models
    def inputs(self, df):
        X = df[STATE_INPUTS + FEED_INPUTS].reset_index(drop=True) # state and feeds
        if self.batch: # add batch features
            X = pd.concat([X, self.feat.loc[df["batch_id"], BATCH_FEATS].reset_index(drop=True)], axis=1) # batch features
        return X

    # observed rates of the training batches -> feed content, substrate uptake, one ridge per rate
    def fit(self, data, train_ids):
        self.feat = data.feat # batch feature table
        tab = rate_table(data, train_ids) # intervals of the training batches
        self.content = feed_content(tab) # substrate per unit of feed
        tab["rS"] = (sum(self.content[f] * tab[f] for f in FEEDS) - tab["dAS"]) / (tab["Vm"] * tab["dt"]) # substrate uptake per volume
        X = self.inputs(tab) # inputs
        self.lo, self.hi = X.quantile(0.01), X.quantile(0.99) # input range seen in training
        self.models = {r: ridge().fit(X, tab[r]) for r in RATES} # one ridge per rate
        self.table = tab # keep for diagnostics
        return self

    # simulate one batch from its day-0 sample, interval by interval
    def simulate(self, data, b):
        pk = data.pk[b] # batch package
        x0, t = pk["x0"], np.r_[0.0, pk["t_pred"]] # day-0 state, times
        V = np.interp(t, pk["volume"]["elapsed_time"], pk["volume"]["volume"]) # volume at the times
        F = interval_feeds(pk["feed_events"], t) # feeds per interval
        N, D, ivcd = x0["vcd"] * V[0], x0["dead"] * V[0], 0.0 # viable, dead, integral of VCD
        AS, AP = x0["substrate"] * V[0], x0["byproduct"] * V[0] # substrate and byproduct amounts
        out = [] # predicted rows
        for k in range(len(t) - 1): # each interval
            dt, Vm, vcd = t[k + 1] - t[k], 0.5 * (V[k] + V[k + 1]), N / V[k] # length, mean volume, VCD at start
            state = pd.DataFrame([{"batch_id": b, "vcd": vcd, "ivcd": ivcd, "S": AS / V[k], "P": AP / V[k], "viab": N / (N + D), "t": t[k], # state
                                   **{f + "_v": F[k, j] / V[k] for j, f in enumerate(FEEDS)}}]) # feeds per volume
            X = self.inputs(state).clip(self.lo, self.hi, axis=1) # inputs kept in the training range
            mu, kd, rP, rS = (float(self.models[r].predict(X)[0]) for r in RATES) # predicted rates
            N1 = N * np.exp(mu * dt) # viable cells at the end
            Nm = 0.5 * (N + N1) # mean viable cells
            D += max(kd, 0.0) * Nm * dt # dead cells
            ivcd += 0.5 * (vcd + N1 / V[k + 1]) * dt # integral of VCD
            AP = max(AP + rP * Vm * dt, 0.0) # byproduct amount
            AS = max(AS + sum(self.content[f] * F[k, j] for j, f in enumerate(FEEDS)) - rS * Vm * dt, 0.0) # substrate: feeds - uptake
            N = N1 # update viable cells
            out.append({"batch_id": b, "elapsed_time": t[k + 1], "vcd": N / V[k + 1], "tcd": (N + D) / V[k + 1], # cells
                        "substrate": AS / V[k + 1], "byproduct": AP / V[k + 1]}) # metabolites
        return out

    # simulate a list of batches
    def predict(self, data, ids):
        return pd.DataFrame([row for b in ids for row in self.simulate(data, b)]) # all rows

    # standardized coefficients of each rate model: effect of +1 std of each input
    def coefficients(self):
        names = self.models["mu"][0].get_feature_names_out() # input names
        return pd.DataFrame({r: self.models[r][-1].coef_ for r in RATES}, index=names) # inputs x rates


# average of the predictions of several models
class Ensemble:
    # makers: functions returning model instances; name: display name
    def __init__(self, makers, name):
        self.makers, self.name = makers, name # store settings

    # fit every member
    def fit(self, data, train_ids):
        self.members = [m().fit(data, train_ids) for m in self.makers] # fitted members
        return self

    # mean of the members' predictions
    def predict(self, data, ids):
        preds = [m.predict(data, ids).sort_values(["batch_id", "elapsed_time"]).reset_index(drop=True) for m in self.members] # aligned predictions
        out = preds[0].copy() # table layout
        out[TARGET_COLS] = sum(p[TARGET_COLS].to_numpy() for p in preds) / len(preds) # mean
        return out
