"""MAPE-oriented features: look-ahead feed doses (controller signals) and volume residuals (hidden base additions of the pH controller)."""

import numpy as np # numerics
import pandas as pd # tables

FEEDS = ["feed_substrate", "feed_b", "feed_c", "feed_d"] # feed columns
SUB_NEXT = ["feed_substrate_next", "feed_substrate_tnext"] # next substrate dose and time to it
B_NEXT = ["feed_b_next", "feed_b_n24"] # next feed B dose and feed B in the next 24 h


# sampling times (day 0 included) and volume at those times for one batch
def _times_volume(data, b):
    pk = data.pk[b] # batch package
    ts = np.r_[0.0, pk["t_pred"]] # day 0 + prediction times
    return ts, np.interp(ts, pk["volume"]["elapsed_time"], pk["volume"]["volume"]) # times, volume


# look-ahead features: for each feed, next dose, time to it, doses in the next 24 h and 2 h (all per volume)
def lookahead_features(data):
    rows = [] # one row per sample after day 0
    for b in data.labeled + data.to_predict: # all batches
        f = data.feeds[data.feeds["batch_id"] == b] # feeds of the batch
        ts, V = _times_volume(data, b) # times and volume
        for t, v in zip(ts[1:], V[1:]): # each sample after day 0
            r = {"batch_id": b, "elapsed_time": t} # identifiers
            for fe in FEEDS: # each feed
                g = f[(f["elapsed_time"] > t) & (f[fe] > 0)] # later additions of this feed
                r[fe + "_next"] = g[fe].iloc[0] / v if len(g) else 0.0 # next dose per volume
                r[fe + "_tnext"] = g["elapsed_time"].iloc[0] - t if len(g) else 14 - t # time to the next dose
                r[fe + "_n24"] = f.loc[(f["elapsed_time"] > t) & (f["elapsed_time"] <= t + 1), fe].sum() / v # dose in the next 24 h
                r[fe + "_n2h"] = f.loc[(f["elapsed_time"] > t) & (f["elapsed_time"] <= t + 2 / 24), fe].sum() / v # dose in the next 2 h
            rows.append(r) # keep
    return pd.DataFrame(rows)


# volume residuals: measured volume minus initial volume minus all listed feeds (base added for pH control)
def volume_features(data):
    rows = [] # one row per sample after day 0
    for b in data.labeled + data.to_predict: # all batches
        f = data.feeds[data.feeds["batch_id"] == b] # feeds of the batch
        ts, V = _times_volume(data, b) # times and volume
        for i in range(1, len(ts)): # each sample after day 0
            t = ts[i] # sample time
            fed = sum(f.loc[f["elapsed_time"] < t, fe].sum() for fe in FEEDS) # all feeds added so far
            fed_i = sum(f.loc[(f["elapsed_time"] < t) & (f["elapsed_time"] >= ts[i - 1]), fe].sum() for fe in FEEDS) # feeds of the last interval
            r = {"batch_id": b, "elapsed_time": t, "V_t": V[i], # identifiers and volume
                 "Vres": V[i] - V[0] - fed, # cumulative volume not explained by feeds
                 "dVres": (V[i] - V[i - 1]) - fed_i} # same, last interval only
            for fe in FEEDS: # cumulative feeds per volume
                r[fe + "_cumV"] = f.loc[f["elapsed_time"] < t, fe].sum() / V[i] # dose so far / volume
            rows.append(r) # keep
    return pd.DataFrame(rows)


# volume-residual profile of each batch: change in every sampling interval, minus the typical change of that interval
def volume_profile(data, n=13):
    prof = {} # batch -> profile
    for b in data.labeled + data.to_predict: # all batches
        f = data.feeds[data.feeds["batch_id"] == b] # feeds of the batch
        ts, V = _times_volume(data, b) # times and volume
        fed = np.array([sum(f.loc[f["elapsed_time"] < t, fe].sum() for fe in FEEDS) for t in ts]) # feeds added up to each time
        prof[b] = np.diff(V - V[0] - fed)[:n] # interval changes of the residual
    P = pd.DataFrame(prof, index=[f"dv{k}" for k in range(1, n + 1)]).T # batches x intervals
    return P - P.loc[data.labeled].median() # deviation from the typical interval


# add all MAPE features to the data object; returns the new column names by group
def add_mape_features(data):
    la, vr, dv = lookahead_features(data), volume_features(data), volume_profile(data) # build features
    data.feat_day = data.feat_day.merge(la, on=["batch_id", "elapsed_time"], how="outer") # per-sample look-ahead
    data.feat_day = data.feat_day.merge(vr, on=["batch_id", "elapsed_time"], how="left") # per-sample volume features
    data.feat = data.feat.join(dv) # per-batch volume profile
    return {"LA": [c for c in la.columns if c not in ("batch_id", "elapsed_time")], # look-ahead columns
            "VR": [c for c in vr.columns if c not in ("batch_id", "elapsed_time")], # volume columns
            "DV": list(dv.columns)} # profile columns
