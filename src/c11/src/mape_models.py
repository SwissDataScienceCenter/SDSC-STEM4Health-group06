"""MAPE-oriented models: corrections and tree models in log space (error on log y = relative error), and a per-target geometric-mean ensemble."""

import numpy as np # numerics
import pandas as pd # tables
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor # tree ensembles

from .data_handling import TARGET_COLS # target names
from .models import RidgeCorrection, ridge # correction base class, ridge regression

FLOOR = 1e-3 # smallest value allowed before taking logs
KEY = ["batch_id", "elapsed_time"] # alignment columns


# mean absolute percentage error per target and its mean over targets
def mape(pred, data, ids):
    truth = data.samples(ids) # measurements
    truth = truth[truth["elapsed_time"] > 0] # scored rows (day >= 1)
    m = truth[KEY + TARGET_COLS].merge(pred, on=KEY, suffixes=("", "_pred")) # align
    out = {c: 100 * ((m[c + "_pred"] - m[c]).abs() / m[c]).mean() for c in TARGET_COLS} # MAPE per target
    out["mean MAPE"] = float(np.mean(list(out.values()))) # leaderboard score
    return pd.Series(out)


# ridge correction learned in log space: log(y) = log(base) + f(features), one ridge per target and day
class LogCorrection(RidgeCorrection):
    # base model + log residuals of the training batches
    def fit(self, data, train_ids):
        self.base.fit(data, train_ids) # base model
        pred = self.base.predict(data, train_ids) # base predictions
        meas = data.samples(train_ids)[KEY + TARGET_COLS] # measurements
        X = self.features(data, pred.merge(meas, on=KEY, suffixes=("_pred", ""))) # side by side, with features
        self.models = {} # (target, day) -> ridge
        for c in TARGET_COLS: # each target
            for d, rows in X.groupby("day"): # each day
                self.models[(c, d)] = ridge().fit(rows[self.feats[c]], np.log(rows[c]) - np.log(rows[c + "_pred"].clip(lower=FLOOR))) # log residual
        return self

    # base prediction times exp(predicted log residual)
    def predict(self, data, ids):
        out = self.base.predict(data, ids).reset_index(drop=True) # base predictions
        X = self.features(data, out) # features
        for (c, d), m in self.models.items(): # each target and day
            idx = (X["day"] == d).to_numpy() # rows of that day
            if idx.any(): # day present
                out.loc[idx, c] = out.loc[idx, c].clip(lower=FLOOR).to_numpy() * np.exp(m.predict(X.loc[idx, self.feats[c]])) # multiplicative correction
        return out


# tree model in log space: log(y) - mean log curve of the day, learned on all days together (time is a feature)
class TreeLog:
    # feats: target -> feature list; kind: "et" (ExtraTrees) or "rf" (random forest)
    def __init__(self, feats, kind="et", leaf=3, mf=0.5, n=300):
        self.feats, self.kind, self.leaf, self.mf, self.n = feats, kind, leaf, mf, n # settings
        self.name = "trees" # display name

    # feature table of measured samples (with targets) or of the prediction times
    def table(self, data, ids, with_y):
        if with_y: # training rows
            rows = data.samples(ids) # measurements
            rows = rows[rows["elapsed_time"] > 0][KEY + TARGET_COLS] # after day 0
        else: # prediction rows
            rows = pd.concat([pd.DataFrame({"batch_id": b, "elapsed_time": data.pk[b]["t_pred"]}) for b in ids]) # requested times
        return RidgeCorrection.features(data, rows) # add features

    # mean log curve per day + one tree ensemble per target on the log deviation
    def fit(self, data, ids):
        X = self.table(data, ids, True) # training table
        self.gm = np.log(X[TARGET_COLS]).groupby(X["day"]).mean() # mean log value per day
        M = ExtraTreesRegressor if self.kind == "et" else RandomForestRegressor # tree type
        self.m = {} # target -> trees
        for c in TARGET_COLS: # each target
            y = np.log(X[c]) - self.gm.loc[X["day"], c].to_numpy() # log deviation from the day mean
            self.m[c] = M(n_estimators=self.n, min_samples_leaf=self.leaf, max_features=self.mf, random_state=0, n_jobs=-1).fit(X[self.feats[c] + ["elapsed_time"]], y) # fit
        return self

    # exp(mean log curve + predicted deviation)
    def predict(self, data, ids):
        X = self.table(data, ids, False).reset_index(drop=True) # prediction table
        out = X[KEY].copy() # identifiers
        for c in TARGET_COLS: # each target
            out[c] = np.exp(self.gm.reindex(X["day"])[c].to_numpy() + self.m[c].predict(X[self.feats[c] + ["elapsed_time"]])) # back to concentration
        return out


# per-target ensemble: each target is the geometric mean of its own set of member models
class PerTargetEnsemble:
    # makers: name -> function returning an unfitted model; plan: target -> list of member names
    def __init__(self, makers, plan, name):
        self.makers, self.plan, self.name = makers, plan, name # settings

    # fit only the members used in the plan
    def fit(self, data, ids):
        used = sorted(set(sum(self.plan.values(), []))) # member names
        self.members = {n: self.makers[n]().fit(data, ids) for n in used} # fitted members
        return self

    # geometric mean of the members of each target; TCD at least VCD
    def predict(self, data, ids):
        P = {n: m.predict(data, ids).sort_values(KEY).reset_index(drop=True) for n, m in self.members.items()} # aligned predictions
        out = next(iter(P.values()))[KEY].copy() # identifiers
        for c, names in self.plan.items(): # each target
            out[c] = np.exp(np.mean([np.log(P[n][c].clip(lower=FLOOR).to_numpy()) for n in names], axis=0)) # geometric mean
        out["tcd"] = np.maximum(out["tcd"], out["vcd"]) # physical constraint
        return out

    # predictions of every member (for inspection)
    def member_predictions(self, data, ids):
        return {n: m.predict(data, ids) for n, m in self.members.items()} # name -> predictions
