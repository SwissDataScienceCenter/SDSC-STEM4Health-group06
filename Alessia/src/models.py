"""Prediction models with a common interface: fit(data, train_ids) learns from labeled batches, predict(data, ids) returns VCD, TCD, substrate, byproduct at the sample times (day >= 1)."""

import numpy as np # numerics
import pandas as pd # tables
from sklearn.linear_model import RidgeCV # ridge regression with automatic penalty
from sklearn.pipeline import make_pipeline # chain of steps
from sklearn.preprocessing import StandardScaler # features to mean 0, std 1

from . import mechanistic as mech # equations and fits
from .data_handling import TARGET_COLS # target names

BATCH_FEATS = ["vcd0", "byproduct0", "substrate0", "volume0", "fb_tot", "fc_n", "fs_tot", "fs_t1"] # batch features
SHAPE_FEATS = ["fb_ramp", "fb_plateau", "fb_late", "fc_early", "fc_t1", "viab0"] # feed-shape features
DAY_FEATS = ["fs_24h", "fb_24h", "fs_cum"] # per-sample feed features


# ridge regression on standardized inputs, penalty chosen by internal cross-validation
def ridge():
    return make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 30))) # scaler + ridge


# mean value of each target per day, from the samples of the training batches
def mean_curve(data, train_ids):
    post = data.samples(train_ids) # training samples
    post = post[post["elapsed_time"] > 0] # after day 0
    return post.groupby(post["elapsed_time"].round())[TARGET_COLS].mean() # one row per day


# fill selected targets of a prediction table with the mean curve of their day
def fill_mean(pred, curve, cols):
    day = pred["elapsed_time"].round() # day index
    for c in cols: # each target
        pred[c] = curve.loc[day, c].to_numpy() # mean value of that day
    return pred


# M0: same mean curve for every batch
class MeanCurve:
    name = "M0 Mean curve" # display name

    # learn the mean curve
    def fit(self, data, train_ids):
        self.curve = mean_curve(data, train_ids) # mean per day
        return self

    # predict the mean curve at each batch's sample times
    def predict(self, data, ids):
        pred = pd.concat([pd.DataFrame({"batch_id": b, "elapsed_time": data.pk[b]["t_pred"]}) for b in ids], ignore_index=True) # rows to predict
        return fill_mean(pred, self.curve, TARGET_COLS) # all targets from the mean curve


# M1: logistic growth from day 0 (density brake), no death; metabolites = mean curve
class LogisticModel:
    name = "M1 Logistic growth" # display name

    # estimate mu_max and VCD_max from observed growth rates
    def fit(self, data, train_ids):
        self.rates = mech.observed_rates(data.samples(train_ids)) # rates per interval
        self.p = mech.logistic_from_rates(self.rates) # logistic parameters
        self.curve = mean_curve(data, train_ids) # mean curve for metabolites
        return self

    # simulate each batch, metabolites from the mean curve
    def predict(self, data, ids):
        pred = pd.concat([mech.simulate("logistic", data.pk[b], self.p, data.pk[b]["t_pred"]) for b in ids], ignore_index=True) # VCD, TCD
        return fill_mean(pred, self.curve, ["substrate", "byproduct"]) # metabolites


# M2 / M3: cell model (density + cumulative brake, death); parameters from daily rates (M2) or joint fit (M3)
class CellModel:
    # joint=False: parameters from daily rates; joint=True: least squares on full curves
    def __init__(self, joint=True):
        self.joint = joint # fitting mode
        self.name = "M3 Cell model (joint fit)" if joint else "M2 Cell model (daily rates)" # display name

    # estimate the 5 cell parameters
    def fit(self, data, train_ids):
        self.rates = mech.observed_rates(data.samples(train_ids)) # rates per interval
        self.p_rates = mech.cells_from_rates(self.rates) # parameters from linear fits of the rates
        self.p = mech.fit_cells(data.packages(train_ids), self.p_rates) if self.joint else self.p_rates # final parameters
        self.curve = mean_curve(data, train_ids) # mean curve for metabolites
        return self

    # simulate each batch, metabolites from the mean curve
    def predict(self, data, ids):
        pred = pd.concat([mech.simulate("cells", data.pk[b], self.p, data.pk[b]["t_pred"]) for b in ids], ignore_index=True) # VCD, TCD
        return fill_mean(pred, self.curve, ["substrate", "byproduct"]) # metabolites


# M4: cell model + substrate and byproduct mass balances with feed boluses
class CellMetaboliteModel:
    name = "M4 Cells + metabolite balances" # display name
    SUB0 = {"yxs": 3.4, "ms": 0.38, "ks": 2.1, "c_fs": 160.0, "c_fb": 138.0} # substrate start values
    BYP0 = {"alpha": 0.5, "beta": 0.05} # byproduct start values

    # fit cells (joint), then substrate and byproduct parameters on their own target
    def fit(self, data, train_ids):
        cells = CellModel(joint=True).fit(data, train_ids) # cell parameters
        post = data.samples(train_ids) # training samples
        sd = post[post["elapsed_time"] > 0][TARGET_COLS].std() # scale per target
        pk = data.packages(train_ids) # training packages
        p_sub = mech.fit_metabolite(pk, {**cells.p, **self.BYP0}, self.SUB0, list(self.SUB0), "substrate", sd["substrate"]) # substrate fit
        p_byp = mech.fit_metabolite(pk, {**cells.p, **p_sub}, self.BYP0, list(self.BYP0), "byproduct", sd["byproduct"]) # byproduct fit
        self.p = {**cells.p, **p_sub, **p_byp} # all parameters
        return self

    # simulate cells and metabolites for each batch
    def predict(self, data, ids):
        return pd.concat([mech.simulate_full(data.pk[b], self.p, data.pk[b]["t_pred"]) for b in ids], ignore_index=True) # all targets


# M6 base: cell model with batch-specific capacity VCD_max predicted from batch features
class GreyBoxCapacity:
    name = "Grey-box base" # display name (used inside M6/M7)

    # global joint fit, VCD_max per training batch, regression ln(VCD_max) ~ batch features
    def fit(self, data, train_ids):
        self.cells = CellModel(joint=True).fit(data, train_ids) # global parameters
        post = data.samples(train_ids) # training samples
        sd = post[post["elapsed_time"] > 0][TARGET_COLS].std() # scale per target
        self.vcd_max_fit = pd.Series({b: mech.fit_vcd_max(data.pk[b], self.cells.p, sd) for b in train_ids}) # per-batch capacity
        self.vmax_model = ridge().fit(data.feat.loc[train_ids, BATCH_FEATS], np.log(self.vcd_max_fit)) # capacity from features
        return self

    # predicted capacity of each batch
    def vcd_max(self, data, ids):
        return pd.Series(np.exp(self.vmax_model.predict(data.feat.loc[ids, BATCH_FEATS])), index=ids) # back from log

    # simulate each batch with its predicted capacity, metabolites from the mean curve
    def predict(self, data, ids):
        vm = self.vcd_max(data, ids) # capacities
        pred = pd.concat([mech.simulate("cells", data.pk[b], {**self.cells.p, "vcd_max": vm[b]}, data.pk[b]["t_pred"]) for b in ids], ignore_index=True) # VCD, TCD
        return fill_mean(pred, self.cells.curve, ["substrate", "byproduct"]) # metabolites


# M5 / M5b / M6 / M7: any base model + ridge correction of its residuals, one model per target and day
class RidgeCorrection:
    # base: a model instance; feats: target -> list of feature names; name: display name
    def __init__(self, base, feats, name):
        self.base, self.feats, self.name = base, feats, name # store settings

    # add batch, feed-shape and per-sample features to a table of (batch, time) rows
    @staticmethod
    def features(data, table):
        out = table.merge(data.feat, left_on="batch_id", right_index=True, how="left") # batch features
        out = out.merge(data.feat_shape, left_on="batch_id", right_index=True, how="left") # feed-shape features
        out = out.merge(data.feat_day, on=["batch_id", "elapsed_time"], how="left") # per-sample features
        out["day"] = out["elapsed_time"].round() # day index
        return out

    # fit the base, compute its residuals on the training batches, learn residual ~ features
    def fit(self, data, train_ids):
        self.base.fit(data, train_ids) # base model
        pred = self.base.predict(data, train_ids) # base predictions, training batches
        meas = data.samples(train_ids)[["batch_id", "elapsed_time"] + TARGET_COLS] # measurements
        res = pred.merge(meas, on=["batch_id", "elapsed_time"], suffixes=("_pred", "")) # side by side
        X = self.features(data, res) # with features
        self.models = {} # (target, day) -> ridge
        for c in TARGET_COLS: # each target
            for d, rows in X.groupby("day"): # each day
                self.models[(c, d)] = ridge().fit(rows[self.feats[c]], rows[c] - rows[c + "_pred"]) # residual = measured - base
        return self

    # base prediction + predicted residual
    def predict(self, data, ids):
        out = self.base.predict(data, ids).reset_index(drop=True) # base predictions
        X = self.features(data, out) # features of every row
        for (c, d), m in self.models.items(): # each target and day
            idx = (X["day"] == d).to_numpy() # rows of that day
            if idx.any(): # day present
                out.loc[idx, c] = out.loc[idx, c].to_numpy() + m.predict(X.loc[idx, self.feats[c]]) # add correction
        return out

    # mean standardized coefficient over days: effect of +1 std of each feature on each target
    def coefficients(self):
        rows = {} # (target, day) -> coefficients
        for (c, d), m in self.models.items(): # each model
            rows[(c, d)] = pd.Series(m[-1].coef_, index=self.feats[c]) # coefficients of the ridge step
        coef = pd.DataFrame(rows) # features x (target, day)
        return coef.T.groupby(level=0).mean().T[TARGET_COLS] # mean over days, features x targets


# feature sets per target used by the correction models
FEATS_BASIC = {c: BATCH_FEATS + (DAY_FEATS if c == "substrate" else []) for c in TARGET_COLS} # M5, M5b, M6
FEATS_SHAPE = {c: BATCH_FEATS + (SHAPE_FEATS + DAY_FEATS if c == "substrate" else []) for c in TARGET_COLS} # M7
