"""Evaluation: team benchmark split and normalized-RMSE score, cross-validation over all labeled batches, submission files in the template format."""

import numpy as np # numerics
import pandas as pd # tables
from sklearn.model_selection import KFold # batch splits for cross-validation

from .data_handling import DATA_DIR, TARGET_COLS # data folder and target names

TEST_BATCHES = ["Batch2", "Batch3", "Batch11", "Batch18", "Batch19", "Batch20", "Batch23", "Batch28", "Batch39", "Batch42"] # team hold-out
KEY = ["batch_id", "elapsed_time"] # columns aligning truth and predictions


# labeled batches split into benchmark training (39) and test (10)
def split_benchmark(data):
    train = [b for b in data.labeled if b not in TEST_BATCHES] # training ids
    return train, list(TEST_BATCHES)


# scale of each target: std of the post-start samples of the training batches
def target_sd(data, train_ids):
    post = data.samples(train_ids) # training samples
    return post[post["elapsed_time"] > 0][TARGET_COLS].std() # one std per target


# team score: errors / std, squared, mean per batch and target, square root of the mean over all batch-target pairs
def team_score(pred, data, test_ids, sd):
    truth = data.samples(test_ids) # test samples
    truth = truth[truth["elapsed_time"] > 0] # scored rows (day >= 1)
    m = truth.merge(pred, on=KEY, suffixes=("", "_pred")) # align truth and prediction
    per = pd.DataFrame({c: (((m[c + "_pred"] - m[c]) / sd[c]) ** 2).groupby(m["batch_id"]).mean() for c in TARGET_COLS}) # m(batch, target)
    return float(np.sqrt(per.to_numpy().mean())), np.sqrt(per.mean()), np.sqrt(per) # overall, per target, per batch and target


# fit on the 39 benchmark-training batches, score on the 10 test batches
def benchmark(make_model, data):
    train, test = split_benchmark(data) # split
    model = make_model().fit(data, train) # fit on training batches
    sd = target_sd(data, train) # scales
    pred_all = model.predict(data, data.labeled) # predictions for all 49 labeled batches (training ones are in-sample)
    total, per_target, per_batch = team_score(pred_all[pred_all["batch_id"].isin(test)], data, test, sd) # score on test batches
    return {"model": model, "pred": pred_all, "score": total, "per_target": per_target, "per_batch": per_batch}


# k-fold cross-validation over all 49 labeled batches (each batch is test once)
def cross_validate(make_model, data, k=5, seed=0):
    ids = np.array(data.labeled) # labeled batches
    scores, per_target = [], [] # results per fold
    for tr, te in KFold(k, shuffle=True, random_state=seed).split(ids): # folds
        model = make_model().fit(data, list(ids[tr])) # fit on k-1 folds
        sd = target_sd(data, list(ids[tr])) # scales from the training folds
        total, pt, _ = team_score(model.predict(data, list(ids[te])), data, list(ids[te]), sd) # score held-out fold
        scores.append(total) # overall score
        per_target.append(pt) # per-target scores
    return float(np.mean(scores)), pd.concat(per_target, axis=1).mean(axis=1) # mean over folds


# fit on all 49 labeled batches, predict the 12 batches, fill the template and save it
def write_submission(make_model, data, path):
    model = make_model().fit(data, data.labeled) # final fit on all labeled batches
    pred = model.predict(data, data.to_predict).sort_values(KEY) # predictions for the 12 batches
    pred["k"] = pred.groupby("batch_id").cumcount() # sample index within batch
    tmpl = pd.read_csv(DATA_DIR / "bioreaction_template.csv") # submission template
    tmpl["k"] = tmpl.groupby("batch_id").cumcount() # sample index within batch (template times have fewer decimals)
    sub = tmpl[["batch_id", "elapsed_time", "k"]].merge(pred[["batch_id", "k", "elapsed_time"] + TARGET_COLS], on=["batch_id", "k"], suffixes=("", "_pred")) # match rows by order
    assert (sub["elapsed_time"] - sub["elapsed_time_pred"]).abs().max() < 1e-6, "template times do not match" # safety check
    sub = sub[["batch_id", "elapsed_time"] + TARGET_COLS] # template columns
    assert len(sub) == len(tmpl) and not sub.isna().any().any(), "incomplete submission" # safety check
    sub.to_csv(path, index=False) # save
    return sub


# table of benchmark results: per-target and overall score of each model
def results_table(results):
    return pd.DataFrame({k: {**r["per_target"].to_dict(), "score": r["score"]} for k, r in results.items()}).T.round(3) # models x (targets + score)


# benchmark in MAPE: fit on the 39 training batches, MAPE on the 10 team test batches
def benchmark_mape(make_model, data):
    from .mape_models import mape # MAPE metric
    train, test = split_benchmark(data) # team split
    model = make_model().fit(data, train) # fit on training batches
    pred = model.predict(data, data.labeled) # predictions for all 49 labeled batches (training ones are in-sample)
    return {"model": model, "pred": pred, "mape": mape(pred[pred["batch_id"].isin(test)], data, test)} # model, predictions, test MAPE
