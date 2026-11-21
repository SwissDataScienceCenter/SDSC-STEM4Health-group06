"""Build a self-contained HTML data viewer for Project 1 (Decode the Bioreactor).

Usage:  python build_viewer.py
Output: ../data_viewer.html  (open it in a browser; works offline)
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly

HERE = Path(__file__).resolve().parent
DATA_FILE = Path("Project 1 - DTB/bioreaction_remaining.csv")
SEARCH_PATHS = (
    root / directory / DATA_FILE
    for directory in ("input-data", "init_data")
    for root in (HERE, *HERE.parents)
)
DATA_PATH = next((path for path in SEARCH_PATHS if path.is_file()), None)
if DATA_PATH is None:
    raise FileNotFoundError(f"Could not find {DATA_FILE} under input-data or init_data.")
OUT = HERE.parent / "data_viewer.html"

MEAS = ["vcd", "tcd", "substrate", "byproduct"]
FEEDS = ["feed_substrate", "feed_b", "feed_c", "feed_d"]

df = pd.read_csv(DATA_PATH, dtype={"batch_id": str})
df["dead"] = df["tcd"] - df["vcd"]
df["viability"] = 100 * df["vcd"] / df["tcd"]

test_ids = set(df.loc[df["to_predict"], "batch_id"])


def num(x):
    return None if pd.isna(x) else round(float(x), 6)


batches = {}
summary = []
for bid, g in sorted(df.groupby("batch_id"), key=lambda kv: int(kv[0].replace("Batch", ""))):
    g = g.sort_values("elapsed_time")
    split = "test" if bid in test_ids else "train"
    series = {}
    for col in MEAS + ["dead", "viability", "volume"] + FEEDS:
        s = g[["elapsed_time", col]].dropna()
        series[col] = [s["elapsed_time"].round(5).tolist(), s[col].round(6).tolist()]
    batches[bid] = {
        "split": split,
        "series": series,
        "raw": g.drop(columns=["dead", "viability"]).replace({np.nan: None}).to_dict(orient="records"),
    }

    m = g.dropna(subset=["vcd"])
    first = m.iloc[0] if len(m) else None
    last = m.iloc[-1] if len(m) else None
    row = {
        "batch": bid,
        "split": split,
        "n_meas": int(len(m)),
        "t_end": num(g["elapsed_time"].max()),
        "vcd0": num(first["vcd"]) if first is not None else None,
        "substrate0": num(first["substrate"]) if first is not None else None,
        "byproduct0": num(first["byproduct"]) if first is not None else None,
        "vcd_max": num(m["vcd"].max()) if split == "train" else None,
        "t_vcd_max": num(m.loc[m["vcd"].idxmax(), "elapsed_time"]) if split == "train" else None,
        "vcd_end": num(last["vcd"]) if split == "train" else None,
        "viab_end": num(last["viability"]) if split == "train" else None,
        "byproduct_max": num(m["byproduct"].max()) if split == "train" else None,
        "substrate_min": num(m["substrate"].min()) if split == "train" else None,
        "vol0": num(g["volume"].dropna().iloc[0]),
        "vol_end": num(g["volume"].dropna().iloc[-1]),
    }
    for f in FEEDS:
        row[f"{f}_total"] = num(g[f].sum())
        row[f"{f}_n"] = int(g[f].notna().sum())
    summary.append(row)

payload = {"batches": batches, "summary": summary}
plotly_js = (Path(plotly.__file__).parent / "package_data" / "plotly.min.js").read_text()

html = (HERE / "template.html").read_text()
html = html.replace("/*__PLOTLY__*/", plotly_js).replace("__DATA__", json.dumps(payload, separators=(",", ":")))
OUT.write_text(html)
print(f"Wrote {OUT} ({OUT.stat().st_size/1e6:.1f} MB) — {len(batches)} batches "
      f"({len(batches)-len(test_ids)} train, {len(test_ids)} test)")
