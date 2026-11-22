# Decode the Bioreactor: Team 6 final model (leaderboard 6.995)

| VCD | TCD | Substrate | Byproduct | **Mean MAPE** |
|---:|---:|---:|---:|---:|
| 6.609 | 6.475 | 6.873 | 8.022 | **6.995** |

A self-contained folder: the teaching notebook, all the code it needs, the challenge data and the cached model inputs. Copy or zip the whole folder; nothing outside it is used.

## Quick start

```bash
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cpu   # imported by one module, not used here
jupyter notebook final_6995.ipynb     # run all cells, ~30 s
# or, without the notebook:
python rebuild.py                     # rebuilds the submitted file and checks it (differences < 1e-13 %)
```

## What the forecast is

| Target | Forecast |
|---|---|
| VCD, TCD | (C21² × v4)^(1/3): teammate ensemble C21 blended with the 10-rotation kinetic model v4 |
| Substrate | C21 unchanged |
| Byproduct | geometric mean of six members (C11's B1, B2, C; whole-curve members K4, K6; amount curve A6), then day 13 × exp(−0.058), then × exp(−0.5 · batch offset predicted from the pH-controller base volume) |

The notebook explains each step with the theory behind it, figures, the cross-validation evidence and exercises.

## Folder layout

```
DTB_Team6_6995/
├── final_6995.ipynb          teaching notebook (start here)
├── rebuild.py                rebuild the submission without the notebook
├── requirements.txt
├── README.md
├── docs/RESEARCH_LOG.md      full trial history, including what did not work
├── input-data/Project 1 - DTB/
│   ├── bioreaction_remaining.csv     challenge data
│   └── bioreaction_template.csv      submission template (used by the v4 refit)
└── src/
    ├── dtb/                  helper package for the notebook
    │   ├── data.py           loading, controller events, base-volume profile
    │   ├── metrics.py        MAPE, best constant forecasts, paired comparisons
    │   ├── cv.py             out-of-fold tables, nested ablation, leaderboard noise
    │   ├── pipeline.py       the forecast step by step, leaderboard history, submit
    │   └── plots.py          all figures
    ├── research/             research code the pipeline calls (copied from the analysis folder)
    │   ├── final_6995.py     the whole forecast in one script (used by rebuild.py)
    │   ├── final_7026.py     v4 loader and byproduct members B1, B2, C, K4, K6
    │   ├── curve_member.py, byproduct_*.py, kinetic_trials.py, ...   member code and shared modules
    │   ├── export_v4_rotated.py (+ export_v4_substrate, export_v3_mape, score_mape)   v4 refit
    │   ├── C21.csv                          INPUT: teammate C21 ensemble (not rebuilt here)
    │   ├── official_candidate_v4_rotated.csv  cached v4 (refit ~10 min, see below)
    │   ├── attempt_v4third_cells.csv        the submitted file (reference for the checks)
    │   ├── team_benchmark_split.csv
    │   └── byproduct_kit/    kit.py, batch inputs and cached out-of-fold predictions
    └── c11/src/              team C11 library (imported as package `src`)
```

**Why the helper package is called `dtb` and not `src`:** the C11 library is imported as a package named `src`, so a second `src` package would collide.

## Notes

- **Cached inputs.** The out-of-fold predictions (`byproduct_kit/oof_with_amount_curves.csv`) take hours to recompute and are loaded from file. v4 is loaded from `official_candidate_v4_rotated.csv`. To refit it (~10 min), run `F.F7026.v4_rotated(force=True)` with `src/research` on the path.
- **C21 is an input.** It comes from teammates and is not rebuilt here.
- **Changes against the original workspace:** in the copies, `byproduct_kit/kit.py` and `run_c11_local.py` locate the data and the C11 library inside this folder, and `dtb/__init__.py` points to `src/research`. No modelling code was changed.
- **Submitting.** The last notebook cell submits to the leaderboard only if `SUBMIT = True`. Every call is a real attempt.
