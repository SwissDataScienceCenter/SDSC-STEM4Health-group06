# Julio · Decode the Bioreactor

- [Interactive notebook](notebooks/bioreactor_viewer.ipynb) for exploring batch measurements, feed profiles, summaries, and raw rows.
- [Offline viewer source](viewer/) for building the same kind of dashboard as a standalone HTML file.
- [Hybrid model plan](references/plan_C_hybrid.md) and [challenge slide](references/challenge-brief.jpeg).

## Run the viewers

Use the team's Python environment with NumPy, pandas, Plotly, and ipywidgets. Both viewers look for `Project 1 - DTB/bioreaction_remaining.csv` in `input-data/` first and fall back to `init_data/`.

Open the notebook in JupyterLab and run all cells. To build the offline viewer, run this command from the repository root.

```sh
python Julio/viewer/build_viewer.py
```

Open `Julio/data_viewer.html` in a browser. The generated HTML embeds the challenge data and is ignored by Git.
