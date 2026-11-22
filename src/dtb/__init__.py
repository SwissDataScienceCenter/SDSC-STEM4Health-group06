"""dtb ("Decode the Bioreactor"): helper code for the Team 6 teaching notebook final_6995.ipynb.

The notebook adds `src/` to the Python path and imports `dtb`. The package is deliberately not called
`src`, because the team's C11 library is itself imported as a package named `src`.

Modules
    data     : loading the challenge data, controller events, base-volume profile
    metrics  : MAPE and paired per-batch comparisons
    cv       : out-of-fold (team cross-validation) tables, nested ablation of the byproduct steps
    pipeline : the forecast itself, step by step (wraps the research code in src/research)
    plots    : all figures
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]              # the export folder (DTB_Team6_6995)
ANALYSIS = ROOT / "src" / "research"                     # research modules and their input files
DATA_DIR = ROOT / "input-data" / "Project 1 - DTB"
KIT = ANALYSIS / "byproduct_kit"

for p in (ANALYSIS, KIT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

TARGETS = ["vcd", "tcd", "substrate", "byproduct"]
TARGET_NAMES = {"vcd": "viable cell density (VCD)", "tcd": "total cell density (TCD)",
                "substrate": "substrate", "byproduct": "byproduct"}
