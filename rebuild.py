"""Rebuild the submitted 6.995 file without the notebook: python rebuild.py
Writes src/research/final_6995.csv and prints the largest difference (%) against the submitted file."""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE / "src" / "research"), str(HERE / "src" / "research" / "byproduct_kit")]
import final_6995 as F  # noqa: E402

F.build()
print("max % difference vs submitted file:", F.compare().round(10).to_dict())
