# NOTE: You MUST read tests/e2e/ft/README.md as source-of-truth and documentations
# Thin per-mode CI entry: registers the test and runs ONE mode via bare `python3 <file>`
# (the CUDA CI runner's execution model). Scenario logic lives in
# tests/e2e/ft/conftest_ft/scenario_with_failure.py.
import sys
from pathlib import Path

_MILES_ROOT: Path = Path(__file__).resolve().parents[3]
_miles_root_str = str(_MILES_ROOT)
if _miles_root_str in sys.path:
    sys.path.remove(_miles_root_str)
sys.path.insert(0, _miles_root_str)

from tests.ci.ci_register import register_cuda_ci
from tests.e2e.ft.conftest_ft.scenario_with_failure import run_ci

register_cuda_ci(est_time=900, suite="stage-c-8-gpu-h200", labels=["ft"])

_MODE: str = "dp2_cp2_pp2"

if __name__ == "__main__":
    run_ci(_MODE)
