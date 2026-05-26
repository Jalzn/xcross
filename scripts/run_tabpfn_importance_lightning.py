"""Run the TabPFN permutation importance on the cloud, synchronously: an active SDK
connection keeps the studio awake (no auto_sleep race), and the studio is explicitly
stopped at the end (no idle billing).

    uv run --with lightning-sdk python scripts/run_tabpfn_importance_lightning.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

from lightning_sdk import Machine, Studio, Teamspace
from lightning_sdk.api.user_api import UserApi

ROOT = Path(__file__).resolve().parent.parent
STUDIO_NAME = "xcross-pipeline"
REMOTE = "xcross"
BRANCH = "models/expand-registry-eval"
MACHINE = Machine.L4
RESULT_FILES = ("importance_xcrossot_success_tabpfn.csv", "importance_xcrossot_shot_tabpfn.csv")

_PATH = "export PATH=$HOME/.local/bin:$PATH"
_ENV = (
    "PYTHONUNBUFFERED=1 XCROSS_TABPFN=1 XCROSS_TABPFN_DEVICE=cuda "
    "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "
    "OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1"
)
SETUP_CMD = (
    f"cd ~/{REMOTE} && git fetch --depth 1 origin {BRANCH} && git reset --hard FETCH_HEAD && "
    f"{_PATH} && uv sync"
)
RUN_CMD = (
    f"cd ~/{REMOTE} && {_PATH} && export {_ENV} && uv run python scripts/tabpfn_importance.py"
)


def _studio() -> Studio:
    api = UserApi()
    return Studio(
        STUDIO_NAME,
        teamspace=Teamspace(name=api._get_all_teamspace_memberships("")[0].name, user=api._get_authed_user_name()),
    )


def _wait_running(studio: Studio, timeout: int = 900) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if "running" in str(studio.status).lower():
            return
        time.sleep(5)
    raise TimeoutError("studio not running")


def run() -> int:
    studio = _studio()
    try:
        if "running" not in str(studio.status).lower():
            studio.start(MACHINE)
            _wait_running(studio)
        if str(MACHINE).lower() not in str(studio.machine).lower():
            studio.switch_machine(MACHINE)
            _wait_running(studio)
        print(studio.run(SETUP_CMD))
        out, code = studio.run_with_exit_code(RUN_CMD)
        print(out)
        if code != 0:
            raise RuntimeError(f"permutation importance failed (exit {code})")
        for name in RESULT_FILES:
            studio.download_file(
                f"{REMOTE}/artifacts/reports/metrics/{name}",
                str(ROOT / "artifacts/reports/metrics" / name),
            )
            print(f"downloaded: {name}")
    finally:
        studio.stop()
        print("Studio stopped.")
    return 0


if __name__ == "__main__":
    sys.exit(run())
