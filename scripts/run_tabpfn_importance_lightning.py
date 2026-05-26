"""Run the TabPFN permutation importance on the cloud, detached so it survives the Mac
going to sleep. Same launch/collect contract as the other cloud orchestrators.

    uv run --with lightning-sdk python scripts/run_tabpfn_importance_lightning.py launch
    uv run --with lightning-sdk python scripts/run_tabpfn_importance_lightning.py collect
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
DONE, FAILED = "~/IMPORTANCE_DONE", "~/IMPORTANCE_FAILED"
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
DETACH_CMD = (
    f"cd ~/{REMOTE} && rm -f {DONE} {FAILED} && nohup bash -lc '"
    f"{_PATH}; export {_ENV}; cd ~/{REMOTE} && "
    "uv run python scripts/tabpfn_importance.py && "
    f"touch {DONE} || touch {FAILED}"
    f"' > ~/importance.log 2>&1 & echo LAUNCHED"
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


def launch() -> int:
    studio = _studio()
    if "running" not in str(studio.status).lower():
        studio.start(MACHINE)
        _wait_running(studio)
    try:
        try:
            studio.auto_sleep = False
        except Exception as error:
            print(f"warning: auto_sleep not set ({error})")
        if str(MACHINE).lower() not in str(studio.machine).lower():
            studio.switch_machine(MACHINE)
            _wait_running(studio)
        print(studio.run(SETUP_CMD))
        print(studio.run(DETACH_CMD))
    except Exception:
        studio.stop()
        raise
    print("TabPFN importance launched detached; studio stays up.")
    print("When done:  uv run --with lightning-sdk python scripts/run_tabpfn_importance_lightning.py collect")
    return 0


def collect() -> int:
    studio = _studio()
    if "running" not in str(studio.status).lower():
        studio.start(MACHINE)
        _wait_running(studio)
    status, _ = studio.run_with_exit_code(
        f"test -f {DONE} && echo IMPORTANCE_DONE || (test -f {FAILED} && echo IMPORTANCE_FAILED || echo RUNNING); "
        "echo '--- tail ---'; tail -15 ~/importance.log"
    )
    print(status)
    if "IMPORTANCE_DONE" not in status:
        return 1
    for name in RESULT_FILES:
        studio.download_file(f"{REMOTE}/artifacts/reports/metrics/{name}", str(ROOT / "artifacts/reports/metrics" / name))
        print(f"downloaded: {name}")
    studio.stop()
    print("Studio stopped.")
    return 0


def run() -> int:
    mode = sys.argv[1] if len(sys.argv) > 1 else "launch"
    return collect() if mode == "collect" else launch()


if __name__ == "__main__":
    sys.exit(run())
