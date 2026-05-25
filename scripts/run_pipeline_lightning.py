"""Run the full xCross model-comparison pipeline on a Lightning AI GPU studio, detached, so
it survives the local machine going to sleep. Two modes:

    uv run --with lightning-sdk lightning login                                   # one-time
    uv run --with lightning-sdk python scripts/run_pipeline_lightning.py launch   # start it
    uv run --with lightning-sdk python scripts/run_pipeline_lightning.py collect  # when back

`launch` uploads the data, clones the pushed branch, runs the coexistence sanity check, then
fires the pipeline with nohup (it keeps running on the studio after we disconnect) and leaves
the studio up. `collect` checks the done marker, downloads the artifacts and stops the studio.
"""

from __future__ import annotations

import shutil
import sys
import time
from pathlib import Path

from lightning_sdk import Machine, Studio, Teamspace
from lightning_sdk.api.user_api import UserApi

ROOT = Path(__file__).resolve().parent.parent
STAGING = ROOT / ".lightning_upload"
BUNDLE = ROOT / ".lightning_bundle.tar.gz"
STUDIO_NAME = "xcross-pipeline"
REMOTE = "xcross"
BRANCH = "models/expand-registry-eval"
REPO_URL = "https://github.com/Jalzn/xcross.git"
MACHINE = Machine.L4
REPORTS = "artifacts/reports"
DONE, FAILED = "~/PIPELINE_DONE", "~/PIPELINE_FAILED"
DATA_GLOBS = ("data/features/*/*/*/features.parquet", "data/processed/*/*/*/meta.parquet")

_PATH = "export PATH=$HOME/.local/bin:$PATH"
_ENV = "XCROSS_TABPFN=1 XCROSS_TABPFN_DEVICE=cuda OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1"
SETUP_CMD = (
    f"if [ -d ~/{REMOTE}/.git ]; then cd ~/{REMOTE} && git fetch --depth 1 origin {BRANCH} && "
    f"git reset --hard FETCH_HEAD; else mv ~/{REMOTE} ~/{REMOTE}.bad.$(date +%s) 2>/dev/null; "
    f"git clone --depth 1 -b {BRANCH} {REPO_URL} ~/{REMOTE}; fi && "
    f"tar xzf ~/data.tar.gz -C ~/{REMOTE} && cd ~/{REMOTE} && "
    f"curl -LsSf https://astral.sh/uv/install.sh | sh && {_PATH} && uv sync"
)
SANITY_CMD = (
    f"cd ~/{REMOTE} && {_PATH} && XCROSS_TABPFN=1 XCROSS_TABPFN_DEVICE=cuda uv run python -c "
    "'import torch, numpy as np; from xcross.model.estimators import ESTIMATORS; "
    "X = np.random.rand(300, 8); y = (np.random.rand(300) < 0.4).astype(int); "
    "[ESTIMATORS[m]().fit(X, y) for m in (\"xgboost\", \"lightgbm\", \"tabpfn\")]; "
    "print(\"COEXIST_OK cuda=\", torch.cuda.is_available())'"
)
DETACH_CMD = (
    f"cd ~/{REMOTE} && rm -f {DONE} {FAILED} && nohup bash -lc '"
    f"{_PATH}; export {_ENV}; cd ~/{REMOTE} && "
    "uv run python -m xcross.model.compare && "
    "uv run python -m xcross.model.robustness && "
    "uv run python -m xcross.model.report && "
    "uv run python -m xcross.model.comparison_figures && "
    f"touch {DONE} || touch {FAILED}"
    f"' > ~/pipeline.log 2>&1 & echo LAUNCHED"
)


def _build_data_bundle() -> None:
    if STAGING.exists():
        shutil.rmtree(STAGING)
    for pattern in DATA_GLOBS:
        for src in ROOT.glob(pattern):
            dst = STAGING / src.relative_to(ROOT)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
    BUNDLE.unlink(missing_ok=True)
    shutil.make_archive(str(BUNDLE).removesuffix(".tar.gz"), "gztar", root_dir=STAGING)


def _studio() -> Studio:
    api = UserApi()
    username = api._get_authed_user_name()
    teamspace = api._get_all_teamspace_memberships("")[0].name
    return Studio(STUDIO_NAME, teamspace=Teamspace(name=teamspace, user=username))


def _wait_running(studio: Studio, timeout: int = 900) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if "running" in str(studio.status).lower():
            return
        time.sleep(5)
    raise TimeoutError(f"studio not running after {timeout}s (status={studio.status})")


def _upload_with_retry(studio: Studio, local: str, remote: str, tries: int = 4) -> None:
    for attempt in range(tries):
        try:
            studio.upload_file(local, remote)
            return
        except Exception as error:
            if attempt == tries - 1:
                raise
            print(f"upload attempt {attempt + 1} failed ({error}); retrying in 30s")
            time.sleep(30)


def launch() -> int:
    _build_data_bundle()
    studio = _studio()
    if "running" not in str(studio.status).lower():
        studio.start(Machine.CPU_SMALL)
        _wait_running(studio)
    try:
        _upload_with_retry(studio, str(BUNDLE), "data.tar.gz")
        print(studio.run(SETUP_CMD))
        try:
            studio.auto_sleep = False
        except Exception as error:
            print(f"warning: could not disable auto_sleep ({error})")
        if str(MACHINE).lower() not in str(studio.machine).lower():
            studio.switch_machine(MACHINE)
            _wait_running(studio)
        sanity_out, sanity_code = studio.run_with_exit_code(SANITY_CMD)
        print(sanity_out)
        if sanity_code != 0 or "COEXIST_OK" not in sanity_out:
            raise RuntimeError(f"coexistence sanity failed (exit {sanity_code})")
        print(studio.run(DETACH_CMD))
    except Exception:
        studio.stop()
        raise
    print("Pipeline launched detached; the studio stays up. Safe to disconnect.")
    print("When back:  uv run --with lightning-sdk python scripts/run_pipeline_lightning.py collect")
    return 0


def collect() -> int:
    studio = _studio()
    if "running" not in str(studio.status).lower():
        studio.start(MACHINE)
        _wait_running(studio)
    status, _ = studio.run_with_exit_code(
        f"test -f {DONE} && echo PIPELINE_DONE || (test -f {FAILED} && echo PIPELINE_FAILED || echo RUNNING); "
        "echo '--- tail ---'; tail -5 ~/pipeline.log"
    )
    print(status)
    if "PIPELINE_DONE" not in status:
        print("Not finished yet — leaving the studio up. Re-run collect later.")
        return 1
    studio.download_folder(f"{REMOTE}/{REPORTS}/metrics", str(ROOT / REPORTS / "metrics"))
    studio.download_folder(f"{REMOTE}/{REPORTS}/figures", str(ROOT / REPORTS / "figures"))
    studio.stop()
    print("Downloaded artifacts and stopped the studio.")
    return 0


def run() -> int:
    mode = sys.argv[1] if len(sys.argv) > 1 else "launch"
    return collect() if mode == "collect" else launch()


if __name__ == "__main__":
    sys.exit(run())
