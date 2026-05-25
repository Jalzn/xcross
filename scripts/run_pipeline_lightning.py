"""Run the full xCross model-comparison pipeline on a Lightning AI GPU studio and download
every artifact — reproducible end to end. The studio clones the pushed branch (code) and
receives only the ~59 MB of parquets it needs; TabPFN runs as a first-class estimator on the
GPU (XCROSS_TABPFN=1). A coexistence sanity check runs first, so if torch and xgboost/lightgbm
clash on this host too, we abort before the heavy pipeline.

    uv run --with lightning-sdk lightning login        # one-time
    uv run --with lightning-sdk python scripts/run_pipeline_lightning.py
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
DATA_GLOBS = ("data/features/*/*/*/features.parquet", "data/processed/*/*/*/meta.parquet")

_PATH = "export PATH=$HOME/.local/bin:$PATH"
_ENV = "XCROSS_TABPFN=1 XCROSS_TABPFN_DEVICE=cuda OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1"
SETUP_CMD = (
    f"if [ -d ~/{REMOTE}/.git ]; then cd ~/{REMOTE} && git fetch --depth 1 origin {BRANCH} && "
    f"git reset --hard FETCH_HEAD; else git clone --depth 1 -b {BRANCH} {REPO_URL} ~/{REMOTE}; fi && "
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
PIPELINE_CMD = (
    f"cd ~/{REMOTE} && {_PATH} {_ENV} && "
    "uv run python -m xcross.model.compare && "
    "uv run python -m xcross.model.robustness && "
    "uv run python -m xcross.model.report && "
    "uv run python -m xcross.model.comparison_figures"
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
    """The upload endpoint returns transient 5xx now and then; retry the whole upload."""
    for attempt in range(tries):
        try:
            studio.upload_file(local, remote)
            return
        except Exception as error:
            if attempt == tries - 1:
                raise
            print(f"upload attempt {attempt + 1} failed ({error}); retrying in 30s")
            time.sleep(30)


def run() -> int:
    _build_data_bundle()
    studio = _studio()
    studio.start(Machine.CPU_SMALL)
    try:
        _wait_running(studio)
        _upload_with_retry(studio, str(BUNDLE), "data.tar.gz")
        print(studio.run(SETUP_CMD))

        studio.switch_machine(MACHINE)
        _wait_running(studio)
        sanity_out, sanity_code = studio.run_with_exit_code(SANITY_CMD)
        print(sanity_out)
        if sanity_code != 0 or "COEXIST_OK" not in sanity_out:
            raise RuntimeError(f"coexistence sanity failed (exit {sanity_code}); TabPFN can't share the process here")

        pipeline_out, pipeline_code = studio.run_with_exit_code(PIPELINE_CMD)
        print(pipeline_out)
        if pipeline_code != 0:
            raise RuntimeError(f"pipeline failed (exit {pipeline_code})")

        studio.download_folder(f"{REMOTE}/{REPORTS}/metrics", str(ROOT / REPORTS / "metrics"))
        studio.download_folder(f"{REMOTE}/{REPORTS}/figures", str(ROOT / REPORTS / "figures"))
        print("Downloaded artifacts.")
    finally:
        studio.stop()
        print("Studio stopped.")
    return 0


if __name__ == "__main__":
    sys.exit(run())
