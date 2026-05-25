"""Run the TabPFN out-of-fold benchmark on a Lightning AI GPU studio, then bring the
result back. TabPFN is O(n²) in the context size, so it is impractical on CPU but runs
in minutes on a GPU.

One-time auth (opens a browser, persists to ~/.lightning/credentials.json):

    uv run --with lightning-sdk lightning login

Then:

    uv run --with lightning-sdk python scripts/run_tabpfn_lightning.py

The studio gets the code by cloning the pushed branch; only the ~59 MB of parquets that
tabpfn_oof needs (gitignored) are uploaded, as one tarball. It uploads on a cheap CPU
machine, switches to an L4 to compute, and downloads comparison_tabpfn.csv into artifacts/.
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
STUDIO_NAME = "xcross-tabpfn"
REMOTE = "xcross"
BRANCH = "models/expand-registry-eval"
REPO_URL = "https://github.com/Jalzn/xcross.git"
RESULT = "artifacts/reports/metrics/comparison_tabpfn.csv"
DATA_GLOBS = ("data/features/*/*/*/features.parquet", "data/processed/*/*/*/meta.parquet")
SETUP_CMD = (
    f"rm -rf ~/{REMOTE} && git clone --depth 1 -b {BRANCH} {REPO_URL} ~/{REMOTE} && "
    f"tar xzf ~/data.tar.gz -C ~/{REMOTE} && cd ~/{REMOTE} && "
    "curl -LsSf https://astral.sh/uv/install.sh | sh && export PATH=$HOME/.local/bin:$PATH && "
    "uv sync && uv run python -c 'import torch; print(\"CUDA available:\", torch.cuda.is_available())'"
)
BENCHMARK_CMD = (
    f"cd ~/{REMOTE} && export PATH=$HOME/.local/bin:$PATH && "
    "XCROSS_TABPFN_DEVICE=cuda uv run python -m xcross.model.tabpfn_oof"
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
    """A GPU start can return before the machine is ready; uploads to it then 501."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if "running" in str(studio.status).lower():
            return
        time.sleep(5)
    raise TimeoutError(f"studio not running after {timeout}s (status={studio.status})")


def run() -> int:
    _build_data_bundle()
    studio = _studio()
    studio.start(Machine.CPU_SMALL)
    try:
        _wait_running(studio)
        studio.upload_file(str(BUNDLE), "data.tar.gz")
        print(studio.run(SETUP_CMD))

        studio.switch_machine(Machine.L4)
        _wait_running(studio)
        print(studio.run(BENCHMARK_CMD))

        (ROOT / RESULT).parent.mkdir(parents=True, exist_ok=True)
        studio.download_file(f"{REMOTE}/{RESULT}", str(ROOT / RESULT))
        print(f"Downloaded {RESULT}")
    finally:
        studio.stop()
        print("Studio stopped.")
    return 0


if __name__ == "__main__":
    sys.exit(run())
