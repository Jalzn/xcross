"""CLI: processed/<league>/<season>/<id>/ -> data/features/<...>/features.parquet.

One row per kept cross with all features (xCross + xCrossOT). A provenance hash of
config + the feature code invalidates stale outputs, like the extraction stage.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import polars as pl
from joblib import Parallel, delayed
from loguru import logger
from tqdm.auto import tqdm

import xcross.config
import xcross.features.assemble
import xcross.features.clearance
import xcross.features.counts
import xcross.features.coverage
import xcross.features.entropy
import xcross.features.events
import xcross.features.flight
import xcross.features.frames
import xcross.features.geometry
import xcross.features.goalkeeper
import xcross.features.grid
import xcross.features.marking
import xcross.features.pitch_control
import xcross.features.pockets
import xcross.features.pressure
import xcross.features.shape
import xcross.features.spatial
import xcross.features.support
import xcross.features.swing
import xcross.features.temporal
from xcross.config import FEATURES_ROOT, PITCH_HALF_LENGTH_M, PITCH_HALF_WIDTH_M, PROCESSED_ROOT
from xcross.features.assemble import cross_features

VERSION_FILE = "_features_version"
_FEATURE_MODULES = (
    xcross.config,
    xcross.features.grid,
    xcross.features.geometry,
    xcross.features.entropy,
    xcross.features.pitch_control,
    xcross.features.frames,
    xcross.features.spatial,
    xcross.features.events,
    xcross.features.counts,
    xcross.features.pressure,
    xcross.features.marking,
    xcross.features.pockets,
    xcross.features.coverage,
    xcross.features.goalkeeper,
    xcross.features.shape,
    xcross.features.flight,
    xcross.features.clearance,
    xcross.features.swing,
    xcross.features.support,
    xcross.features.temporal,
    xcross.features.assemble,
)


def _features_version() -> str:
    digest = hashlib.sha256()
    for module in _FEATURE_MODULES:
        digest.update(Path(module.__file__).read_bytes())
    return digest.hexdigest()[:16]


def _features_dir(match_dir: Path) -> Path:
    league, season = match_dir.parent.parent.name, match_dir.parent.name
    return FEATURES_ROOT / league / season / match_dir.name


def _is_built(match_dir: Path) -> bool:
    out = _features_dir(match_dir)
    version = out / VERSION_FILE
    return (out / "features.parquet").exists() and version.exists() and version.read_text() == _features_version()


def list_processed(only: set[str] | None) -> list[Path]:
    out: list[Path] = []
    if not PROCESSED_ROOT.is_dir():
        return out
    for crosses in sorted(PROCESSED_ROOT.glob("*/*/*/crosses.parquet")):
        match_dir = crosses.parent
        if only is None or match_dir.name in only:
            out.append(match_dir)
    return out


def _goalkeeper_ids(match_dir: Path) -> dict[int, set[int]]:
    """team_id -> {goalkeeper player_id(s)} from the roster (a team may have subbed keepers)."""
    roster_path = match_dir / "roster.parquet"
    if not roster_path.exists():
        return {}
    keepers = pl.read_parquet(roster_path).filter(pl.col("position_group") == "GK")
    grouped = keepers.group_by("team_id").agg(pl.col("player_id"))
    return {tid: set(pids) for tid, pids in zip(grouped["team_id"], grouped["player_id"], strict=True)}


def _process_one(match_dir_str: str) -> dict:
    match_dir = Path(match_dir_str)
    try:
        crosses = pl.read_parquet(match_dir / "crosses.parquet")
        meta = pl.read_parquet(match_dir / "meta.parquet").row(0, named=True)
        out = _features_dir(match_dir)
        out.mkdir(parents=True, exist_ok=True)

        if crosses.height == 0:
            pl.DataFrame().write_parquet(out / "features.parquet", compression="zstd")
            (out / VERSION_FILE).write_text(_features_version())
            return {"ok": True, "match_id": match_dir.name, "n": 0}

        half_length = (meta["pitch_length_m"] or 2 * PITCH_HALF_LENGTH_M) / 2
        half_width = (meta["pitch_width_m"] or 2 * PITCH_HALF_WIDTH_M) / 2
        fps = meta["fps"]
        tracking = pl.read_parquet(match_dir / "tracking.parquet").partition_by("cross_id", as_dict=True)
        ball = pl.read_parquet(match_dir / "ball.parquet").partition_by("cross_id", as_dict=True)
        gk_ids = _goalkeeper_ids(match_dir)

        rows = [
            cross_features(
                row, tracking[(row["cross_id"],)], ball[(row["cross_id"],)],
                meta["league"], meta["season"], half_length, half_width, fps, gk_ids,
            )
            for row in crosses.iter_rows(named=True)
        ]
        pl.DataFrame(rows).write_parquet(out / "features.parquet", compression="zstd")
        (out / VERSION_FILE).write_text(_features_version())
        return {"ok": True, "match_id": match_dir.name, "n": len(rows)}
    except Exception as exc:
        return {"ok": False, "match_dir": match_dir_str, "error": f"{type(exc).__name__}: {exc}"}


def build(only: set[str] | None, workers: int, rebuild: bool) -> int:
    match_dirs = list_processed(only)
    logger.info(f"Found {len(match_dirs)} processed matches.")
    if not rebuild and only is None:
        before = len(match_dirs)
        match_dirs = [d for d in match_dirs if not _is_built(d)]
        logger.info(f"Skipping {before - len(match_dirs)} already-built; {len(match_dirs)} to do.")
    if not match_dirs:
        logger.info("Nothing to do.")
        return 0

    FEATURES_ROOT.mkdir(parents=True, exist_ok=True)
    failures: list[dict] = []
    total = 0
    results = Parallel(n_jobs=max(1, workers), return_as="generator")(
        delayed(_process_one)(str(d)) for d in match_dirs
    )
    with tqdm(total=len(match_dirs), desc="features") as pbar:
        for res in results:
            if res["ok"]:
                total += res["n"]
            else:
                failures.append(res)
            pbar.update(1)

    if failures:
        logger.warning(f"{len(failures)} matches failed:")
        for f in failures[:5]:
            logger.warning(f"  {f['match_dir']}: {f['error']}")
    logger.info(f"Feature rows written: {total}")
    return 0 if not failures else 2


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--matches", default=None, help="Comma-separated match ids.")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--rebuild", action="store_true", default=False)
    args = ap.parse_args(argv)
    only = set(args.matches.split(",")) if args.matches else None
    return build(only, args.workers, args.rebuild)


if __name__ == "__main__":
    sys.exit(main())
