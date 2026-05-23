"""CLI: iterate data/raw/ matches and write data/processed/<league>/<season>/<id>/."""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import traceback
from collections.abc import Iterable
from pathlib import Path

from joblib import Parallel, delayed
from loguru import logger
from tqdm.auto import tqdm

import xcross.config
import xcross.data.extract
import xcross.data.meta
from xcross.config import PROCESSED_ROOT, RAW_ROOT
from xcross.data.extract import extract_match

OUTPUT_FILES = ("crosses.parquet", "tracking.parquet", "ball.parquet", "meta.parquet", "roster.parquet")
VERSION_FILE = "_build_version"
_PIPELINE_MODULES = (xcross.config, xcross.data.meta, xcross.data.extract)


def _build_version() -> str:
    """Hash of the extraction code + config, so processed/ is invalidated and rebuilt
    whenever the pipeline changes (no silent drift between code and data)."""
    digest = hashlib.sha256()
    for module in _PIPELINE_MODULES:
        digest.update(Path(module.__file__).read_bytes())
    return digest.hexdigest()[:16]


def list_match_dirs(
    leagues: Iterable[str] | None = None,
    seasons: Iterable[str] | None = None,
    only: set[str] | None = None,
) -> list[Path]:
    league_filter = set(leagues) if leagues else None
    season_filter = set(seasons) if seasons else None
    out: list[Path] = []
    if not RAW_ROOT.is_dir():
        return out
    for league_dir in sorted(p for p in RAW_ROOT.iterdir() if p.is_dir()):
        if league_filter is not None and league_dir.name not in league_filter:
            continue
        for season_dir in sorted(p for p in league_dir.iterdir() if p.is_dir()):
            if season_filter is not None and season_dir.name not in season_filter:
                continue
            for match_dir in sorted(season_dir.iterdir()):
                if not match_dir.is_dir():
                    continue
                if only is not None and match_dir.name not in only:
                    continue
                if (match_dir / f"{match_dir.name}.jsonl.bz2").exists():
                    out.append(match_dir)
    return out


def _output_dir(match_dir: Path) -> Path:
    league = match_dir.parent.parent.name
    season = match_dir.parent.name
    return PROCESSED_ROOT / league / season / match_dir.name


def _is_processed(match_dir: Path) -> bool:
    out = _output_dir(match_dir)
    if not all((out / name).exists() for name in OUTPUT_FILES):
        return False
    version = out / VERSION_FILE
    return version.exists() and version.read_text() == _build_version()


def _process_one(match_dir_str: str) -> dict:
    match_dir = Path(match_dir_str)
    try:
        league = match_dir.parent.parent.name
        result = extract_match(match_dir, league)
        out = _output_dir(match_dir)
        out.mkdir(parents=True, exist_ok=True)
        result.crosses.write_parquet(out / "crosses.parquet", compression="zstd")
        result.tracking.write_parquet(out / "tracking.parquet", compression="zstd")
        result.ball.write_parquet(out / "ball.parquet", compression="zstd")
        result.meta.write_parquet(out / "meta.parquet", compression="zstd")
        result.roster.write_parquet(out / "roster.parquet", compression="zstd")
        (out / VERSION_FILE).write_text(_build_version())
        return {
            "ok": True,
            "match_id": match_dir.name,
            "n_crosses": len(result.crosses),
            "n_tracking": len(result.tracking),
            "stats": result.stats,
        }
    except Exception as exc:
        return {
            "ok": False,
            "match_dir": match_dir_str,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }


def build(
    leagues: list[str] | None,
    seasons: list[str] | None,
    only: set[str] | None,
    workers: int,
    rebuild: bool,
) -> int:
    match_dirs = list_match_dirs(leagues, seasons, only)
    logger.info(f"Found {len(match_dirs)} matches.")
    if not rebuild and only is None:
        before = len(match_dirs)
        match_dirs = [d for d in match_dirs if not _is_processed(d)]
        logger.info(f"Skipping {before - len(match_dirs)} already-processed; {len(match_dirs)} to do.")

    if not match_dirs:
        logger.info("Nothing to do.")
        return 0

    PROCESSED_ROOT.mkdir(parents=True, exist_ok=True)

    failures: list[dict] = []
    funnel = {"n_cr_detected": 0, "n_in_zone": 0, "n_kept": 0}
    no_yield: list[str] = []
    results = Parallel(n_jobs=max(1, workers), return_as="generator")(
        delayed(_process_one)(str(d)) for d in match_dirs
    )
    with tqdm(total=len(match_dirs), desc="extract") as pbar:
        for res in results:
            if res["ok"]:
                for key in funnel:
                    funnel[key] += res["stats"][key]
                if res["stats"]["n_cr_detected"] >= 10 and res["stats"]["n_in_zone"] == 0:
                    no_yield.append(res["match_id"])
            else:
                failures.append(res)
            pbar.update(1)

    if failures:
        logger.warning(f"{len(failures)} matches failed:")
        for f in failures[:5]:
            logger.warning(f"  {f['match_dir']}: {f['error']}")
            logger.debug(f["traceback"])

    if no_yield:
        logger.warning(
            f"{len(no_yield)} matches had crosses but 0 in-zone (likely poor/estimated "
            f"tracking, coordinates collapsed to centre): {no_yield[:10]}"
        )

    dropped_geometry = funnel["n_cr_detected"] - funnel["n_in_zone"]
    dropped_unresolved = funnel["n_in_zone"] - funnel["n_kept"]
    logger.info(
        f"Crosses: {funnel['n_cr_detected']} CR detected -> "
        f"{dropped_geometry} dropped (outside origin/target zone) -> "
        f"{dropped_unresolved} dropped (unresolved outcome) -> "
        f"{funnel['n_kept']} kept."
    )
    return 0 if not failures else 2


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--leagues", default=None, help="Comma-separated league names.")
    ap.add_argument("--seasons", default=None, help="Comma-separated season folders.")
    ap.add_argument("--matches", default=None, help="Comma-separated match ids.")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 1))
    ap.add_argument("--rebuild", action="store_true", default=False)
    args = ap.parse_args(argv)
    leagues = [s.strip() for s in args.leagues.split(",")] if args.leagues else None
    seasons = [s.strip() for s in args.seasons.split(",")] if args.seasons else None
    only = set(args.matches.split(",")) if args.matches else None
    return build(leagues, seasons, only, args.workers, args.rebuild)


if __name__ == "__main__":
    sys.exit(main())
