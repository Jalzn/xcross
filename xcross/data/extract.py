"""Single-match extraction: stream PFF tracking, emit 5 parquet tables.

Per cross window we standardise coordinates twice: first so the crosser team
always attacks +x (direction), then so every cross originates from the +y flank
(flank). The result is a single canonical orientation -- left to right, top to
bottom -- regardless of pitch side. We also filter out crosses whose ball never
reached the opposing penalty box (intercepted at origin).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl

from xcross.config import (
    BOX_DEPTH_M,
    BOX_HALF_WIDTH_M,
    CORNER_X_MARGIN_M,
    CORNER_Y_MARGIN_M,
    CROSS_MAX_WINDOW_S,
    END_LINE_TOLERANCE_M,
    PITCH_HALF_LENGTH_M,
    PITCH_HALF_WIDTH_M,
    SHOT_WINDOW_S,
)
from xcross.data.io import iter_frames
from xcross.data.meta import (
    MatchMeta,
    PlayerEntry,
    attack_direction,
    load_metadata,
    load_roster,
)

CROSS_SCHEMA: dict[str, pl.DataType] = {
    "cross_id": pl.Utf8,
    "match_id": pl.Utf8,
    "period": pl.Int32,
    "start_frame": pl.Int64,
    "end_frame": pl.Int64,
    "start_time_s": pl.Float64,
    "start_x": pl.Float64,
    "start_y": pl.Float64,
    "end_x": pl.Float64,
    "end_y": pl.Float64,
    "crosser_player_id": pl.Int64,
    "crosser_team_id": pl.Int64,
    "end_event_type": pl.Utf8,
    "end_event_team_id": pl.Int64,
    "ball_out_in_window": pl.Boolean,
    "success": pl.Boolean,
    "shot_in_window": pl.Boolean,
    "setpiece_type": pl.Utf8,
    "prev_pe_type": pl.Utf8,
}

TRACKING_SCHEMA: dict[str, pl.DataType] = {
    "cross_id": pl.Utf8,
    "frame_num": pl.Int64,
    "player_id": pl.Int64,
    "team_id": pl.Int64,
    "x": pl.Float64,
    "y": pl.Float64,
}

BALL_SCHEMA: dict[str, pl.DataType] = {
    "cross_id": pl.Utf8,
    "frame_num": pl.Int64,
    "x": pl.Float64,
    "y": pl.Float64,
    "z": pl.Float64,
}

META_SCHEMA: dict[str, pl.DataType] = {
    "match_id": pl.Utf8,
    "league": pl.Utf8,
    "season": pl.Utf8,
    "date": pl.Utf8,
    "home_team_id": pl.Int64,
    "home_team_name": pl.Utf8,
    "away_team_id": pl.Int64,
    "away_team_name": pl.Utf8,
    "home_team_start_left": pl.Boolean,
    "fps": pl.Float64,
    "pitch_length_m": pl.Float64,
    "pitch_width_m": pl.Float64,
}

ROSTER_SCHEMA: dict[str, pl.DataType] = {
    "match_id": pl.Utf8,
    "player_id": pl.Int64,
    "team_id": pl.Int64,
    "shirt_num": pl.Utf8,
    "position_group": pl.Utf8,
    "nickname": pl.Utf8,
}


@dataclass(slots=True)
class _OpenCross:
    cross_id: str
    period: int
    start_frame: int
    start_time_s: float
    crosser_team_id: int | None
    crosser_player_id: int | None
    direction: int  # +1 or -1, applied to x,y when emitting rows
    setpiece_type: str | None
    prev_pe_type: str | None
    start_x: float | None = None
    start_y: float | None = None
    end_x: float | None = None
    end_y: float | None = None
    ball_out_flag: bool = False


@dataclass(slots=True)
class MatchExtraction:
    crosses: pl.DataFrame
    tracking: pl.DataFrame
    ball: pl.DataFrame
    meta: pl.DataFrame
    roster: pl.DataFrame
    stats: dict[str, int]  # cross funnel counts, see extract_match


def _in_target_zone(x: float | None, y: float | None, half_length: float) -> bool:
    if x is None or y is None:
        return False
    return (
        half_length - BOX_DEPTH_M <= x <= half_length + END_LINE_TOLERANCE_M
        and abs(y) <= BOX_HALF_WIDTH_M
    )


def _in_origin_zone(
    x: float | None, y: float | None, half_length: float, half_width: float
) -> bool:
    if x is None or y is None:
        return False
    final_third_line = half_length / 3
    if x < final_third_line or abs(y) < BOX_HALF_WIDTH_M:
        return False
    in_corner = (
        x >= half_length - CORNER_X_MARGIN_M and abs(y) >= half_width - CORNER_Y_MARGIN_M
    )
    return not in_corner


def _select_kept_ids(
    cross_rows: list[dict[str, Any]], meta: MatchMeta
) -> tuple[set[str], dict[str, int]]:
    """Keep crosses inside both zones with a resolved outcome; report the funnel."""
    half_length = (meta.pitch_length_m or 2 * PITCH_HALF_LENGTH_M) / 2
    half_width = (meta.pitch_width_m or 2 * PITCH_HALF_WIDTH_M) / 2
    in_zone_ids = {
        r["cross_id"] for r in cross_rows
        if r["crosser_team_id"] is not None
        and _in_origin_zone(r["start_x"], r["start_y"], half_length, half_width)
        and _in_target_zone(r["end_x"], r["end_y"], half_length)
    }
    # Drop crosses whose outcome could not be resolved (capped/period-end windows, or
    # an end event with no team): the success label would be null.
    kept_ids = {
        r["cross_id"] for r in cross_rows
        if r["cross_id"] in in_zone_ids and r["success"] is not None
    }
    stats = {
        "n_cr_detected": len(cross_rows),
        "n_in_zone": len(in_zone_ids),
        "n_kept": len(kept_ids),
    }
    return kept_ids, stats


def _collect_shot_events(
    pe_log: list[dict[str, Any]], ge_lookup: dict[int, dict[str, Any]]
) -> list[tuple[int, int, int | None]]:
    """(frame, period, team_id) for every shot, used to flag shots after a cross."""
    events: list[tuple[int, int, int | None]] = []
    for entry in pe_log:
        if entry["type"] != "SH":
            continue
        ctx = ge_lookup.get(entry["ge_id"]) if entry["ge_id"] is not None else None
        events.append((entry["frame"], entry["period"], ctx.get("team_id") if ctx else None))
    return events


def _emit_players(
    frame_num: int,
    cross_id: str,
    players: list[dict[str, Any]],
    team_id: int,
    direction: int,
    jersey_to_player: dict[tuple[int, str], int],
    sink: list[dict[str, Any]],
) -> None:
    for p in players:
        pid = jersey_to_player.get((team_id, str(p["jerseyNum"])))
        if pid is None:
            continue
        px, py = p.get("x"), p.get("y")
        if px is None or py is None:
            continue
        sink.append({
            "cross_id": cross_id,
            "frame_num": frame_num,
            "player_id": pid,
            "team_id": team_id,
            "x": float(px) * direction,
            "y": float(py) * direction,
        })


def _emit_window_frame(
    open_cr: _OpenCross,
    fnum: int,
    frame: dict[str, Any],
    meta: MatchMeta,
    jersey_to_player: dict[tuple[int, str], int],
    track_rows: list[dict[str, Any]],
    ball_rows: list[dict[str, Any]],
) -> None:
    """Emit one window frame's player and ball rows; track the cross start/end ball."""
    cid, direction = open_cr.cross_id, open_cr.direction
    _emit_players(fnum, cid, frame.get("homePlayersSmoothed") or [],
                  meta.home_team_id, direction, jersey_to_player, track_rows)
    _emit_players(fnum, cid, frame.get("awayPlayersSmoothed") or [],
                  meta.away_team_id, direction, jersey_to_player, track_rows)
    ball = frame.get("ballsSmoothed") or {}
    bx, by = ball.get("x"), ball.get("y")
    if bx is None or by is None:
        return
    bz = ball.get("z")
    mx, my = float(bx) * direction, float(by) * direction
    ball_rows.append({
        "cross_id": cid, "frame_num": fnum, "x": mx, "y": my,
        "z": float(bz) if bz is not None else 0.0,
    })
    if open_cr.start_x is None:
        open_cr.start_x = mx
        open_cr.start_y = my
    open_cr.end_x = mx
    open_cr.end_y = my


def _open_cross(
    pe: dict[str, Any],
    pe_id: int,
    fnum: int,
    period: int,
    match_id: str,
    ge_lookup: dict[int, dict[str, Any]],
    meta: MatchMeta,
    prev_pe_type: str | None,
) -> _OpenCross:
    ge_id = pe.get("game_event_id")
    ge_ctx = ge_lookup.get(int(ge_id)) if ge_id is not None else None
    team_id = ge_ctx.get("team_id") if ge_ctx else None
    # The CR's game event shares its frame and is recorded first, so team_id is known
    # here. If it ever is not, direction stays +1 but crosser_team_id is None, so the
    # cross is dropped by the zone filter rather than emitted with a wrong orientation.
    direction = attack_direction(meta, period, team_id) if team_id is not None else 1
    return _OpenCross(
        cross_id=f"{match_id}-{pe_id}",
        period=period,
        start_frame=fnum,
        start_time_s=float(pe["start_time"]),
        crosser_team_id=team_id,
        crosser_player_id=ge_ctx.get("player_id") if ge_ctx else None,
        direction=direction,
        setpiece_type=ge_ctx.get("setpiece_type") if ge_ctx else None,
        prev_pe_type=prev_pe_type,
    )


def _build_cross_row(
    oc: _OpenCross,
    match_id: str,
    end_frame: int,
    end_event_type: str | None,
) -> dict[str, Any]:
    return {
        "cross_id": oc.cross_id,
        "match_id": match_id,
        "period": oc.period,
        "start_frame": oc.start_frame,
        "end_frame": end_frame,
        "start_time_s": oc.start_time_s,
        "start_x": oc.start_x,
        "start_y": oc.start_y,
        "end_x": oc.end_x,
        "end_y": oc.end_y,
        "crosser_player_id": oc.crosser_player_id,
        "crosser_team_id": oc.crosser_team_id,
        "end_event_type": end_event_type,
        "end_event_team_id": None,
        "ball_out_in_window": oc.ball_out_flag,
        "success": None,
        "shot_in_window": None,
        "setpiece_type": oc.setpiece_type,
        "prev_pe_type": oc.prev_pe_type,
    }


def _resolve_outcomes(
    cross_rows: list[dict[str, Any]],
    end_ge_ids: list[int | None],
    ge_lookup: dict[int, dict[str, Any]],
    sh_events: list[tuple[int, int, int | None]],
    shot_window_frames: int,
) -> None:
    for row, end_ge_id in zip(cross_rows, end_ge_ids, strict=True):
        if end_ge_id is not None:
            ctx = ge_lookup.get(end_ge_id)
            if ctx:
                row["end_event_team_id"] = ctx.get("team_id")

        team = row["crosser_team_id"]
        end_team = row["end_event_team_id"]
        if team is None:
            row["success"] = None
        elif row["ball_out_in_window"]:
            row["success"] = False
        elif end_team is None:
            row["success"] = None
        else:
            row["success"] = end_team == team

        if team is None:
            row["shot_in_window"] = None
            continue
        shot = row["end_event_type"] == "SH" and end_team == team
        if not shot:
            deadline = row["end_frame"] + shot_window_frames
            for sh_frame, sh_period, sh_team in sh_events:
                if (
                    sh_period == row["period"]
                    and row["end_frame"] <= sh_frame <= deadline
                    and sh_team == team
                ):
                    shot = True
                    break
        row["shot_in_window"] = shot


def _normalize_flank(
    cross_rows: list[dict[str, Any]],
    track_rows: list[dict[str, Any]],
    ball_rows: list[dict[str, Any]],
) -> None:
    """Reflect each cross across the x-axis so its origin sits on the +y flank.

    Direction already standardises attack toward +x; this puts left- and
    right-wing crosses on one canonical side so they are spatially comparable
    (paper Sec. 2.1, Fig. 1: a single origin region).
    """
    flank_sign = {
        row["cross_id"]: -1.0 if row["start_y"] < 0 else 1.0
        for row in cross_rows
    }
    for row in cross_rows:
        sign = flank_sign[row["cross_id"]]
        row["start_y"] *= sign
        row["end_y"] *= sign
    for row in track_rows:
        row["y"] *= flank_sign[row["cross_id"]]
    for row in ball_rows:
        row["y"] *= flank_sign[row["cross_id"]]


def extract_match(match_dir: Path, league: str) -> MatchExtraction:
    """Extract one match. Crosses are CR PEs that originate in the flank zone and
    whose ball reached the opposing box (set pieces and central balls dropped)."""
    match_id = match_dir.name
    meta = load_metadata(match_dir / "metadata.json", league)
    roster = load_roster(match_dir / "rosters.json")
    jersey_to_player: dict[tuple[int, str], int] = {
        (p.team_id, p.shirt_num): p.player_id for p in roster
    }

    cross_rows: list[dict[str, Any]] = []
    end_ge_ids: list[int | None] = []  # aligned with cross_rows: end event's game_event_id
    track_rows: list[dict[str, Any]] = []
    ball_rows: list[dict[str, Any]] = []
    pe_log: list[dict[str, Any]] = []
    ge_lookup: dict[int, dict[str, Any]] = {}
    last_frame_per_period: dict[int, int] = {}

    cross_window_frames = int(CROSS_MAX_WINDOW_S * meta.fps)
    open_cr: _OpenCross | None = None
    last_pe_type: str | None = None

    def close_open(
        end_frame: int,
        end_event_type: str | None,
        end_event_ge_id: int | None,
    ) -> None:
        nonlocal open_cr
        assert open_cr is not None
        cross_rows.append(_build_cross_row(open_cr, match_id, end_frame, end_event_type))
        end_ge_ids.append(end_event_ge_id)
        open_cr = None

    for frame in iter_frames(match_dir / f"{match_id}.jsonl.bz2"):
        fnum = int(frame["frameNum"])
        period = int(frame["period"])
        last_frame_per_period[period] = max(last_frame_per_period.get(period, 0), fnum)

        ge = frame.get("game_event")
        pe = frame.get("possession_event")
        ge_id = frame.get("game_event_id")
        pe_id = frame.get("possession_event_id")

        if ge is not None and ge_id is not None:
            ge_lookup[int(ge_id)] = {
                "team_id": int(ge["team_id"]) if ge.get("team_id") is not None else None,
                "player_id": int(ge["player_id"]) if ge.get("player_id") is not None else None,
                "type": ge.get("game_event_type"),
                "setpiece_type": ge.get("setpiece_type"),
            }

        # PFF repeats the CR frame several times; ignore repeats of the cross already
        # open, otherwise it would close and reopen, emitting zero-length crosses.
        if (
            pe is not None
            and pe_id is not None
            and open_cr is not None
            and f"{match_id}-{int(pe_id)}" == open_cr.cross_id
        ):
            continue

        if open_cr is not None and period != open_cr.period:
            close_open(
                end_frame=last_frame_per_period.get(open_cr.period, fnum),
                end_event_type=None,
                end_event_ge_id=None,
            )

        if pe is not None:
            new_pe_type = pe["possession_event_type"]
            new_pe_ge_id = int(pe["game_event_id"]) if pe.get("game_event_id") is not None else None
            pe_log.append({
                "frame": fnum,
                "period": period,
                "type": new_pe_type,
                "ge_id": new_pe_ge_id,
            })

            if open_cr is not None and open_cr.period == period:
                close_open(
                    end_frame=fnum,
                    end_event_type=new_pe_type,
                    end_event_ge_id=new_pe_ge_id,
                )

            if new_pe_type == "CR":
                open_cr = _open_cross(
                    pe, int(pe_id), fnum, period, match_id, ge_lookup, meta, last_pe_type
                )

            last_pe_type = new_pe_type

        if open_cr is not None and fnum > open_cr.start_frame:
            if ge is not None and ge.get("game_event_type") == "OUT":
                open_cr.ball_out_flag = True
                close_open(
                    end_frame=fnum,
                    end_event_type="OUT",
                    end_event_ge_id=int(ge_id) if ge_id is not None else None,
                )
            elif fnum - open_cr.start_frame >= cross_window_frames:
                close_open(end_frame=fnum, end_event_type=None, end_event_ge_id=None)

        if open_cr is not None and fnum >= open_cr.start_frame:
            _emit_window_frame(
                open_cr, fnum, frame, meta, jersey_to_player, track_rows, ball_rows
            )

    if open_cr is not None:
        close_open(
            end_frame=last_frame_per_period.get(open_cr.period, open_cr.start_frame),
            end_event_type=None,
            end_event_ge_id=None,
        )

    sh_events = _collect_shot_events(pe_log, ge_lookup)
    _resolve_outcomes(cross_rows, end_ge_ids, ge_lookup, sh_events, int(SHOT_WINDOW_S * meta.fps))

    kept_ids, stats = _select_kept_ids(cross_rows, meta)
    cross_rows = [r for r in cross_rows if r["cross_id"] in kept_ids]
    track_rows = [r for r in track_rows if r["cross_id"] in kept_ids]
    ball_rows = [r for r in ball_rows if r["cross_id"] in kept_ids]

    _normalize_flank(cross_rows, track_rows, ball_rows)

    # PFF repeats event frames: the same (frameNum, period) appears several times with
    # the possession event duplicated, so the CR frame's tracking is emitted more than
    # once. Dedupe to one row per (cross, frame, player) and per (cross, frame).
    return MatchExtraction(
        crosses=pl.DataFrame(cross_rows, schema=CROSS_SCHEMA),
        tracking=pl.DataFrame(track_rows, schema=TRACKING_SCHEMA).unique(
            subset=["cross_id", "frame_num", "player_id"], keep="first", maintain_order=True
        ),
        ball=pl.DataFrame(ball_rows, schema=BALL_SCHEMA).unique(
            subset=["cross_id", "frame_num"], keep="first", maintain_order=True
        ),
        meta=pl.DataFrame([_meta_row(meta)], schema=META_SCHEMA),
        roster=pl.DataFrame(
            [_roster_row(p, match_id) for p in roster], schema=ROSTER_SCHEMA
        ),
        stats=stats,
    )


def _meta_row(meta: MatchMeta) -> dict[str, Any]:
    return {
        "match_id": meta.match_id,
        "league": meta.league,
        "season": meta.season,
        "date": meta.date,
        "home_team_id": meta.home_team_id,
        "home_team_name": meta.home_team_name,
        "away_team_id": meta.away_team_id,
        "away_team_name": meta.away_team_name,
        "home_team_start_left": meta.home_team_start_left,
        "fps": meta.fps,
        "pitch_length_m": meta.pitch_length_m,
        "pitch_width_m": meta.pitch_width_m,
    }


def _roster_row(p: PlayerEntry, match_id: str) -> dict[str, Any]:
    return {
        "match_id": match_id,
        "player_id": p.player_id,
        "team_id": p.team_id,
        "shirt_num": p.shirt_num,
        "position_group": p.position_group,
        "nickname": p.nickname,
    }
