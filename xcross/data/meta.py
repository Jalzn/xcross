"""Load PFF match metadata and rosters, plus attack-direction helper."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class MatchMeta:
    match_id: str
    league: str
    season: str
    date: str
    home_team_id: int
    home_team_name: str
    away_team_id: int
    away_team_name: str
    home_team_start_left: bool
    fps: float
    pitch_length_m: float | None
    pitch_width_m: float | None


@dataclass(frozen=True, slots=True)
class PlayerEntry:
    player_id: int
    team_id: int
    shirt_num: str
    position_group: str
    nickname: str


def load_metadata(path: Path, league: str) -> MatchMeta:
    raw = json.loads(path.read_text())
    pitches = raw.get("stadium", {}).get("pitches") or [{}]
    pitch = pitches[0] if pitches else {}
    return MatchMeta(
        match_id=str(raw["gameId"]),
        league=league,
        season=str(raw["season"]),
        date=str(raw.get("date", "")),
        home_team_id=int(raw["homeTeam"]["id"]),
        home_team_name=str(raw["homeTeam"]["name"]),
        away_team_id=int(raw["awayTeam"]["id"]),
        away_team_name=str(raw["awayTeam"]["name"]),
        home_team_start_left=bool(raw["homeTeamStartLeft"]),
        fps=float(raw["videos"]["fps"]),
        pitch_length_m=float(pitch["length"]) if pitch.get("length") is not None else None,
        pitch_width_m=float(pitch["width"]) if pitch.get("width") is not None else None,
    )


def load_roster(path: Path) -> list[PlayerEntry]:
    raw = json.loads(path.read_text())
    return [
        PlayerEntry(
            player_id=int(e["player"]["id"]),
            team_id=int(e["team"]["id"]),
            shirt_num=str(e["shirtNumber"]),
            position_group=str(e["positionGroupType"]),
            nickname=str(e["player"]["nickname"]),
        )
        for e in raw
    ]


def attack_direction(meta: MatchMeta, period: int, team_id: int) -> int:
    """+1 if `team_id` attacks toward +x in `period`, -1 otherwise.

    Period 1 & 3: teams keep `homeTeamStartLeft` orientation (home attacks +x
    iff home started on the left). Period 2 & 4: teams switch sides.
    """
    is_home = team_id == meta.home_team_id
    home_attacks_positive_x = meta.home_team_start_left
    if period % 2 == 0:
        home_attacks_positive_x = not home_attacks_positive_x
    home_dir = 1 if home_attacks_positive_x else -1
    return home_dir if is_home else -home_dir
