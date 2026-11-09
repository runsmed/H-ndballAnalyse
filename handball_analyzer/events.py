"""Datamodell for hendelser identifisert i en håndballkamp."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, asdict
from typing import List, Optional

EVENT_TYPES = {
    "goal": "Mål",
    "shot_on_target": "Skudd på mål",
    "shot_wide": "Skudd utenfor",
    "shot_post": "Stolpe",
    "shot_blocked": "Skudd blokkert",
    "free_throw": "Frikast",
    "exclusion": "Utvisning (2 min)",
    "warning": "Advarsel (gult kort)",
    "disqualification": "Diskvalifikasjon (rødt kort)",
    "penalty": "Straffekast",
    "turnover": "Ballmisting/pasningsfeil",
    "timeout": "Timeout",
    "empty_net": "Tomt mål (målvakt byttet ut for angriper)",
    "goalkeeper_back": "Målvakt tilbake",
    "numerical_advantage": "Overtall",
    "numerical_disadvantage": "Undertall",
    "tactic_press": "Press",
    "tactic_counter_attack": "Kontringsangrep",
    "tactic_set_offense": "Etablert angrep",
    "other": "Annet",
    # Heuristiske hendelser fra YOLO-basert lokal analyse (ikke semantisk
    # bekreftet - kun avledet fra ball-/spillerposisjoner og -bevegelse).
    "heuristic_possible_shot": "Mulig skudd (heuristikk)",
    "heuristic_fast_break": "Mulig kontring (heuristikk)",
    "heuristic_high_density": "Høy spillertetthet (heuristikk)",
}

# Skuddtype - HVORDAN skuddet oppsto taktisk (separat fra EVENT_TYPES, som
# er utfallet). Terminologi hentet fra offisiell håndball-kamprapportering.
SHOT_TYPES = {
    "breakthrough": "Gjennombrudd",
    "long_range": "Distanseskudd",
    "fast_break": "Kontring",
    "close_range": "Nærskudd",
}

TACTIC_EVENT_TYPES = {"tactic_press", "tactic_counter_attack", "tactic_set_offense"}


def format_timestamp(seconds: float) -> str:
    """Formater sekunder som mm:ss."""
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes:02d}:{secs:02d}"


@dataclass
class MatchEvent:
    timestamp: float
    event_type: str
    description: str
    team: Optional[str] = None
    confidence: Optional[float] = None
    jersey_color: Optional[str] = None
    player_number: Optional[str] = None
    shot_zone: Optional[int] = None
    goal_zone: Optional[int] = None
    shot_type: Optional[str] = None
    assist_player_number: Optional[str] = None

    @property
    def timecode(self) -> str:
        return format_timestamp(self.timestamp)

    def to_dict(self) -> dict:
        data = asdict(self)
        data["timecode"] = self.timecode
        data["event_type_label"] = EVENT_TYPES.get(self.event_type, self.event_type)
        if self.shot_type:
            data["shot_type_label"] = SHOT_TYPES.get(self.shot_type, self.shot_type)
        return data


def events_from_raw_list(raw_events: list) -> List["MatchEvent"]:
    """Bygg MatchEvent-objekter fra en rå liste med dicts (fra en LLM-respons)."""
    events: List[MatchEvent] = []
    for item in raw_events:
        if not isinstance(item, dict):
            continue
        try:
            events.append(MatchEvent(
                timestamp=float(item["timestamp"]),
                event_type=str(item.get("event_type", "other")),
                description=str(item.get("description", "")),
                team=item.get("team"),
                confidence=item.get("confidence"),
                jersey_color=item.get("jersey_color"),
                player_number=item.get("player_number"),
                shot_zone=_parse_optional_int(item.get("shot_zone")),
                goal_zone=_parse_optional_int(item.get("goal_zone")),
                shot_type=item.get("shot_type"),
                assist_player_number=item.get("assist_player_number"),
            ))
        except (KeyError, TypeError, ValueError):
            continue
    return events


def _parse_optional_int(value) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def parse_json_event_list(text: str) -> List["MatchEvent"]:
    """Parse en LLM-respons (evt. med omkringliggende tekst/kodeblokker) til hendelser."""
    match = re.search(r"\[.*\]", text, re.DOTALL)
    json_text = match.group(0) if match else text
    try:
        raw_events = json.loads(json_text)
    except json.JSONDecodeError:
        return []
    return events_from_raw_list(raw_events)
