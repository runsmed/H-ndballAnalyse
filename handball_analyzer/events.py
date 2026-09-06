"""Datamodell for hendelser identifisert i en håndballkamp."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional

EVENT_TYPES = {
    "goal": "Mål",
    "shot_on_target": "Skudd på mål",
    "shot_wide": "Skudd utenfor",
    "shot_blocked": "Skudd blokkert",
    "free_throw": "Frikast",
    "exclusion": "Utvisning (2 min)",
    "penalty": "Straffekast",
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

    @property
    def timecode(self) -> str:
        return format_timestamp(self.timestamp)

    def to_dict(self) -> dict:
        data = asdict(self)
        data["timecode"] = self.timecode
        data["event_type_label"] = EVENT_TYPES.get(self.event_type, self.event_type)
        return data
