"""Send utpakkede video-frames til Claude for håndball-hendelsesanalyse."""
from __future__ import annotations

import base64
import json
import re
import time
from typing import List, Sequence, Tuple

import anthropic

from .events import MatchEvent, format_timestamp

DEFAULT_MODEL = "claude-3-5-sonnet-latest"

SYSTEM_PROMPT = """Du er en ekspert håndballanalytiker som ser på enkeltbilder \
(frames) fra opptak av en håndballkamp. Bildene vises i rekkefølge med \
tidsstempel (mm:ss) fra kampen. Bruk gjerne konteksten fra flere \
påfølgende bilder til å vurdere bevegelse og hendelser som strekker seg \
over tid, men rapporter hver hendelse med tidsstempelet (i sekunder) til \
bildet der den er tydeligst synlig.

For hvert bilde, vurder om det viser noen av følgende hendelsestyper:

- goal: Mål
- shot_on_target: Skudd på mål (ikke mål)
- shot_wide: Skudd utenfor/i stolpe
- shot_blocked: Skudd blokkert av forsvarsspiller/keeper
- free_throw: Frikast
- exclusion: Utvisning (2 minutter)
- penalty: Straffekast
- numerical_advantage: Tydelig overtallssituasjon for angripende lag
- numerical_disadvantage: Tydelig undertallssituasjon for angripende lag
- tactic_press: Presspill/høyt forsvar
- tactic_counter_attack: Kontringsangrep
- tactic_set_offense: Etablert/posisjonsangrep

Svar KUN med gyldig JSON: en liste av objekter med feltene
timestamp (tall, sekunder), event_type (en av nøklene over), description
(kort norsk beskrivelse av hva som skjer), team ("angripende",
"forsvarende" eller null hvis ukjent), confidence (flyttall 0-1 for hvor
sikker du er).

Hvis ingen hendelser er synlige i bildene, svar med en tom liste: []
Ikke inkluder forklarende tekst, kun JSON."""


class ClaudeVisionAnalyzer:
    """Wrapper rundt Anthropic-API-et for batchvis frame-analyse."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
        max_retries: int = 4,
    ):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.max_retries = max_retries

    def analyze_batch(self, frames: Sequence[Tuple[float, bytes]]) -> List[MatchEvent]:
        """Send en batch (tidsstempel, jpeg_bytes) til Claude og parse hendelser."""
        if not frames:
            return []

        content = []
        for timestamp, jpeg_bytes in frames:
            content.append({
                "type": "text",
                "text": f"Bilde ved tidspunkt {format_timestamp(timestamp)} "
                        f"({timestamp:.1f} sekunder):",
            })
            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": base64.standard_b64encode(jpeg_bytes).decode("utf-8"),
                },
            })
        content.append({
            "type": "text",
            "text": "Analyser bildene over og svar med JSON-listen som beskrevet "
                    "i systeminstruksjonen.",
        })

        response_text = self._call_with_retry(content)
        return self._parse_events(response_text)

    def _call_with_retry(self, content: list) -> str:
        delay = 2.0
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=2048,
                    system=SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": content}],
                )
                return "".join(
                    block.text for block in response.content if block.type == "text"
                )
            except (
                anthropic.RateLimitError,
                anthropic.APIStatusError,
                anthropic.APIConnectionError,
            ) as exc:
                last_error = exc
                if attempt == self.max_retries:
                    break
                time.sleep(delay)
                delay *= 2
        raise RuntimeError(f"Claude API-kall feilet etter flere forsøk: {last_error}")

    @staticmethod
    def _parse_events(text: str) -> List[MatchEvent]:
        match = re.search(r"\[.*\]", text, re.DOTALL)
        json_text = match.group(0) if match else text
        try:
            raw_events = json.loads(json_text)
        except json.JSONDecodeError:
            return []

        events = []
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
                ))
            except (KeyError, TypeError, ValueError):
                continue
        return events
