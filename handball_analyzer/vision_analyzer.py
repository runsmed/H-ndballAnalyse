"""Send utpakkede video-frames til Claude for håndball-hendelsesanalyse."""
from __future__ import annotations

import base64
import time
from typing import List, Sequence, Tuple

import anthropic
import numpy as np

from .events import MatchEvent, format_timestamp, parse_json_event_list
from .frame_extractor import encode_jpeg

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

VIKTIG - bruk dommerens håndtegn som bevis når det er synlig i bildet
(disse bildene har INGEN lyd, så du kan ikke høre fløyten - se etter selve
håndtegnet, som ofte vises noen sekunder etter at spillet er stoppet).
Offisielle IHF-håndtegn:
- Arm strukket ut, peker i angrepsretning = frikast (free_throw)
- Arm/hånd peker nedover mot straffemerket (7-meter) = straffekast (penalty)
- Arm løftet med 2 fingre vist = utvisning 2 minutter (exclusion)
- Peker mot midten av banen = mål (goal)
Et tydelig håndtegn er et sterkere bevis enn å gjette ut fra ballbevegelse
alene - sett høyere confidence når du faktisk ser et slikt tegn.

For hendelser av type goal, shot_on_target, shot_wide, shot_blocked eller
penalty, angi i tillegg (hvis du kan vurdere det ut fra bildet - ellers
null, ikke gjett):

shot_zone (heltall) - hvor på banen skuddet ble avfyrt fra, sett fra
kameraets perspektiv (bildets venstre til høyre, ikke angripende lags
venstre/høyre):
  1-5 = mellomdistanse (ca. 6-9 meter fra mål), 5 soner fra venstre til
        høyre (1=lengst til venstre, 5=lengst til høyre)
  6-8 = langskudd (mer enn 9 meter fra mål), 3 soner fra venstre til høyre
  10  = straffekast/7-meter
  (sone 9 finnes ikke - ikke bruk den)

goal_zone (heltall 1-9) - hvor i målet ballen var på vei mot, sett fra
kameraets perspektiv:
  1=nede til venstre, 2=midt til venstre, 3=oppe til venstre,
  4=nede i midten, 5=midt i midten, 6=oppe i midten,
  7=nede til høyre, 8=midt til høyre, 9=oppe til høyre

VIKTIG: goal_zone skal ALLTID fylles ut for shot_on_target, INKLUDERT når
keeper redder skuddet - da er goal_zone stedet ballen var på vei mot FØR
keeper reddet den, altså nøyaktig der keeper dekket. Dette er spesielt
verdifull informasjon for keeper-analyse (hvilke soner keeperen dekker
godt/dårlig), så ikke hopp over dette feltet bare fordi skuddet ble reddet.
Fyll også ut for goal. IKKE fyll ut for shot_wide/shot_blocked (ballen gikk
da ikke mot en definerbar målsone) eller penalty med mindre skuddretningen
er tydelig synlig.

Svar KUN med gyldig JSON: en liste av objekter med feltene
timestamp (tall, sekunder), event_type (en av nøklene over), description
(kort norsk beskrivelse av hva som skjer), team ("angripende",
"forsvarende" eller null hvis ukjent), confidence (flyttall 0-1 for hvor
sikker du er), jersey_color (fargen på drakten til spilleren involvert i
hendelsen, f.eks. "rød", "blå", "hvit" - kun hvis du tydelig kan se fargen,
ellers null), player_number (draktnummeret til spilleren - for goal/
shot_on_target/shot_wide/shot_blocked/penalty er dette SKYTTERENS nummer,
altså spilleren som avfyrte skuddet, ikke andre spillere i bildet; for
andre hendelser som exclusion er det spilleren hendelsen gjelder. KUN hvis
tallet er tydelig lesbart i bildet - IKKE gjett, svar null hvis usikker),
shot_zone (se over, ellers null), goal_zone (se over, ellers null).

Hvis ingen hendelser er synlige i bildene, svar med en tom liste: []
Ikke inkluder forklarende tekst, kun JSON."""


class ClaudeVisionAnalyzer:
    """Wrapper rundt Anthropic-API-et for batchvis frame-analyse."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
        max_retries: int = 4,
        reference_text: str | None = None,
    ):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.max_retries = max_retries
        self.system_prompt = SYSTEM_PROMPT + (reference_text or "")

    def analyze_batch(self, frames: Sequence[Tuple[float, np.ndarray]]) -> List[MatchEvent]:
        """Send en batch (tidsstempel, frame) til Claude og parse hendelser."""
        if not frames:
            return []

        content = []
        for timestamp, frame in frames:
            jpeg_bytes = encode_jpeg(frame)
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
        return parse_json_event_list(response_text)

    def _call_with_retry(self, content: list) -> str:
        delay = 2.0
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=2048,
                    system=self.system_prompt,
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
