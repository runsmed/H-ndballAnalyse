"""Send hele videofiler direkte til Gemini for håndball-hendelsesanalyse.

I motsetning til Claude-integrasjonen (som får tilsendt enkeltbilder) laster
denne modulen opp HELE videofilen til Gemini via Files API. Gemini sampler
selv frames internt (standard 1 bilde/sekund) og resonnerer over bevegelse
og hendelser direkte i videoen - vi trenger ikke trekke ut frames selv.

Lange kamper deles opp i tidsbolker (chunk_seconds) og analyseres med flere
påfølgende kall (samme opplastede fil gjenbrukes), for å holde hvert svar
innenfor en håndterbar lengde og unngå å treffe modellens grense for
maks antall utdata-tokens per svar.

Merk: Gemini-modellnavn og gratis-kvoter endres ofte. DEFAULT_MODEL bruker
Googles "-latest"-alias, som automatisk peker på nyeste Flash-modell -
sjekk https://ai.google.dev/gemini-api/docs/pricing for gjeldende status
på hva som er gratis.
"""
from __future__ import annotations

import time
from typing import Callable, List, Optional

from google import genai
from google.genai import errors, types

from .events import MatchEvent, format_timestamp, parse_json_event_list
from .frame_extractor import get_video_duration

DEFAULT_MODEL = "gemini-flash-latest"

SYSTEM_PROMPT = """Du er en ekspert håndballanalytiker som ser på video fra en \
håndballkamp. Du får tilsendt et tidssegment av kampen (med starttidspunkt \
oppgitt i sekunder fra start av HELE kampen, ikke fra start av segmentet).

Se gjennom segmentet og identifiser alle forekomster av følgende \
hendelsestyper:

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

For hendelser av type goal, shot_on_target, shot_wide, shot_blocked eller
penalty, angi i tillegg (hvis du kan vurdere det ut fra videoen - ellers
null, ikke gjett):

shot_zone (heltall) - hvor på banen skuddet ble avfyrt fra, sett fra
kameraets perspektiv (bildets venstre til høyre, ikke angripende lags
venstre/høyre):
  1-5 = mellomdistanse (ca. 6-9 meter fra mål), 5 soner fra venstre til
        høyre (1=lengst til venstre, 5=lengst til høyre)
  6-8 = langskudd (mer enn 9 meter fra mål), 3 soner fra venstre til høyre
  10  = straffekast/7-meter
  (sone 9 finnes ikke - ikke bruk den)

goal_zone (heltall 1-9) - KUN for goal og shot_on_target (ikke for
shot_wide/shot_blocked, siden ballen da ikke går i mål) - hvor i målet
ballen traff/var på vei mot, sett fra kameraets perspektiv:
  1=nede til venstre, 2=midt til venstre, 3=oppe til venstre,
  4=nede i midten, 5=midt i midten, 6=oppe i midten,
  7=nede til høyre, 8=midt til høyre, 9=oppe til høyre

Svar KUN med gyldig JSON: en liste av objekter med feltene
timestamp (tall, ABSOLUTT sekund fra start av HELE kampen - ikke fra start
av segmentet du fikk), event_type (en av nøklene over), description (kort
norsk beskrivelse av hva som skjer), team ("angripende", "forsvarende"
eller null hvis ukjent), confidence (flyttall 0-1 for hvor sikker du er),
jersey_color (fargen på drakten til spilleren involvert i hendelsen, f.eks.
"rød", "blå", "hvit" - kun hvis du tydelig kan se fargen, ellers null),
player_number (draktnummeret til spilleren, KUN hvis tallet er tydelig
lesbart i videoen - IKKE gjett, svar null hvis usikker), shot_zone (se
over, ellers null), goal_zone (se over, ellers null).

Hvis ingen hendelser er synlige i segmentet, svar med en tom liste: []
Ikke inkluder forklarende tekst, kun JSON."""

ProgressCallback = Callable[[int, int, float, float], None]


class GeminiVideoAnalyzer:
    """Wrapper rundt Google Gen AI SDK-et for hel-video håndballanalyse."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = DEFAULT_MODEL,
        max_retries: int = 4,
        request_delay_seconds: float = 7.0,
    ):
        self.client = genai.Client(api_key=api_key) if api_key else genai.Client()
        self.model = model
        self.max_retries = max_retries
        self.request_delay_seconds = request_delay_seconds

    def upload_video(self, video_path: str) -> types.File:
        """Last opp videofilen og vent til Gemini har gjort den klar for analyse."""
        video_file = self.client.files.upload(file=video_path)
        while video_file.state == types.FileState.PROCESSING:
            time.sleep(5)
            video_file = self.client.files.get(name=video_file.name)

        if video_file.state == types.FileState.FAILED:
            raise RuntimeError(f"Gemini kunne ikke behandle videofilen: {video_file.error}")

        return video_file

    def analyze_segment(
        self,
        video_file: types.File,
        start_seconds: float,
        end_seconds: float,
        fps: float = 1.0,
    ) -> List[MatchEvent]:
        """Analyser ett tidssegment av en allerede opplastet video."""
        video_part = types.Part(
            file_data=types.FileData(file_uri=video_file.uri, mime_type=video_file.mime_type),
            video_metadata=types.VideoMetadata(
                start_offset=f"{int(start_seconds)}s",
                end_offset=f"{int(round(end_seconds))}s",
                fps=fps,
            ),
        )
        prompt_text = (
            f"Dette segmentet dekker kampens absolutte tidsrom "
            f"{format_timestamp(start_seconds)}-{format_timestamp(end_seconds)} "
            f"({start_seconds:.0f}-{end_seconds:.0f} sekunder fra kampstart). "
            f"Husk at timestamp i svaret ditt skal være absolutt sekund fra "
            f"kampstart (altså minst {start_seconds:.0f})."
        )

        response_text = self._call_with_retry(video_part, prompt_text)
        return parse_json_event_list(response_text)

    def _call_with_retry(self, video_part: types.Part, prompt_text: str) -> str:
        delay = self.request_delay_seconds
        last_error: Optional[Exception] = None

        for attempt in range(self.max_retries + 1):
            try:
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=[video_part, prompt_text],
                    config=types.GenerateContentConfig(
                        system_instruction=SYSTEM_PROMPT,
                        response_mime_type="application/json",
                    ),
                )
                # Enkel rate-limiting for å holde oss innenfor gratis-kvoten
                # (typisk 10-15 forespørsler/minutt på Flash-modeller).
                time.sleep(self.request_delay_seconds)
                return response.text or "[]"
            except errors.APIError as exc:
                last_error = exc
                if attempt == self.max_retries:
                    break
                # Lengre ventetid ved rate limit (429) enn ved andre feil.
                wait = delay * 3 if getattr(exc, "code", None) == 429 else delay
                time.sleep(wait)
                delay *= 2

        raise RuntimeError(f"Gemini API-kall feilet etter flere forsøk: {last_error}")

    def analyze_video(
        self,
        video_path: str,
        chunk_seconds: float = 600.0,
        fps: float = 1.0,
        on_progress: Optional[ProgressCallback] = None,
    ) -> List[MatchEvent]:
        """Last opp og analyser en hel videofil, delt opp i tidsbolker."""
        duration = get_video_duration(video_path)
        video_file = self.upload_video(video_path)

        total_chunks = max(1, int(-(-duration // chunk_seconds))) if duration > 0 else 1
        all_events: List[MatchEvent] = []
        start = 0.0
        chunk_index = 0

        while start < duration:
            end = min(start + chunk_seconds, duration)
            chunk_index += 1
            if on_progress:
                on_progress(chunk_index, total_chunks, start, end)
            all_events.extend(self.analyze_segment(video_file, start, end, fps=fps))
            start = end

        return all_events
