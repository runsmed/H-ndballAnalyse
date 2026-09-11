"""Last inn tidligere korrigerte kamprapporter som few-shot-eksempler i prompten.

Dette er IKKE trening/fine-tuning av modellen - Claude og Gemini er
ferdigtrente modeller som ikke lærer mellom API-kall eller husker noe fra
forrige kjøring. Det vi gjør her er å sende med 1 eller flere tidligere
VERIFISERTE (manuelt rettede) kamprapporter som eksempler i selve
forespørselen ("few-shot"-prompting), slik at modellen har noe konkret å
kalibrere format og nøyaktighetsnivå mot. Dette pleier å gi mer konsistent
output for domenespesifikke mønstre (kameravinkel, baneoppsett, hvordan mål
bekreftes i akkurat dine opptak), men er IKKE det samme som at modellen
faktisk blir smartere over tid.

Hvordan lage et referanseeksempel:
1. Kjør en vanlig analyse (analyze.py eller analyze_gemini.py) som normalt
2. Åpne den resulterende JSON-rapporten og rett opp feil for hånd i
   "event_log" (fjern hendelser som ikke skjedde, legg til de som mangler,
   korriger felter som team/jersey_color/player_number/shot_zone/goal_zone)
3. Legg gjerne til et "context"-felt øverst i JSON-filen med en kort
   beskrivelse, f.eks. "context": "Hvit drakt vs mørkeblå drakt, kamera fra
   sidelinjen, bredt utsnitt"
4. Bruk filen som referanse på neste kjøring: --reference rettet_kamp1.json
"""
from __future__ import annotations

import json
from typing import List, Optional

REFERENCE_EVENT_FIELDS = (
    "timecode", "event_type", "event_type_label", "description", "team",
    "jersey_color", "player_number", "shot_zone", "goal_zone", "confidence",
)


def load_reference_examples(paths: List[str]) -> List[dict]:
    """Last inn en eller flere rapport-JSON-filer (rettet for hånd) som referanseeksempler."""
    examples = []
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if "event_log" not in data:
            raise ValueError(
                f"'{path}' mangler 'event_log' - er dette en gyldig (rettet) kamprapport?"
            )
        examples.append(data)
    return examples


def build_reference_prompt_text(examples: List[dict]) -> Optional[str]:
    """Bygg en tekstblokk med few-shot-eksempler til bruk i systemprompten."""
    if not examples:
        return None

    blocks = []
    for i, example in enumerate(examples, start=1):
        context = (example.get("context") or "").strip()
        trimmed_events = [
            {k: v for k, v in event.items() if k in REFERENCE_EVENT_FIELDS}
            for event in example.get("event_log", [])
        ]
        context_line = f"Kontekst: {context}\n" if context else ""
        blocks.append(
            f"--- Referanseeksempel {i} (fra en tidligere, manuelt verifisert analyse) ---\n"
            f"{context_line}"
            f"Korrekt hendelseslogg:\n{json.dumps(trimmed_events, ensure_ascii=False, indent=2)}"
        )

    return (
        "\n\nDU FÅR OGSÅ REFERANSEEKSEMPLER FRA TIDLIGERE, MANUELT VERIFISERTE "
        "ANALYSER. Disse er fra ANDRE klipp/kamper - ikke anta at spillere, "
        "drakter eller konkrete hendelser er de samme i videoen du skal "
        "analysere nå. Bruk dem KUN som mal for format, presisjonsnivå og "
        "hvor grundig du bør fylle ut felter som goal_zone (også ved "
        "reddede skudd) og shot_zone.\n\n" + "\n\n".join(blocks)
    )
