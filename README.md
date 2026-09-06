# Håndballanalyse

Python-verktøy som analyserer video av håndballkamper med Claude AI. Verktøyet
trekker ut frames fra videoen med jevne mellomrom, sender dem til Claude for
bildeanalyse, og genererer en kamprapport med hendelseslogg, statistikk og
taktiske observasjoner.

## Funksjonalitet

1. Tar inn en videofil (MP4, MOV eller lignende) som argument
2. Trekker ut frames automatisk med jevne mellomrom (f.eks. hvert sekund)
3. Sender frames (i batcher, for kontekst og kostnadseffektivitet) til Claude,
   som identifiserer:
   - Mål og målsituasjoner
   - Skudd (på mål, over, blokkert)
   - Frikast og utvisninger
   - Overtall/undertall-situasjoner
   - Taktiske mønstre (press, kontring, etablert angrep)
4. Lagrer alle hendelser med tidsstempel
5. Genererer en kamprapport (JSON) med:
   - Hendelseslogg med tidskoder
   - Statistikk (antall skudd, mål, frikast osv.)
   - Taktiske observasjoner

## Oppsett

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Sett API-nøkkelen din som miljøvariabel:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

(Se `.env.example` for referanse — verktøyet leser variabelen direkte fra
miljøet, det brukes ikke noe `.env`-bibliotek.)

## Bruk

```bash
python analyze.py --video kamp.mp4 --interval 1 --output rapport.json
```

### Argumenter

| Flagg             | Beskrivelse                                                        | Default              |
|-------------------|---------------------------------------------------------------------|-----------------------|
| `--video`         | Sti til videofil (påkrevd)                                          | –                     |
| `--interval`      | Sekunder mellom hvert frame som analyseres                          | `1.0`                 |
| `--output`        | Filsti for JSON-rapport                                              | `rapport.json`        |
| `--batch-size`    | Antall frames sendt per API-kall til Claude                          | `5`                   |
| `--model`         | Claude-modell som brukes                                              | `claude-3-5-sonnet-latest` |
| `--max-frames`    | Maks antall frames å analysere (nyttig for testing/kostnadskontroll) | ingen grense          |
| `--max-dimension` | Maks bredde/høyde på frames før sending (px)                         | `768`                 |
| `--quiet`         | Ikke skriv sammendrag til konsoll                                    | av                    |
| `--dry-run`       | Vis kostnadsanslag og avslutt uten å kalle Claude API                | av                    |
| `-y`, `--yes`     | Ikke spør om bekreftelse før API-kall (for skript/CI)                | av                    |

### Eksempel: rask test på et kort klipp

```bash
python analyze.py --video kamp.mp4 --interval 2 --max-frames 20 --output test_rapport.json
```

## Om kostnad og ytelse

**Ja, dette koster ekte penger** – hvert frame som sendes til Claude forbruker
API-tokens, og en full kamp (60+ minutter) med 1 sekunds intervall betyr
tusenvis av bilder og hundrevis av API-kall.

### Innebygd kostnadssperre

Verktøyet viser **alltid** et kostnadsanslag før noe sendes til API-et, og ber
om bekreftelse:

```bash
python analyze.py --video kamp.mp4 --interval 1
```

```
=== Kostnadsanslag (grovt, før analyse starter) ===
Modell: claude-3-5-sonnet-latest
Antall frames: 3 600  (API-kall/batcher: 720)
Anslått input-tokens: ~2 019 600
Anslått output-tokens: ~216 000
Anslått kostnad: ~$9.30 USD

Fortsette og sende disse frames til Claude API? [y/N]
```

Bruk `--dry-run` for kun å se anslaget uten å bli spurt (avslutter automatisk),
eller `-y`/`--yes` for å hoppe over bekreftelsen (f.eks. i skript). Estimatet
er grovt (basert på Anthropics tommelfingerregel for bildetokens) og er ikke
en garanti – sjekk faktisk forbruk på
[console.anthropic.com](https://console.anthropic.com).

### Grove kostnadseksempler (60 minutters kamp, claude-3-5-sonnet)

| Intervall | Antall frames | Anslått kostnad |
|-----------|---------------|------------------|
| 1 sekund  | ~3 600        | ~$9 USD          |
| 3 sekunder| ~1 200        | ~$3 USD          |
| 5 sekunder| ~720          | ~$2 USD          |

Bytt til en billigere modell (f.eks. `--model claude-3-5-haiku-latest`) for
et førsteutkast eller for testing – det kutter kostnaden med 70–80 %, på
bekostning av noe nøyaktighet.

### Anbefalinger for å holde kostnaden nede

- Bruk et større `--interval` (f.eks. 2–5 sekunder) for lange kamper eller
  førsteutkast av analysen.
- `--batch-size` grupperer flere frames i én API-forespørsel, som gir Claude
  bedre kontekst (bevegelse over tid) og reduserer antall kall.
- `--max-dimension` skalerer ned bilder før sending for å redusere
  tokenforbruk.
- Bruk `--max-frames` og/eller `--dry-run` for å teste verktøyet på en liten
  del av videoen først, før du kjører hele kampen.
- Vurder en billigere modell (`--model claude-3-5-haiku-latest`) for grovsortering,
  og kjør kun de mest interessante periodene på nytt med Sonnet.

## Prosjektstruktur

```
analyze.py                        CLI-inngangspunkt
handball_analyzer/
  frame_extractor.py              Frame-ekstraksjon med OpenCV
  vision_analyzer.py              Claude API-integrasjon (batching, retry, parsing)
  events.py                       Datamodell for hendelser
  report.py                       Rapportbygging (statistikk, taktiske observasjoner)
requirements.txt
.env.example
```

## Begrensninger

- Analysen er basert på enkeltbilder/korte bildesekvenser og er ikke like
  presis som manuell videoanalyse eller sporingsdata — bruk rapporten som et
  hjelpemiddel/utgangspunkt, ikke en fasit.
- Taktiske mønstre (press, kontring, etablert angrep) og
  overtall/undertall-vurderinger krever at situasjonen er tydelig synlig i
  bildet Claude får – kameravinkel og bildekvalitet påvirker nøyaktigheten.
- Modellen kan gjøre feil eller overse hendelser som skjer mellom to
  analyserte frames; kortere `--interval` gir høyere gjenkjenningsrate, men
  koster mer.
