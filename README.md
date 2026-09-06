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

### Eksempel: rask test på et kort klipp

```bash
python analyze.py --video kamp.mp4 --interval 2 --max-frames 20 --output test_rapport.json
```

## Om kostnad og ytelse

Hvert frame som sendes til Claude koster API-tokens. For en full kamp (60+
minutter) med 1 sekunds intervall blir det svært mange bilder og kan bli
kostbart og tregt. Anbefalinger:

- Bruk et større `--interval` (f.eks. 2–5 sekunder) for lange kamper eller
  førsteutkast av analysen.
- `--batch-size` grupperer flere frames i én API-forespørsel, som gir Claude
  bedre kontekst (bevegelse over tid) og reduserer antall kall.
- `--max-dimension` skalerer ned bilder før sending for å redusere
  tokenforbruk.
- Bruk `--max-frames` for å teste verktøyet på en liten del av videoen først.

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
