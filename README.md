# Håndballanalyse

Python-verktøy som analyserer video av håndballkamper med AI. Tre verktøy,
velg det som passer:

1. **`analyze_yolo.py`** – lokal, **gratis** forhåndsanalyse med YOLO
   (objektgjenkjenning). Finner ball og spillere, og foreslår tidsseksjoner
   med sannsynlig aktivitet (skudd, kontring, tett spill) helt uten
   API-kostnad.
2. **`analyze.py`** – sender frames til **Claude** for faktisk klassifisering
   av hendelser (mål, skudd, frikast, utvisning, taktikk). Kan kjøres på hele
   kampen, eller kun på seksjonene YOLO foreslo – noe som kutter
   API-kostnaden kraftig. Krever betalt API-nøkkel (pay-as-you-go).
3. **`analyze_gemini.py`** – sender **hele videofilen** direkte til **Google
   Gemini**, som selv sampler frames internt. Enklere oppsett (ingen
   frame-ekstraksjon/YOLO nødvendig), og Gemini har et gratis API-nivå
   (Flash-modeller, med lave hastighetsgrenser) i tillegg til betalt bruk.

```
Alternativ A - Claude (mest presist, koster fra første token):
                 GRATIS, lokalt                 KOSTER API-tokens
video.mp4  ──▶  analyze_yolo.py  ──▶  yolo_rapport.json (foreslåtte seksjoner)
                                              │
                                              ▼
                                       analyze.py --sections-from-report
                                              │
                                              ▼
                                       rapport.json (mål, skudd, frikast, taktikk)

Alternativ B - Gemini (enklest, ofte gratis innenfor kvote):
video.mp4  ──▶  analyze_gemini.py  ──▶  rapport_gemini.json
```

## Funksjonalitet

### YOLO-analyse (analyze_yolo.py) – gratis, lokal

- Trekker ut frames og kjører YOLO-objektgjenkjenning/-sporing lokalt (ingen
  API-kall, ingen kostnad)
- Sporer ball- og spillerposisjoner over tid
- Avleder **heuristiske** forslag til interessante øyeblikk: mulig skudd
  (rask ballbevegelse mot målsone), mulig kontring (ball krysser banen
  raskt), høy spillertetthet nær mål
- Foreslår tidsseksjoner (`suggested_sections`) som kan sendes videre til
  Claude for faktisk klassifisering

### Claude-analyse (analyze.py) – semantisk klassifisering

1. Tar inn en videofil (MP4, MOV eller lignende) som argument
2. Trekker ut frames automatisk med jevne mellomrom, for hele videoen eller
   kun utvalgte tidsseksjoner (manuelt eller fra en YOLO-rapport)
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

**Viktig om YOLO-delen:** YOLO ser kun objekter (ball, spillere) og
posisjonene deres – den forstår ikke håndballregler. Hendelsene den
foreslår er derfor grove heuristikker ("ballen beveger seg raskt mot en
målsone"), ikke bekreftede hendelser som mål, frikast eller utvisning. Den
er ment som et gratis filter for å finne *hvor* i videoen noe skjer – Claude
avgjør fortsatt *hva* som faktisk skjer.

### Gemini-analyse (analyze_gemini.py) – hel video, ofte gratis

1. Tar inn en videofil og laster den opp direkte til Gemini (ingen egen
   frame-ekstraksjon)
2. Gemini sampler selv bilder internt fra videoen (standard 1/sekund) og
   identifiserer samme hendelsestyper som Claude-varianten over
3. Lange kamper deles automatisk opp i tidsbolker (`--chunk-seconds`, default
   10 min) og analyseres med flere påfølgende kall
4. Genererer samme type kamprapport (JSON) som de andre verktøyene

**Viktig om Gemini-kostnad:** Google sitt gratis API-nivå gjelder kun
*Flash*-modeller, har lave hastighetsgrenser (typisk 10-15 forespørsler i
minuttet), og skifter jevnlig hvilken modell som er "gratis-modellen".
Dette verktøyet bruker som standard modellalias-et `gemini-flash-latest`,
som Google selv holder oppdatert mot nyeste Flash-modell. **En betalt
Google AI Pro/Ultra-abonnement for Gemini-*appen* (chat) er en egen
tjeneste og gir ikke automatisk gratis/utvidet API-tilgang** – sjekk din
kvote på [aistudio.google.com](https://aistudio.google.com/apikey) med
samme konto for å se hva du faktisk har tilgjengelig.

## Oppsett

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Sett API-nøkkelen(e) du trenger som miljøvariabel:

```bash
export ANTHROPIC_API_KEY=sk-ant-...      # for analyze.py (Claude)
export GEMINI_API_KEY=...                # for analyze_gemini.py (Gemini) - gratis å opprette på aistudio.google.com/apikey
```

Du trenger kun nøkkelen for det verktøyet du faktisk skal bruke – `analyze_yolo.py`
krever ingen av dem.

(Se `.env.example` for referanse — verktøyet leser variablene direkte fra
miljøet, det brukes ikke noe `.env`-bibliotek.)

## Bruk

### Anbefalt arbeidsflyt: YOLO først (gratis), så Claude på utvalgte seksjoner

```bash
# 1. Gratis, lokal forhåndsanalyse - finner interessante tidsseksjoner
python analyze_yolo.py --video kamp.mp4 --output yolo_rapport.json

# 2. Send kun disse seksjonene til Claude for faktisk klassifisering
python analyze.py --video kamp.mp4 --sections-from-report yolo_rapport.json --output rapport.json
```

### Alternativ: kjør Claude direkte på hele kampen (eller egne seksjoner)

```bash
python analyze.py --video kamp.mp4 --interval 1 --output rapport.json

# eller kun en manuelt valgt del av kampen:
python analyze.py --video kamp.mp4 --sections "12:30-13:00,45:10-45:40" --output rapport.json
```

### Alternativ: Gemini på hele kampen i ett steg (enklest, ofte gratis)

```bash
python analyze_gemini.py --video kamp.mp4 --output rapport_gemini.json
```

### Argumenter: analyze_yolo.py (gratis, lokal)

| Flagg                  | Beskrivelse                                                          | Default            |
|------------------------|-----------------------------------------------------------------------|---------------------|
| `--video`              | Sti til videofil (påkrevd)                                            | –                   |
| `--interval`           | Sekunder mellom hvert frame som analyseres                            | `0.5`               |
| `--output`             | Filsti for JSON-rapport                                                | `yolo_rapport.json` |
| `--model`              | YOLO-modellvekter (lastes ned automatisk første gang)                  | `yolov8n.pt`        |
| `--confidence`         | Minimum deteksjonssikkerhet                                            | `0.3`               |
| `--max-dimension`      | Maks bredde/høyde på frames                                            | `768`               |
| `--max-frames`         | Maks antall frames (for testing)                                       | ingen grense        |
| `--goal-zone-fraction` | Andel av bildebredden som regnes som målsone i hver ende               | `0.15`              |
| `--pad-seconds`        | Sekunder lagt til før/etter hver foreslått seksjon                     | `5`                 |
| `--min-gap-seconds`    | Slår sammen foreslåtte seksjoner nærmere hverandre enn dette           | `10`                |
| `--quiet`              | Ikke skriv sammendrag til konsoll                                      | av                  |

### Argumenter: analyze.py (Claude)

| Flagg                    | Beskrivelse                                                        | Default              |
|--------------------------|---------------------------------------------------------------------|-----------------------|
| `--video`                | Sti til videofil (påkrevd)                                          | –                     |
| `--interval`             | Sekunder mellom hvert frame som analyseres                          | `1.0`                 |
| `--output`               | Filsti for JSON-rapport                                              | `rapport.json`        |
| `--batch-size`           | Antall frames sendt per API-kall til Claude                          | `5`                   |
| `--model`                | Claude-modell som brukes                                              | `claude-3-5-sonnet-latest` |
| `--max-frames`           | Maks antall frames å analysere (nyttig for testing/kostnadskontroll) | ingen grense          |
| `--max-dimension`        | Maks bredde/høyde på frames før sending (px)                         | `768`                 |
| `--sections`             | Kun analyser disse tidsintervallene, f.eks. `"12:30-13:00,45:10-45:40"` | hele videoen       |
| `--sections-from-report` | Bruk `suggested_sections` fra en YOLO-rapport i stedet for `--sections` | –                  |
| `--quiet`                | Ikke skriv sammendrag til konsoll                                    | av                    |
| `--dry-run`              | Vis kostnadsanslag og avslutt uten å kalle Claude API                | av                    |
| `-y`, `--yes`            | Ikke spør om bekreftelse før API-kall (for skript/CI)                | av                    |

### Argumenter: analyze_gemini.py (Gemini, hel video)

| Flagg              | Beskrivelse                                                          | Default               |
|--------------------|-------------------------------------------------------------------------|-------------------------|
| `--video`          | Sti til videofil (påkrevd)                                              | –                       |
| `--output`         | Filsti for JSON-rapport                                                  | `rapport_gemini.json`   |
| `--model`          | Gemini-modell som brukes                                                 | `gemini-flash-latest`   |
| `--chunk-seconds`  | Sekunder video analysert per API-kall (lengre kamper deles opp)          | `600` (10 min)          |
| `--fps`            | Bilder/sekund Gemini sampler internt fra videoen (maks 24.0)             | `1.0`                   |
| `--quiet`          | Ikke skriv sammendrag til konsoll                                        | av                      |
| `-y`, `--yes`      | Ikke spør om bekreftelse før API-kall (for skript/CI)                    | av                      |

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

### Gemini som gratis(ere) alternativ

`analyze_gemini.py` bruker Googles gratis API-nivå for Flash-modeller når du
er innenfor kvoten – ofte reelt gratis for en enkelt kamp, i motsetning til
Claude som alltid koster fra første kall. Ulempene: lav hastighetsgrense på
gratis-nivået (så en lang kamp kan ta noen minutter pga. venting mellom
kall), og modellnavnene/kvotene Google tilbyr gratis endres oftere enn hos
Anthropic – sjekk alltid [aistudio.google.com](https://aistudio.google.com/apikey)
for gjeldende status før du kjører en lang kamp. Nøyaktigheten kan også
avvike noe fra Claude – sammenlign gjerne begge på samme testklipp første
gang.

### Anbefalinger for å holde kostnaden nede

- **Kjør `analyze_yolo.py` først** – helt gratis, og gir deg konkrete
  tidsseksjoner å sende til Claude i stedet for hele kampen. Dette er den
  mest effektive kostnadsreduksjonen (ofte 50–90 % færre frames, siden
  dødtid/pauser filtreres bort automatisk).
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
analyze.py                        CLI: Claude-analyse (koster API-tokens)
analyze_yolo.py                   CLI: YOLO-forhåndsanalyse (gratis, lokal)
analyze_gemini.py                 CLI: Gemini-analyse (hel video, ofte gratis innenfor kvote)
handball_analyzer/
  frame_extractor.py              Frame-ekstraksjon med OpenCV (hele video eller seksjoner)
  yolo_analyzer.py                YOLO-deteksjon/-sporing + heuristikker (skudd, kontring, tetthet)
  vision_analyzer.py              Claude API-integrasjon (batching, retry, parsing)
  gemini_analyzer.py              Gemini API-integrasjon (hel-video-opplasting, chunking, retry)
  cost_estimator.py               Kostnadsanslag før API-kall (Claude)
  sections.py                     Parsing/sammenslåing av tidsseksjoner
  events.py                       Datamodell for hendelser + delt JSON-parsing
  report.py                       Rapportbygging (statistikk, taktiske observasjoner)
requirements.txt
.env.example
```

## Hendelsesfelter: draktfarge og draktnummer

Hver hendelse i `event_log` kan inneholde `jersey_color` (f.eks. "rød",
"blå") og `player_number` (draktnummer) når disse er tydelig synlige i
bildet/videoen. Modellene er instruert til å svare `null` fremfor å gjette
når de er usikre. **Draktfarge er vanligvis pålitelig** (store, tydelige
fargeflater), mens **draktnummer er langt mer usikkert** — på lavere
oppløsning (f.eks. 480p) eller fra avstand er tall ofte for små/uskarpe til
å leses korrekt. Ikke stol blindt på draktnummer i rapporten uten å
verifisere mot videoen.

## Hendelsesfelter: skuddsone og målsone (eksperimentelt)

For skudd-relaterte hendelser (`goal`, `shot_on_target`, `shot_wide`,
`shot_blocked`, `penalty`) forsøker modellen i tillegg å angi:

**`shot_zone`** – hvor på banen skuddet ble avfyrt fra (sett fra kameraets
perspektiv, altså bildets venstre/høyre – ikke angripende lags
venstre/høyre):

| Sone | Betydning |
|------|-----------|
| 1–5  | Mellomdistanse (ca. 6–9 m fra mål), 5 soner fra venstre til høyre |
| 6–8  | Langskudd (mer enn 9 m fra mål), 3 soner fra venstre til høyre |
| 10   | Straffekast/7-meter |

(Det finnes ingen sone 9 i denne skalaen – det er ikke en trykkfeil.)

**`goal_zone`** – hvor i målet ballen traff/var på vei mot (kun for `goal`
og `shot_on_target`), sett fra kameraets perspektiv, 3×3-rutenett:

| 3 (oppe venstre) | 6 (oppe midt) | 9 (oppe høyre) |
|---|---|---|
| **2** (midt venstre) | **5** (midt midt) | **8** (midt høyre) |
| **1** (nede venstre) | **4** (nede midt) | **7** (nede høyre) |

**Viktig – dette er eksperimentelt og ikke verifisert som pålitelig.** Da
dette ble testet, rapporterte Gemini 4 mål i en kamp som faktisk endte 0-0 –
altså en fundamental feil, ikke bare unøyaktighet i detaljer. Det er derfor
ingen grunn til å anta at presis skudd-/målsoneplassering er mer pålitelig
enn selve mål-tellingen. Bruk disse feltene som et utgangspunkt for manuell
gjennomgang, ikke som ferdig statistikk. Presis og pålitelig skuddkart-
generering (posisjon på banen + plassering i mål) krever i praksis
kalibrert datasyn (bane-homografi og ball-/spillersporing), ikke bare en
generell AI-modell som tolker bilder – noe som ligger utenfor dette
verktøyets nåværende omfang.

## Begrensninger

- **AI-analysen kan ta feil, også på grunnleggende ting som antall mål** —
  test alltid resultatet manuelt mot et kjent klipp før du stoler på en full
  kamprapport. Lavere oppløsning, rask kameraføring og lav samplingsrate
  (bilder/sekund) øker sjansen for feil.
- Analysen er basert på enkeltbilder/korte bildesekvenser og er ikke like
  presis som manuell videoanalyse eller sporingsdata — bruk rapporten som et
  hjelpemiddel/utgangspunkt, ikke en fasit.
- Taktiske mønstre (press, kontring, etablert angrep) og
  overtall/undertall-vurderinger krever at situasjonen er tydelig synlig i
  bildet Claude får – kameravinkel og bildekvalitet påvirker nøyaktigheten.
- Modellen kan gjøre feil eller overse hendelser som skjer mellom to
  analyserte frames; kortere `--interval` gir høyere gjenkjenningsrate, men
  koster mer.
- **YOLO-heuristikkene er grove og kan overse hendelser** som ikke gir seg
  utslag i rask ballbevegelse eller spillertetthet (f.eks. et rolig plassert
  skudd, eller en utvisning uten mye bevegelse). Bruk `--pad-seconds` og
  `--min-gap-seconds` generøst, eller kjør Claude på hele kampen i stedet,
  hvis fullstendighet er viktigere enn kostnad.
- YOLO bruker en generisk objektmodell (COCO) – den kjenner igjen "person"
  og "sports ball", ikke håndballspesifikke ting som mål, dommer eller
  draktnummer. Nøyaktigheten på ballgjenkjenning kan variere med
  kameravinkel, avstand og bildekvalitet.
- **Gemini-integrasjonen er nyere og mindre gjennomtestet i praksis** enn
  Claude-delen (verifisert mot SDK-et, men ikke kjørt mot ekte kampvideo med
  en reell API-nøkkel i utviklingen av dette verktøyet). Modellnavn, gratis
  kvoter og API-form hos Google endres også oftere enn hos Anthropic – test
  gjerne på et kort klipp først (`--chunk-seconds` satt lavt) før du kjører
  en hel kamp.
