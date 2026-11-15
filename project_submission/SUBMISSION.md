# PlanSight: final submission text

Copy each section into the matching field of the Junction submission form.

---

## Project name

**PlanSight**

## Punchline

Paste this (200 characters max):

```text
Ask any blueprint a question, in any language. Every answer in your drawings, in plain sight.
```

## Short description (for cards and listings, ~300 characters)

PlanSight answers questions about construction drawings, in any language, and draws the evidence on the sheet: every door, room or WC it counted, numbered and clickable. It reads the drawing's own CAD/BIM source when it exists, uses NVIDIA Nemotron Nano Omni when it doesn't, and says "needs review" instead of guessing.

## Description

Paste the text between the `description:start` and `description:end` markers (5,000 characters max, Markdown supported).

<!-- description:start -->
Site engineers lose hours hunting through drawings: *how many doors on level 3? what is the net area of bedroom 003?* The answers hide in symbols, not text, and a vision model's bare number can't be checked. Stoa already answers with **the page behind every sentence**. PlanSight adds **the symbol behind every number**: boxed, numbered and clickable on the sheet, so a human can verify it in seconds.

### Five answers, five proofs

- **Swiss plan**, *"How many doors?"* → **36**, all 36 numbered on the sheet. Read from the DXF: 0 tokens, 0.3 s.
- **IFC model**, *"Net floor area of Bedroom 414B-5-003-001?"* → **10.71 m²**, citing `IfcSpace` GlobalId `2J_oFeqlHDNPq7R3anXaug`.
- **Finnish apartment scan**, *"Kuinka monta pesu- ja tiskiallasta…?"* → **2**, washbasin and kitchen sink boxed. ✓ matches the dataset.
- **A century-old archive scan**, *Detect everything* → 29 windows, rooms, stairs and fixtures, colour-coded in one click.
- **The one we refuse**: Nemotron counts **6 doors** on a CubiCasa plan, and 6 is even right, but its boxes form a straight column through a hallway. A right number with wrong evidence is still unverifiable, so PlanSight holds it for review.

### How it works: the most reliable source wins

1. **CAD.** If a PDF ships with its DXF, PlanSight counts entities per layer and maps each one back onto the PDF as a numbered box. **0 tokens, deterministic.**
2. **BIM.** Plans rendered from an IFC model are answered from the model: counts per storey, unit-scaled lengths, volumes and room areas, with IFC GlobalIds as evidence. **0 tokens.**
3. **NVIDIA Nemotron 3 Nano Omni** for scans and raster plans. It never returns a bare number: it returns a strict-JSON inventory of boxes, and **Python does the counting**. Boxes are validated and de-duplicated, and a guard catches invented evidence (six "doors" stamped in a straight line). Then PlanSight answers *needs review* with its proposal, instead of a confident guess.

### Visualized reasoning: a live app, styled as a Stoa companion

- **Live blueprint.** Drop in any PNG, JPG or PDF, even a century-old archive scan. *Detect everything* boxes every door, window, fixture, room and staircase. Or ask in English, Finnish, German or French: the answer arrives with numbered boxes, a one-line reason, the route that answered (DXF, IFC or Nemotron), its token cost, and a ✓ when it matches the dataset.
- **Benchmark.** Accuracy per dataset with the dataset's own scoring rules. Open any drawing, compare expected and saved answers, and re-run any question live.
- A static review report with filters and a reviewer export is generated for every run.

### Accuracy

| Dataset | Accuracy | Route |
|---|---|---|
| IFC models | **100%** (3,062 / 3,062, 9 languages) | IFC take-off · 0 tokens |
| Swiss Dwellings | **99.0%** on 18,528 held-out questions | DXF · 0 tokens |
| CubiCasa5K scans | 43% final, **55% when it answers** | Nemotron |
| Opening Design | 12.5% | Nemotron |

Scored with the dataset's rules (exact counts, ±2% on areas); scan figures from an 805-question sample. PlanSight would rather abstain than guess: across that sample, **85% of the answers it gives are correct**, and the rest are flagged for review.

### Low inference cost

- **0 tokens** for about a third of all 127k questions (CAD and BIM routes), ~30 ms each on a laptop CPU.
- **3B active parameters**: Nemotron 3 Nano Omni is a 30B mixture-of-experts.
- **~1,000 tokens per question** on average. One cached inventory answers both the English and the Finnish version of a question, and a per-drawing mode **halves the requests on CubiCasa** (51 → 24).
- **Open weights.** One command serves the FP8 checkpoint on a single rented H100 (~$2.5/h) in the EU, matching Stoa's promise: data held in the EU, never trained on.

### Creativity: what next year's models won't make obsolete

CAD files are free training labels. From 526 Swiss DXFs we generated 61k labelled boxes with zero hand labelling, and trained a 2.6M-parameter YOLO11n symbol detector: **mAP@50 0.89** on held-out sites. It already finds the real doors on Finnish scans it has never seen. It targets exactly the symbols vision models miss (doors, windows, fixtures), and every CAD file Stoa connects makes it better.

### Why Stoa

Stoa syncs M-Files, SharePoint and Dalux, where CAD and BIM files sit next to their PDFs. PlanSight turns those pairs into exact answers, and into training data for the scans that have none. It can ship as a Stoa skill: **count and prove.**
<!-- description:end -->

## Challenge

**Stoa: Make Blueprints Talk** (Junction 2026, Finland).

> Build a program that answers natural language questions about construction drawings and building information models accurately and verifiably. A great answer is correct and shows where in the drawing it came from so a human can verify it in seconds.

## How PlanSight scores on each judging criterion

| Criterion (25% each) | What PlanSight does |
|---|---|
| Accuracy | 99–100% where a CAD/BIM source exists; the vision path abstains (*needs review*) instead of guessing; scored with the dataset's own rules |
| Visualized reasoning | Live app: drop any drawing, detect everything or ask a question, and see numbered boxes on the actual page (from DXF entities or the model's inventory), IFC GlobalIds, route, cost and justification per answer; a benchmark tab replays every dataset question; review export |
| Creativity | Uses the blueprint's own source as ground truth, and turns CAD files into free detector labels; a guard against invented boxes |
| Low inference cost | 0 tokens for ~1/3 of questions; 3B-active Nemotron for the rest; cache shared across languages; reduced mode with −53% requests on CubiCasa; self-hostable open weights with a one-command GPU setup |

## Technologies used

- **NVIDIA Nemotron 3 Nano Omni** (`nemotron-3-nano-omni-30b-a3b-reasoning`), via NVIDIA's hosted API (build.nvidia.com, `integrate.api.nvidia.com`), with a vLLM preset for self-hosting the open BF16 weights.
- Python 3.13, pydantic (strict evidence schemas), httpx (OpenAI-compatible client, retries and 429 handling).
- PyMuPDF (PDF rendering, page geometry), Pillow.
- ezdxf (DXF layers to evidence boxes), IfcOpenShell 0.8 (IFC take-off, unit scaling, geometry for room areas).
- Ultralytics YOLO11n + PyTorch (symbol detector trained on DXF-generated labels).
- FastAPI + uvicorn web app (`drawingqa serve`) with a dependency-free HTML/CSS/JavaScript front end: Live blueprint and Benchmark tabs.
- A self-contained HTML/JavaScript review report (no server needed).
- Playwright (Chrome DevTools screencast) and ffmpeg to record the app for the video.
- Qwen3-VL-8B preset for model comparison.
- Data: the Junction construction-drawing dataset (Swiss Dwellings, CubiCasa5K, IFC models, Opening Design, IFC-Bench).

## Links

| What | Link |
|---|---|
| Source code | <https://github.com/thierry-silbermann/Junction_2026_Finland_Make_Blueprints_Talk> |
| Live demo (review report) | <https://raw.githack.com/thierry-silbermann/Junction_2026_Finland_Make_Blueprints_Talk/main/project_submission/demo/index.html> (works once the repo is pushed and public; GitHub Pages is an alternative, see README) |
| Live app (local) | `pip install -e '.[app]' && drawingqa serve`, then <http://127.0.0.1:8000/>; it is shown in the video. It needs `NVIDIA_API_KEY` for drawings that are not cached |
| Video pitch (≤ 2:00) | `TODO: paste YouTube/Vimeo link (set to Public or Unlisted)` |
| Pitch deck (PDF) | `project_submission/PlanSight_pitch_deck.pdf` (upload it, and link the GitHub copy) |

## Credits and licences

The drawings in the demo report keep their own licences: CubiCasa5K is CC BY-NC-SA 4.0, Swiss Dwellings is CC BY 4.0, and IFC models are licensed per file (see `data/<source>/LICENSE.md`). They are shown here for non-commercial use in the Junction challenge. Ultralytics is AGPL-3.0 and is kept in a separate optional environment.
