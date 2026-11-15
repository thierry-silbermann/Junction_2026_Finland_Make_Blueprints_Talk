# PlanProof: final submission text

Copy each section into the matching field of the Junction submission form.

---

## Project name

**PlanProof**

## Punchline

**Every number from a drawing, with the proof drawn on it.**

Alternatives, if the form needs something shorter or different:

- "Ask a drawing anything. See exactly what was counted."
- "Blueprints that answer, and show their work."

## Short description (for cards and listings, ~300 characters)

PlanProof answers questions about construction drawings, in any language, and draws the evidence on the sheet: every door, room or WC it counted, numbered and clickable. It reads the drawing's own CAD/BIM source when it exists, uses NVIDIA Nemotron Nano Omni when it doesn't, and says "needs review" instead of guessing.

## Description

Site engineers spend hours a day hunting for facts buried in drawings: how many doors on level 3, what is the net area of this bedroom, how many WCs in the apartment. Stoa already answers project questions with *the document and page behind every sentence*. PlanProof takes that one step further for drawings: **the symbol behind every number.**

**How it works.** Every question goes through a router that picks the most reliable source available:

1. **The drawing's own source file.** Many drawings are exports from CAD or BIM. When a PDF has its DXF next to it, PlanProof counts the entities on the DXF layers and maps each one back onto the PDF page as a numbered box. When a plan was rendered from an IFC model, it answers from the model: counting doors and walls per storey, summing wall lengths, slab volumes and room areas, all unit-scaled. Each answer lists the IFC GlobalIds it used. No model call, no tokens, deterministic.
2. **NVIDIA Nemotron 3 Nano Omni for everything else** (scans, raster plans). The model doesn't answer with a bare number. It returns an inventory of evidence boxes in strict JSON, and Python does the counting. The boxes are validated and de-duplicated, and a guard flags invented patterns, such as six "doors" stamped in a straight column. In those cases PlanProof returns *needs review* with its proposed answer, instead of a confident wrong number.
3. **A small symbol detector (YOLO11n, 2.6M parameters)**, trained on labels generated *for free* from the Swiss DXF files, with no hand labelling. It reaches mAP@50 0.89 on held-out sites and already boxes the real doors on Finnish CubiCasa scans it never saw. This is the part next year's frontier models won't make obsolete: it gets cheaper and better with every CAD file Stoa ingests.

**Verification is the product.** Every answer lands in an interactive HTML report: the page, numbered boxes (green = counted, orange = uncertain), a one-line justification, the route that answered (DXF / IFC / VLM), filters per dataset, and a review export. A human can confirm or reject an answer in seconds.

**Results** (held-out sample, 805 questions across 5 dataset folders, dataset scoring rules: exact counts, 2% on areas):

| Dataset | Questions in full set | Accuracy | Route |
|---|---|---|---|
| Swiss Dwellings | 37,584 | **99.1%** (99.0% on 18,528 held-out questions) | DXF layers |
| IFC models (plans and .ifc) | 3,062 | **100%** (3,062 / 3,062, 9 languages) | IFC take-off |
| CubiCasa5K scans | 85,514 | 43% final, 55% of answered | Nemotron Nano Omni |
| Opening Design sets | 64 | 12.5% | Nemotron Nano Omni |

**Low inference cost.** About a third of all questions cost **0 tokens** and ~30 ms each on a laptop CPU. The vision path uses Nemotron 3 Nano Omni, a mixture-of-experts model with **3B active parameters** (30B total). It needs about 0.5 requests per question, because one cached evidence inventory answers both the English and the Finnish version of a question. That averages about **1,000 tokens per question** over the whole 127k-question dataset. A reduced-request mode goes further: one inventory per drawing answers every recognized question, with Python doing the counting, which **halves the requests on CubiCasa** (51 → 24 on our sample). Because Nemotron's weights are open, the same pipeline runs self-hosted on a single GPU in an EU region. A one-command script serves the FP8 checkpoint on a rented H100 (about $2.5/h), with the run cost reported in GPU-hours. That matches Stoa's promise that data is held in the EU and never trained on.

**Why it fits Stoa.** Stoa connects to M-Files, SharePoint and Dalux, where CAD and BIM files sit next to their PDFs. PlanProof turns those pairs into exact answers. It also turns them into an ever-growing training set for the detector, which covers the scans that have no source file. PlanProof can ship as a Stoa skill: "count and prove".

## Challenge

**Stoa: Make Blueprints Talk** (Junction 2026, Finland).

> Build a program that answers natural language questions about construction drawings and building information models accurately and verifiably. A great answer is correct and shows where in the drawing it came from so a human can verify it in seconds.

## How PlanProof scores on each judging criterion

| Criterion (25% each) | What PlanProof does |
|---|---|
| Accuracy | 99–100% where a CAD/BIM source exists; the vision path abstains (*needs review*) instead of guessing; scored with the dataset's own rules |
| Visualized reasoning | Numbered boxes on the actual page, mapped from DXF entities or the model's inventory; IFC GlobalIds; route and justification per answer; review export |
| Creativity | Uses the blueprint's own source as ground truth, and turns CAD files into free detector labels; a guard against invented boxes |
| Low inference cost | 0 tokens for ~1/3 of questions; 3B-active Nemotron for the rest; cache shared across languages; reduced mode with −53% requests on CubiCasa; self-hostable open weights with a one-command GPU setup |

## Technologies used

- **NVIDIA Nemotron 3 Nano Omni** (`nemotron-3-nano-omni-30b-a3b-reasoning`), via NVIDIA's hosted API (build.nvidia.com, `integrate.api.nvidia.com`), with a vLLM preset for self-hosting the open BF16 weights.
- Python 3.13, pydantic (strict evidence schemas), httpx (OpenAI-compatible client, retries and 429 handling).
- PyMuPDF (PDF rendering, page geometry), Pillow.
- ezdxf (DXF layers to evidence boxes), IfcOpenShell 0.8 (IFC take-off, unit scaling, geometry for room areas).
- Ultralytics YOLO11n + PyTorch (symbol detector trained on DXF-generated labels).
- A self-contained HTML/JavaScript review report (no server needed).
- Qwen3-VL-8B preset for model comparison.
- Data: the Junction construction-drawing dataset (Swiss Dwellings, CubiCasa5K, IFC models, Opening Design, IFC-Bench).

## Links

| What | Link |
|---|---|
| Source code | <https://github.com/thierry-silbermann/Junction_2026_Finland_Make_Blueprints_Talk> |
| Live demo (review report) | <https://raw.githack.com/thierry-silbermann/Junction_2026_Finland_Make_Blueprints_Talk/main/project_submission/demo/index.html> (works once the repo is pushed and public; GitHub Pages is an alternative, see README) |
| Video pitch (≤ 2:00) | `TODO: paste YouTube/Vimeo link (set to Public or Unlisted)` |
| Pitch deck (PDF) | `project_submission/PlanProof_pitch_deck.pdf` (upload it, and link the GitHub copy) |

## Credits and licences

The drawings in the demo report keep their own licences: CubiCasa5K is CC BY-NC-SA 4.0, Swiss Dwellings is CC BY 4.0, and IFC models are licensed per file (see `data/<source>/LICENSE.md`). They are shown here for non-commercial use in the Junction challenge. Ultralytics is AGPL-3.0 and is kept in a separate optional environment.
