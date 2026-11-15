# DrawingQA

A production-oriented first version of an evidence-first question-answering system for construction drawings. When a drawing comes with its source CAD/BIM file (a same-named DXF, or the IFC model a plan was rendered from), questions are answered deterministically from that file. Otherwise it asks interchangeable Nemotron and Qwen vision endpoints, validates structured visual evidence and counts detected instances in Python. Every answer goes into an interactive report for human review.

Model weights are served separately; the Python client also works from a laptop using a remote endpoint. A door/window/fixture detector trained on labels generated from the Swiss DXF data (`scripts/detector/`, mAP@50 0.93 on Swiss validation tiles) is not yet part of the pipeline. Start with [INSTALLATION.md](INSTALLATION.md); current scores, run times and open work are in [NEXT_STEPS.md](NEXT_STEPS.md).

## Project description

Build a program that answers natural language questions about construction drawings and building information models accurately and verifiably.

Construction site engineers spend hours every day looking for information. General-purpose AI models frequently fail when analyzing complex technical drawings. This is one of the hardest unsolved problems in construction tech, and we want to see what a fresh approach can do with real data.

Create a program which answers natural language questions about construction drawings. You will get a dataset of real technical drawings in many formats and languages together with example questions. A great answer is correct and shows where in the drawing it came from so a human can verify it in seconds.

We are interested in solutions which next year’s frontier models won’t make obsolete. They can be, for example, hybrid solutions which combine frontier models, computer vision, 3D, geometry or small fine-tuned models.

## Judging criteria

| Criterion | Weight | First-version implementation |
|---|---:|---|
| Accuracy of answers | 25% | Deterministic DXF and IFC take-off routes where source files exist (99–100% on those folders), shared counting rules, validated output schema, abstention on ambiguity, per-category and per-dataset evaluation |
| How well the solution visualizes its reasoning so a human can verify the answer | 25% | Numbered evidence boxes (DXF entities mapped onto the PDF page), source pages, concise justifications, IFC entity GUIDs and review exports |
| Creativity | 25% | Using the drawing's own CAD/BIM source, reusable evidence inventories, deterministic counting, detector trained on CAD-generated labels, interchangeable model backends |
| Low inference cost | 25% | Native routes need no model call; a reduced-request mode answers many questions from one shared inventory per drawing (−53% requests on CubiCasa); content-addressed caching, canonical bilingual count intents, bounded views, token, latency and GPU-hour accounting |

The report shows observable evidence and short justifications, not a model's private reasoning trace. A predicted bounding box is an inspectable claim, not proof of correctness.

## What is implemented

- Recursive input discovery across the dataset's subfolders, with explicit errors for ambiguous bare filenames.
- Raster image ingestion, rendered PDF pages and multi-page TIFFs. Optional native IFC entity counting avoids vision inference for supported questions.
- DXF sibling route (`dxf.py`): PDFs with a same-named semantic DXF are answered from its layers, with evidence boxes mapped onto the PDF page.
- IFC take-off route (`ifc_takeoff.py`): quoted storey, room and model questions in nine languages (counts, wall/slab quantities, room and storey areas), for `.ifc` files and for plans rendered from them.
- A common OpenAI-compatible HTTP adapter for local vLLM servers and compatible hosted services. No dependency on the OpenAI SDK or an OpenAI account.
- Qwen3-VL-8B-Instruct and Nemotron 3 Nano Omni presets, plus an optional NVIDIA-hosted Nano Omni preset. A per-model `bbox_scale` accepts the 0..1000 box convention these models use.
- A guard that flags runs of identical boxes stamped along one line, a common invented-output pattern on dense plans.
- Canonical English/Finnish intents for the sample's 12 counting categories. Other questions use the general evidence-grounded path.
- Inventory mode for instance-based counts and a direct-answer baseline for comparison.
- Three run profiles (`--profile normal | reduced | gpu`): one request per question, one shared inventory per drawing (`inventory.py`), or the shared inventory against your own vLLM server on a rented GPU (`scripts/gpu/`).
- Source coordinates, crop remapping, overlap deduplication, explicit incomplete/uncertain outcomes and retained proposed answers.
- Cached extraction, bounded concurrency, retry handling, resumable batch runs and machine-readable predictions.
- An HTML review report with model and dataset breakdown tables (including which route answered), filters, zoom, evidence inspection and downloadable human-review decisions.
- Accuracy with the dataset's scoring rules (exact counts, 2% tolerance on areas, lengths and volumes), count error, answer coverage, latency/token/cost summaries and paired run comparisons.
- `scripts/sample_questions.py` draws reproducible per-folder samples; `scripts/detector/` generates detector training tiles from the Swiss DXFs and trains a YOLO model.

## Data layout

The drawings live under `data/`, with the question files next to them. Subfolder names are preserved; there is no requirement to flatten or rename files. Files with the same name and a different extension are the same drawing in another format. The dataset's own description is in [README_2.md](README_2.md).

| Path | Contents | Questions |
|---|---|---:|
| `data/questions.jsonl` | Full question set | 127,044 |
| `data/questions_sample.jsonl` | Sample (four CubiCasa plans, English and Finnish) | 66 |
| `data/cubicasa5k/` | Finnish floor-plan images (4,787 with questions) | 85,514 |
| `data/swiss_dwellings/` | Swiss floors as PDF, DXF and PNG (1,040) | 37,584 |
| `data/ifc_models/` | IFC models with PDF/SVG plans rendered from them | 3,062 |
| `data/ifc_bench/` | IFC models with free-text IFC-Bench questions | 820 |
| `data/opening_design/` | Real drawing sets with door/finish schedules | 64 |
| `data/wikimedia/`, `data/habs/`, `data/finna/`, `data/europeana/` | Additional drawing collections | none |

Each JSONL line is one object:

```json
{"path":"data/cubicasa5k/colorful-10052.png","question":"How many windows are shown on this floor plan?","answer":7,"unit":"count","type":"count"}
```

`path` and `question` are required. `answer`, `unit`, `type`, `id`, and a 1-based `page` are optional. An omitted `answer` means an unlabeled question, not an answer of zero. With no `page`, count questions inspect all pages, subject to configured limits; multi-page counts remain proposed occurrence counts requiring review because elements can repeat across sheets. General questions on multi-page documents require an explicit page. Prefer explicit paths relative to `data/`. A bare filename is accepted only if it resolves uniquely. Paths beginning with `data/` are also accepted when they refer to the configured data root. Paths outside the root are rejected.

**Ground-truth answers are never included in model prompts or extraction-cache keys.** They are attached only for evaluation and displayed separately in the report.

## Quick start

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -r requirements-ifc.txt -r requirements-dxf.txt   # native IFC and DXF routes
drawingqa demo --output runs/demo
```

The demo is synthetic and clearly labeled. It tests the pipeline/report without API credentials or model downloads; it is not a model-accuracy result.

After starting or configuring a model endpoint as explained in [INSTALLATION.md](INSTALLATION.md):

```bash
drawingqa scan --data-root data
drawingqa doctor --config configs/models.yaml
drawingqa evaluate --questions data/questions_sample.jsonl --data-root data --models qwen nemotron --config configs/models.yaml --output runs/comparison
```

To measure every dataset folder without scoring all 127k questions, sample whole drawings per folder:

```bash
python scripts/sample_questions.py --per-folder 6 --output runs/samples/per-folder-6.jsonl
drawingqa evaluate --questions runs/samples/per-folder-6.jsonl --data-root data --models nemotron_nvidia --output runs/per-folder-6
```

Open the run's `report.html`, or serve it locally:

```bash
python -m http.server 8080 --bind 127.0.0.1 --directory runs/comparison
```

Then visit [http://127.0.0.1:8080/report.html](http://127.0.0.1:8080/report.html). The report's review edits remain in your browser until you export them. They do not silently modify model predictions or the original question labels.

## Three ways to run

Profiles in `configs/models.yaml` bundle the model and the request strategy. Explicit flags (`--models`, `--strategy`, `--concurrency`, `--no-native-routes`, ...) override the profile.

| Profile | Model | Strategy | When to use |
|---|---|---|---|
| `normal` | `nemotron_nvidia` (hosted) | `per_question`: one request per question; canonical count questions share one per category | Reference accuracy |
| `reduced` | `nemotron_nvidia` (hosted) | `per_drawing`: one request per page per group (openings, fixtures, spaces, structure) answers every recognized question; Python selects, counts and sums | Fewer requests on the rate-limited free API |
| `gpu` | `nemotron_gpu` (your vLLM on a Runpod H100) | `per_drawing`, concurrency 48 | Full-data runs; cost is reported as GPU time |

```bash
drawingqa evaluate --profile reduced --questions runs/samples/per-folder-6.jsonl --data-root data --output runs/reduced
scripts/gpu/runpod_up.sh <pod-ip> <ssh-port>      # once the pod is rented; see INSTALLATION.md, section 3C
drawingqa evaluate --profile gpu --questions data/questions.jsonl --data-root data --output runs/gpu-full
```

Questions the inventory cannot express (free text, schedules, multi-page general questions) and inventories that fail (for example a looping list on a dense sheet) fall back to the per-question path automatically, once per drawing and group. `--no-native-routes` sends Swiss and `ifc_models` PDFs to the vision model even though their DXF/IFC source exists, for a fair "any image" measurement.

## Current results

Per-folder sample (805 questions, 6 drawings per folder, Swiss drawings from held-out sites), `nemotron_nvidia` for the vision questions. Full details, run-time estimates for the whole dataset and open work are in [NEXT_STEPS.md](NEXT_STEPS.md).

| Dataset | Accuracy (all labelled) | Coverage | Answered by |
|---|---:|---:|---|
| `swiss_dwellings` | 99.1% | 99.1% | DXF route |
| `ifc_models` | 100% | 100% | IFC take-off route |
| `cubicasa5k` | 43.1% | 78.4% | VLM |
| `opening_design` | 12.5% | 79.7% | VLM |
| `ifc_bench` | not gradable (free text) | 0% | — |

Larger checks of the native routes: the DXF route scores 99.0% on 18,528 held-out Swiss questions; the IFC take-off route scores 100% on all 3,062 `ifc_models` questions, with rules derived on that same set. Weighting by question count, the projected full-data score is about 60%, held back by CubiCasa (two thirds of all questions).

The native scores come from the drawings' own source files, so they measure the routing, not image understanding. On the same PDFs with `--no-native-routes`, Nemotron alone scores 36% on Swiss and 1% on `ifc_models` plans.

Normal versus reduced profile, same 468 questions on 15 drawings, empty cache (`runs/reduced-compare`):

| Dataset | Requests, normal | Requests, reduced | Accuracy, normal | Accuracy, reduced |
|---|---:|---:|---:|---:|
| `cubicasa5k` | 51 | 24 (−53%) | 43.1% | 39.2% |
| `swiss_dwellings` (VLM only) | 214 | 156 (−27%) | 36.0% | 33.6% |
| `ifc_models` plans (VLM only) | 152 | 150 | 1.3% | 1.3% |

On dense Swiss sheets the "spaces" inventory often loops until the token cap and falls back to per-question requests, which limits the saving there. `ifc_models` questions mostly name a storey or model and fall back. The GPU profile has not been benchmarked yet; the estimate is 5–11× the hosted rate (see [NEXT_STEPS.md](NEXT_STEPS.md)).

## Model comparison

| Alias | Actual model | Default endpoint |
|---|---|---|
| `qwen` | `Qwen/Qwen3-VL-8B-Instruct` | Local port 8001 |
| `nemotron` | `nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16` | Local port 8002 |
| `nemotron_nvidia` | `nvidia/nemotron-3-nano-omni-30b-a3b-reasoning` | NVIDIA hosted API; requires `NVIDIA_API_KEY` |
| `nemotron_gpu` | `nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-FP8`, served as `nemotron-omni` | Rented GPU through the SSH tunnel on local port 8002 |

Changing `model`, `base_url` or other generation settings requires only configuration changes. The hosted alias serves Nano Omni under NVIDIA's own precision and revision; report it as a separate run when comparing results. The GPU alias uses the FP8 checkpoint (NVIDIA reports under 1% accuracy loss versus BF16), so it is also a separate measurement. Both `nemotron` and `nemotron_gpu` use local port 8002; only one can be up at a time. All Nemotron presets need `chat_template_kwargs.enable_thinking: false`, otherwise reasoning uses up the whole token budget. All presets use `bbox_scale: 1000`, because these models return boxes on a 0..1000 scale; boxes are normalized before validation. Hosted runs at concurrency 4 hit HTTP 429 rate limits; use 2 and `--resume` to retry failed rows. Keep preprocessing, question set, quantization and generation settings recorded. Changing only the serving host can also change speed and cost.

```bash
drawingqa evaluate --questions data/questions.jsonl --data-root data --models qwen --config configs/models.yaml --output runs/qwen
drawingqa evaluate --questions data/questions.jsonl --data-root data --models nemotron --config configs/models.yaml --output runs/nemotron
drawingqa compare runs/qwen runs/nemotron --output runs/paired
```

Use separate model runs if one GPU cannot hold both models at once. A small smoke run with `--limit 4` is recommended before a full batch. `--resume` resumes a compatible interrupted run; configuration/source mismatches must not be treated as valid cached predictions.

The CLI also accepts `--mode direct`, `--mode inventory`, `--views overview`, `--views tiles` and `--concurrency N` overrides for `ask`/`evaluate`. Direct mode still asks for inspectable evidence and abstains when the count and evidence disagree; it is not an unrestricted answer-only baseline. Known count intents cache one category inventory per source/view, rather than precomputing every category on first upload.

For `.ifc` inputs, routing is native regardless of the selected model alias. The actual model ID identifies the adapter (`ifc-takeoff-v1` or `ifc-entities-v1`; `dxf-layers-v1` for the DXF route), and the report's dataset table shows the route per folder. Repeated native IFC results under two aliases are not two independent VLM measurements; analyze raster/PDF model comparisons separately.

For a `.pdf` with a same-named `.dxf` beside it (the `swiss_dwellings` set), supported door, window, fixture, apartment, named-room and area questions are answered from the DXF's semantic layers (`DOOR`, `WINDOW`, `FIXTURE`, `ROOM_OUTLINE`, `ROOM_TEXT`, `UNIT_OUTLINE`) and drawn back onto the PDF page as `dxf_entity` evidence. Areas come from room outline polygons and are sent to review when they disagree with the printed area. On held-out Swiss sites (odd site numbers, 18,528 questions) this route scores 99.0%. Set `pipeline.use_dxf_sibling: false` to measure the VLM on the same PDFs instead.

Questions that name a storey, room or model in quotes ("How many doors are shown on the floor plan of 'Level 1'?", "What is the net floor area of the room 'Kitchen (A103)' on 'Level 1', in m²?") are answered by the IFC take-off route, both for `.ifc` files and for floor plans `<model>_plan-NN-<level>.pdf` next to `<model>.ifc` (the `ifc_models` set). It counts elements by storey containment, sums stored wall/slab quantities with unit scaling, and takes room areas from the stored net area or, failing that, the IfcSpace geometry. On all 3,062 `ifc_models` questions in nine languages it scores 100%; the rules were derived on this same set. Set `pipeline.use_ifc_sibling: false` to send those PDFs to the VLM instead.

## Evaluation interpretation

Counts must match exactly; areas, lengths and volumes (`m`, `m2`, `m3`) match within 2%, as in the dataset's scoring rules.

Separate proposal diagnostics report the accuracy and coverage of unaccepted proposed answers, which helps compare direct-model outputs without presenting them as verified results. Per-model usage includes requests, tokens, cache hits, API time and configured token-cost estimates.

Exact accuracy over labeled questions includes errors and abstentions in its denominator. Answered-only accuracy and coverage are reported separately so abstaining cannot inflate the headline score. Numeric count error measures how far numeric predictions are from labels; it is not defined for unsupported/non-numeric predictions. Metrics with no applicable labels remain unavailable.

The provided sample has 66 rows but only four drawings and 33 underlying image/category facts. English/Finnish duplicates are useful for interface consistency, not independent evidence of generalization. Canonical count questions may reuse the same extraction, so their agreement is not an independent multilingual model test. Use new drawing/property groups for held-out testing and keep near-duplicates together. Inspect both zero and nonzero cases. The model can get a total right while identifying the wrong objects.

Token-priced cost estimates require rates in `configs/models.yaml`; unset rates remain unknown. For a rented GPU, set `gpu_cost_per_hour` on the model; `summary.json` then reports `estimated_gpu_cost_usd` as run wall time × rate, and the report shows it as GPU time. That covers the run only, not pod start-up, model download or idle time before you stop the pod.

## Limits of this version

- Measured VLM accuracy comes from small samples (CubiCasa: 6 drawings, 102 questions, about ±10 points). The local Qwen and Nemotron presets have not been run; results so far use the NVIDIA-hosted Nano Omni.
- The reduced profile trades a few accuracy points for fewer requests (CubiCasa 43.1% → 39.2% on 51 questions, within sample noise but not shown to be free). The GPU scripts follow the model card and Runpod's documentation but have not been run end to end.
- The inventory remains a VLM prediction. Schema validation and deterministic counting improve auditability, but do not establish whether every object was found. On CubiCasa, doors, windows and bathroom fixtures are currently 0% correct.
- Native IFC support covers entity/explicit-label counts and the take-off question templates (quoted storey, room or model). It exposes GUIDs and properties, not an interactive 3D BIM viewer or general geometric reasoning engine. Free-text IFC-Bench questions are not answered, and IFC answers on PDFs list GUIDs without drawing boxes on the page.
- The detector is trained on Swiss-style drawings only and is not integrated; its accuracy on CubiCasa is unknown.
- DXF is only read as a sibling of a PDF and only with the Swiss semantic layer convention; arbitrary DXF, DWG, RVT and other CAD/native formats need a separate approved renderer/export adapter. Unsupported files are reported rather than guessed from their extension.
- Source image quality is a hard limit. Enlarging a low-resolution image cannot recover missing symbols or text. Unknown scales do not support trustworthy physical dimensions or areas.
- Finnish sample questions map to English canonical intents. Arbitrary language coverage and Finnish drawing OCR still require validation; Nano Omni's official model card lists English support.
- Adjacent window grouping, doors with multiple leaves, duplicate sheets, alternative layouts and room naming policies need dataset-specific review. Multi-page counts refer to depicted instances, not deduplicated physical building elements across different drawing views.
- More detailed tile extraction can cost more and split/duplicate objects. Start with overview mode; validate the effect of tiles before enabling them at scale.
- This is a local research/service component, not a deployed multi-tenant product. Add authentication, access control, encrypted object storage, resource quotas, monitoring, a durable queue and data-retention policies before exposing it to untrusted users.

The included [synthetic review report](examples/demo_report/report.html) can be opened without model credentials. See [docs/VALIDATION.md](docs/VALIDATION.md) for the completed checks and untested deployment components.

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for extension points and the path toward specialist detectors, OCR, geometric validation and a production service.

## Upstream references

Model configuration was checked against primary documentation on 14 November 2026:

- [Qwen3-VL-8B-Instruct model card](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct)
- [Nemotron 3 Nano Omni model card and serving guide](https://huggingface.co/nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16), [FP8 checkpoint](https://huggingface.co/nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-FP8) and [NVIDIA inference guide](https://docs.nvidia.com/nemotron/latest/nemotron/omni3/inference.html)
- [Runpod SSH connection guide](https://docs.runpod.io/pods/configuration/use-ssh)
- [NVIDIA hosted models (build.nvidia.com)](https://build.nvidia.com); the earlier `nvidia/nemotron-nano-12b-v2-vl` endpoint is retired and returns HTTP 410
- [CubiCasa5K dataset and annotations](https://github.com/CubiCasa/CubiCasa5k)

Model weights, serving software and source drawings retain their respective licenses. CubiCasa5K's published license is non-commercial; obtain suitable rights before commercial reuse. PyMuPDF is distributed under AGPL/commercial licensing; review the appropriate license or replace the PDF renderer for your deployment. The detector scripts use Ultralytics (AGPL-3.0) and torch in a separate `.venv-detector`; they are not dependencies of the `drawingqa` package.
