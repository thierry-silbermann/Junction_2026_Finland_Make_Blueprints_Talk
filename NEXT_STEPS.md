# Next steps

Status on 15 Nov 2026. Two of the five labelled folders are essentially solved because the folder contains the source CAD/BIM file next to each drawing. The rest, and above all CubiCasa (67% of all questions), still depends on the vision model and is the main gap.

## Current scores

Sample: 6 drawings per folder (3 for `opening_design`), all questions for each drawing, Swiss drawings from held-out odd sites only. Run: `runs/per-folder-6-v2` (`report.html` has a per-dataset table and a dataset filter).

| Dataset | Questions in full data | Sample questions | Accuracy (all labelled) | Coverage | Answered by |
|---|---|---|---|---|---|
| `swiss_dwellings` | 37,584 | 214 | 99.1% | 99.1% | DXF layers (`dxf.py`) |
| `ifc_models` | 3,062 | 182 | 100% | 100% | IFC take-off (`ifc_takeoff.py`) |
| `cubicasa5k` | 85,514 | 102 | 43.1% (55% of answered) | 78% | VLM (Nemotron Nano Omni) |
| `opening_design` | 64 | 64 | 12.5% | 80% | VLM |
| `ifc_bench` | 820 | 243 | not gradable | 0% | none (free-text answers) |

Larger validations of the native routes, run outside the pipeline:

- Swiss DXF: 99.0% on 18,528 held-out questions (odd sites), 99.1% on the dev sites. The weakest types are apartment area (96.1%) and floor total (95.1%).
- IFC take-off: 100% on all 3,062 `ifc_models` questions in nine languages. The rules were derived on this same set, so there is no held-out check yet.

**Projected score on all 127,044 questions: about 60%.** This weights each folder's sample accuracy by its question count. The CubiCasa figure comes from only 6 drawings (102 questions), so the projection is uncertain by roughly ±10 points; an earlier 4-drawing run scored 55%.

CubiCasa by category (sample): floors, sinks, saunas, bedrooms and balconies are mostly right. Doors, windows, showers/bathtubs and WC rooms are 0/38.

## Time and cost on the full data

| Dataset | Route | Measured rate | Full-data estimate |
|---|---|---|---|
| `swiss_dwellings` (1,040 drawings) | local DXF | 222 questions in 7.5 s, including page rendering | 20–30 min |
| `ifc_models` (~620 PDFs + IFCs) | local IFC | 3,062 questions in 50 s, plus PDF rendering | 5–10 min |
| `ifc_bench` | none | — | seconds (all unsupported) |
| `opening_design` | VLM | 62 requests, about 2 s each | about 2 min (already fully run) |
| `cubicasa5k` (4,787 drawings) | VLM | about 8.5 requests per drawing, about 6.8 s each, 2.5k input + 0.7k output tokens per request | about 40,000 requests: about 38 h at concurrency 2, ideally about 17 h if the rate limit allows 40 requests/min; about 100M input and 27M output tokens |

### The three run profiles

`--profile normal` is the measured setup above. `--profile reduced` shares one inventory per page and element group. On a 468-question sample (`runs/reduced-compare`, empty cache) it needed 24 instead of 51 requests on CubiCasa (−53%), at 39.2% against 43.1% accuracy, and 156 instead of 214 on Swiss VLM questions (−27%), at 33.6% against 36.0%. On full CubiCasa that is roughly 19,000 requests instead of 40,000, so about 18 h on the hosted API instead of 38 h.

`--profile gpu` runs the reduced strategy against Nemotron FP8 on a rented Runpod H100 (`scripts/gpu/runpod_up.sh`, setup in INSTALLATION.md, Option C). It has not been benchmarked. The estimate from published throughput figures is 90–200 requests a minute, 5–11× the hosted rate:

| Scenario | Requests | Hosted API (18/min) | 1× H100 (90–200/min) | H100 cost (~$2.5/h) |
|---|---:|---:|---:|---:|
| CubiCasa, normal | ~40,000 | ~37 h | 3.5–7.5 h | ~$10–20 |
| CubiCasa, reduced | ~19,000 | ~18 h | 1.5–3.5 h | ~$4–9 |
| All folders through the VLM (`--no-native-routes`), normal | ~80,000 | ~74 h | 7–15 h | ~$18–40 |

Add about 15–20 minutes of pod time on the first start for the install and the 33 GB download. Benchmark 6 CubiCasa drawings first (the commands are in INSTALLATION.md); `summary.json` then gives the actual rate and `estimated_gpu_cost_usd`.

Notes:

- Concurrency 4 hit NVIDIA HTTP 429 rate limits (38 of 805 rows); `--resume` recovers them. Pricing is unset in `configs/models.yaml`, so cost is reported as unknown, not zero.
- The native routes cost nothing and need no API key.
- Parallel runs overwrite each other's extraction cache entries (the cache lock is in-process only). Run one evaluation at a time, or give each run its own cache.

## Detector (trained, not integrated)

- Training data: 3,156 tiles cut from the PNGs of even-numbered Swiss sites (526 drawings), with about 61k boxes generated from the DXFs (doors, windows, toilets, washbasins, bathtubs, showers). Odd sites stay held out.
- Model: YOLO11n at 640 px input, 15 epochs, CPU, in the separate `.venv-detector` because Ultralytics is AGPL-3.0. Apple MPS produced NaN losses with or without AMP.
- Result: training finished in 1.75 h (about 7 min per epoch). Final validation on held-back even-site tiles (318 tiles, 5,172 boxes): mAP@50 0.93, mAP@50-95 0.78, precision 0.93, recall 0.87. Per class mAP@50: door 0.99, bathtub 0.99, window 0.94, toilet 0.92, washbasin 0.91, shower 0.86 (shower recall is only 0.70, with 78 examples). Inference takes about 29 ms per 640 px tile on CPU. Weights: `runs/detector/swiss_symbols/weights/best.pt` (5.5 MB); log: `runs/detector/train.log`.
- Risk: it learns the Swiss drawing style. CubiCasa scans look very different, so accuracy there is unknown until tested.

## What is missing

1. **CubiCasa accuracy** is what decides the overall score. Doors, windows and bathroom fixtures are 0% with the VLM.
2. **The detector is not wired into the pipeline** and has not been tested on CubiCasa.
3. **`opening_design`** (door and finish schedules) is 12.5%. These are tables on large sheets that the VLM reads poorly.
4. **`ifc_bench`** answers are free text ("list every room and its area"), which exact matching cannot grade. The IFC data needed to answer many of them is now reachable through `ifc_takeoff.py`.
5. **Evidence on the drawing for the IFC route**: IFC answers list element GlobalIds, but no boxes are drawn on the PDF. The `.svg` export next to each plan could provide the positions.
6. **Held-out checks**: the IFC rules have no held-out split. CubiCasa needs a larger sample for a trustworthy estimate.

## Next steps to test, in order

1. **Run the native folders in full** (free, about 40 min):

   ```bash
   python scripts/sample_questions.py --per-folder 100000 --folders swiss_dwellings ifc_models --output runs/samples/native-full.jsonl
   drawingqa evaluate --questions runs/samples/native-full.jsonl --data-root data --output runs/native-full --models nemotron_nvidia
   ```

   This confirms about 99% and 100% at full scale. Note that `sample_questions.py` keeps only odd Swiss sites; drop the filter to also score the even sites, which the detector trains on but which the DXF route never saw during rule design.

2. **Get a reliable CubiCasa baseline**: 100 drawings (about 850 requests, about 1.5–2 h). This narrows the ±10-point uncertainty before spending 1–2 days on the full set.

   ```bash
   python scripts/sample_questions.py --per-folder 100 --folders cubicasa5k --output runs/samples/cubicasa-100.jsonl
   ```

3. **Detector on CubiCasa**: when training finishes, run it on the same 100 CubiCasa drawings and compare door, window, toilet, sink and shower/bathtub counts with the labels. Integrate it into the pipeline only if it beats the VLM on those categories. If the style gap is too large, fine-tune it on a small hand-checked set of CubiCasa drawings.
4. **`opening_design` schedules**: extract tables from the PDF text layer (PyMuPDF `find_tables`) and answer count and size questions from them. Only 64 questions, but the same approach applies to any real drawing set with schedules.
5. **IFC evidence boxes**: map IFC GlobalIds onto the PDF through the `.svg` sibling, so reviewers can see what was counted.
6. **Benchmark the GPU profile**: rent the H100, run `scripts/gpu/runpod_up.sh`, score 6 CubiCasa drawings with `--profile gpu`, and compare its accuracy with the hosted run (FP8 versus NVIDIA's hosted precision) and its rate with the estimate above.
7. **Fix the looping "spaces" inventory** on dense Swiss sheets, for example by splitting rooms from apartments/floors, or tiling the page for that group only. This is what limits the reduced saving on Swiss.
8. **Full CubiCasa run**, only after steps 2–3 have settled the method. On the hosted API use `--profile reduced --resume` at concurrency 2 (about 18 h); on the GPU, `--profile gpu` should take a few hours.

## Are we at a good point?

Partly.

- **Swiss and `ifc_models` are done**: about 32% of all questions, answered near-perfectly, fast, free, and with checkable evidence.
- **The headline score is held back by CubiCasa**: two thirds of the questions, at about 43–55%. The reduced profile halves its request count; a rented GPU should bring a full run down to a few hours.
- **Next**: the detector and a larger CubiCasa sample will show whether the 60% projection can move substantially. Without a CubiCasa improvement, the full-data score stays around 60%.
