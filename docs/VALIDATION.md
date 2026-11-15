# Validation record

Last updated 15 November 2026, with Python 3.13.5, ifcopenshell 0.8.5 and ezdxf 1.4.5.

## Automated tests

**136 unittest tests pass.** They cover provider request/response handling, retries, 0..1000 box rescaling, cache isolation and concurrent request deduplication, English/Finnish intents, ground-truth isolation, evidence mapping and fusion, the repeated-box guard, PDF/TIFF ingestion, safe file resolution, native IFC classification, evaluation (including the 2% measurement tolerance), report escaping and batch resume integrity. They also cover:

- the DXF route: question parsing in English, German, French and Italian, and counts, areas and abstentions on a synthetic DXF/PDF pair (`tests/test_dxf.py`);
- IFC take-off: parsing in several languages, including apostrophes in room names, plan-to-model pairing, and counts, unit-scaled sums and unknown-storey abstention on a generated IFC (`tests/test_ifc_takeoff.py`);
- the reduced-request strategy: question-to-selector mapping (canonical, Swiss and IFC wording, model-scoped questions rejected), counts, areas and sums from an inventory, two questions sharing one request per group, a failed inventory requested once and then falling back, and the shipped `normal`/`reduced`/`gpu` profiles (`tests/test_inventory.py`);
- GPU-time cost in the run summary when a model has `gpu_cost_per_hour` (`tests/test_cli.py`).

The DXF and IFC tests are skipped when `ezdxf` or `ifcopenshell` is not installed.

```bash
python -m unittest discover -s tests -v
```

## Live and dataset checks

- **Hosted model:** `nemotron_nvidia` (`nvidia/nemotron-3-nano-omni-30b-a3b-reasoning`) was called end to end. `drawingqa doctor --probe` and evaluation runs succeed with thinking disabled and `bbox_scale: 1000`. The local `qwen` and `nemotron` vLLM presets have not been run.
- **Per-folder sample** (`runs/per-folder-6-v2`, 805 questions, 0 errors):

  | Dataset | Accuracy |
  |---|---|
  | Swiss | 99.1% |
  | `ifc_models` | 100% |
  | CubiCasa | 43.1% |
  | `opening_design` | 12.5% |
  | `ifc_bench` | not gradable |

  The report renders the model and dataset tables and draws DXF evidence boxes on the PDF page; this was checked in a browser.
- **DXF route on the dataset:**
  - Dev sites (even): 99.1% on 7,212 questions from 200 files.
  - Held-out sites (odd): 99.0% on 18,528 questions from 513 files.
  - Weakest types: apartment area 96.1%, floor total 95.1%.
- **IFC take-off on the dataset:** 3,062 of 3,062 `ifc_models` questions correct in nine languages, in about 50 seconds. The rules were derived on this set; there is no held-out split.
- **Vision model on the native folders** (`--no-native-routes`, same PDFs and questions): Nemotron scores 36% on Swiss and 1% on `ifc_models` plans, against 99% and 100% for the native routes.
- **Reduced versus normal strategy** (`runs/reduced-compare`, 468 questions on 15 drawings, empty cache): requests fall by 53% on CubiCasa (51 → 24) and 27% on Swiss (214 → 156), with accuracy 43.1% → 39.2% and 36.0% → 33.6%. The Swiss run had 22 errors in normal mode and none in reduced mode. The "spaces" inventory often loops on dense Swiss sheets and falls back.
- **Detector:** training data was checked by drawing the generated boxes over tiles, and they align with the drawn symbols. After 15 epochs, validation on held-back even-site tiles gives mAP@50 0.93 (precision 0.93, recall 0.87; door 0.99, shower 0.86). This is Swiss-style data only.
- **Demo:** the synthetic CLI demo still generates predictions, metrics and a report. `examples/demo_report/report.html` uses only synthetic fixture responses.

## What is not validated

- **Accuracy and cost:** VLM accuracy comes from small samples; CubiCasa uses 6 drawings, an uncertainty of about ±10 points. No full-dataset run has been made; see [NEXT_STEPS.md](../NEXT_STEPS.md) for time estimates. Pricing is unset, so cost is reported as unknown.
- **IFC take-off:** has no held-out check.
- **Detector:** not tested on CubiCasa and not integrated into the pipeline.
- **Deployment:** the Runpod scripts (`scripts/gpu/`) pass a shell syntax check and follow the model card and Runpod's SSH documentation, but no pod has been started; GPU throughput, FP8 accuracy on this data, local vLLM compatibility and multi-user operation are untested.

The run metrics distinguish accepted final outputs from proposed answers requiring review. Evidence boxes from the VLM remain model predictions until independently checked. Zero detections are not proof of absence. API-time metrics exclude image preparation and local infrastructure costs. Known usage from failed requests is retained when a run is resumed; charges incurred before a process could persist a response may remain unknowable.
