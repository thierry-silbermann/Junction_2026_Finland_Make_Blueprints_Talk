# DrawingQA review and suggested improvements

Review date: 14 November 2026. Scope: everything under `src/drawingqa/`, `configs/`, `docs/`, `README.md`, `INSTALLATION.md`, packaging files and the examples. The unit tests were **not** run during this review because the local Python environment did not have the project dependencies installed.

Items are ordered by expected impact. Each item names the file it concerns and a concrete next step.

## Status update (15 November 2026)

Since the review, the work went beyond these items: native routes now answer the Swiss and `ifc_models` folders from their DXF/IFC source files (99–100%). Current scores and open work are in [NEXT_STEPS.md](NEXT_STEPS.md).

| Item | Status |
|---|---|
| 2.1 Real model results | Done for the hosted Nano Omni: per-folder results are in the README and `NEXT_STEPS.md`. The local Qwen/Nemotron presets are still unrun |
| 2.2 Intent matching | Open for the VLM path. The native routes have their own multilingual parsers (`dxf.py`, `ifc_takeoff.py`) |
| 2.3 0..1000 box convention | Done: `ModelConfig.bbox_scale`, prompt wording and rescaling before validation |
| 2.4 Guided decoding | Open |
| 2.5 Abstention policy | Partly: a guard now flags runs of identical boxes along one line (`flag_repeated_boxes`). The policy itself is unchanged |
| 2.6 Specialist detector | Trained (mAP@50 0.93 on Swiss validation tiles), not tested on CubiCasa. `data/` has no CubiCasa SVG annotations, so labels are generated from the Swiss DXFs instead (`scripts/detector/`). Not yet integrated |
| 2.7 OCR | Open |
| 2.8 Report UX | Partly: dataset breakdown table, route column and dataset filter. Zoom-to-evidence and review import are open |
| 2.9 Tile cost | Open. Tiles over-counted in testing (33% against 55% for overview), so overview stays the default |
| New: request count | Done: the `reduced` profile (`pipeline.strategy: per_drawing`, `inventory.py`) shares one inventory per page and element group, cutting requests by 53% on CubiCasa and 27% on Swiss VLM questions. The `gpu` profile and `scripts/gpu/` target a rented Runpod H100 and report GPU-hour cost; not yet run on a pod |
| 3.4 Answer scoring | Partly: areas, lengths and volumes match within 2% (dataset rule), in both `evaluation.py` and the report. Text normalization is open |
| 3.x other correctness items | Open |
| 5 Tooling | Partly: `.gitignore` exists. There are no commits yet, and no CI, LICENSE or ruff |
| 6 Docs | Partly: README has a results table, the per-folder sampling command and the three run profiles; restructuring is open |

---

## 1. Summary and strengths

DrawingQA is in good shape for a first version. The things that stand out:

- **Clean layering.** Perception (the VLM call), evidence (the `Extraction`/`Evidence` schema), deterministic counting and evaluation are separate. A future detector or OCR backend can produce the same `Evidence` objects without touching the report or metrics.
- **Ground truth isolation.** Labels never enter prompts, cache keys or `stable_id` (`data.py`, `pipeline.py`). Labels are attached only after inference (`attach_ground_truth`).
- **Careful engineering.**
  - Strict JSON parsing that rejects duplicate keys, non-finite numbers and prose around the object (`providers.py`).
  - Sandboxed path resolution (`data.py`).
  - An HTML report that is safe against script injection.
  - CSV formula escaping (`evaluation.py`).
  - Atomic writes, and resumable runs with identity checks (`cli.py`).
- **Honest metrics.** Abstentions and errors stay in the accuracy denominator. Proposed answers are reported separately, and unknown cost is shown as unknown rather than zero.
- **Thorough documentation of limits**, in `README.md` and `docs/`.

The main gap is not in code quality. The system has never been run against a real model, and a few design choices may reduce measured accuracy and coverage. Those are covered first.

---

## 2. Highest impact for the judging criteria

### 2.1 Produce real model results (Accuracy, 25%)

At review time, `docs/VALIDATION.md` stated that no live endpoint was called. Until there are numbers, the accuracy criterion has nothing to score. (Since done for the hosted model; see the status table.)

- Run `drawingqa evaluate` on `examples/questions_sample.jsonl` with `qwen` and `nemotron`, then run `drawingqa compare`.
- Add a short results table to `README.md`: final accuracy, coverage, proposal accuracy, latency and cost per model. Add a screenshot of `report.html`.
- Use these runs to tune the abstention policy (2.5) and the fusion settings (3.1).

### 2.2 Intent matching only recognizes exact template strings (Accuracy)

`resolve_intent` in `src/drawingqa/intents.py` matches only exact normalized strings built from a fixed set of templates. Any rephrasing goes down the general path, which loses the counting rule, the cache reuse and the inventory mode. Examples are "How many windows does this plan have?" or "Count all the doors in the drawing".

- Add a cheap classification step that returns `{category, scope}`. This could be a small LLM call with a JSON schema, or embedding similarity against the canonical questions plus a threshold.
- If a scope qualifier is detected (floor, room, wing, "on the left"), keep the general path, as the current docstring intends.
- Extend `tests/test_intents.py` with paraphrases, Finnish inflections and negative cases ("windows on floor two" must not map to all windows).

### 2.3 Qwen3-VL bounding-box convention (Accuracy, Verification)

`src/drawingqa/prompts.py` tells every model: "Never use pixel coordinates, percentages or the 0..1000 coordinate convention." Qwen3-VL is trained to ground objects with relative 0..1000 coordinates. Forcing 0..1 floats is likely to make its boxes worse, and outputs above 1 fail `Candidate.valid_box`, so the whole view becomes an error.

- Add a `bbox_format` field to `ModelConfig` in `src/drawingqa/schemas.py`, for example `"unit"` or `"per_mille"`.
- Choose the prompt wording from that field, and normalize to 0..1 after parsing and before validation.
- Compare localization on a few hand-annotated images with both conventions.

### 2.4 Use structured (guided) decoding (Accuracy, Cost)

Both local presets in `configs/models.yaml` use `json_mode: false`. vLLM supports `response_format: {"type": "json_schema", ...}` (guided decoding), which eliminates most `InvalidExtraction` failures and wasted retries. `ModelConfig.no_core_overrides` currently forbids `response_format` inside `extra_body`, so this needs explicit support:

- Add `json_mode: "off" | "object" | "schema"`.
- In `"schema"` mode, send `Extraction.model_json_schema()` as the schema. The prompt can then stop including the whole JSON schema, which saves input tokens on every request.

### 2.5 The abstention policy may cost too much coverage (Accuracy)

In `Pipeline.predict` (`src/drawingqa/pipeline.py`), a count becomes `needs_review` if any of these is true:

- any fused evidence item is `uncertain`;
- any view reports `complete=false`;
- the model's stated count differs from its own inventory;
- there is more than one page.

Combined with the fusion step marking overlapping boxes as uncertain (3.1), most real answers will probably end up as `needs_review`. That makes `exact_accuracy_all_labeled` low even when the proposals are correct.

- After 2.1, compare `proposal_diagnostics.exact_accuracy_all_eligible_labeled` against the final accuracy. If proposals are much better, the policy is too strict.
- Possible refinements:
  - Accept the answer when the uncertain items are all `counted=false`.
  - Accept it when the counts with and without uncertain items are the same.
  - Report a confidence band (minimum to maximum count) instead of a binary status.
- Keep the policy configurable in `PipelineConfig`, so the strict mode remains available.

### 2.6 Add a specialist detector (Creativity, durability)

The challenge asks for solutions that next year's frontier models will not make obsolete. The public CubiCasa5K release ships SVG annotations for doors, windows and fixtures, but the copy in `data/` does not include them. The Swiss DXFs provide equivalent boxes, and a small YOLO-style detector trained on those is realistic in hackathon time (now in progress, see the status table).

- Emit the same `Evidence` objects, with a new `provenance` value such as `"detector"`, as described in extension point 1 of `docs/ARCHITECTURE.md`.
- Use agreement between the detector and the VLM as an acceptance signal, and send disagreements to review. This also strengthens the abstention story in 2.5.

### 2.7 Add OCR for room labels (Accuracy, Verification)

Room counts (bedrooms, bathrooms, WC rooms, saunas, kitchens) depend on short Finnish codes such as MH, KH, WC, S and K. These are small text that VLMs often misread at 2048 px.

- Add an OCR pass (Tesseract with Finnish data, or PaddleOCR) that keeps text boxes.
- Feed the detected labels and their positions into the prompt, or match them against the model's room boxes.
- Show OCR boxes as a separate overlay layer in the report, so a reviewer can see why a room was counted.

### 2.8 Verification UX in the report (Verification, 25%)

`src/drawingqa/report.py` is already good, but a few additions would make "verify in seconds" more literal:

- When an evidence item is clicked, scroll and zoom the viewport to its box. Currently the box is only highlighted, so on a large drawing the reviewer must search for it.
- Keyboard shortcuts: next/previous question, next/previous evidence, and accept/reject.
- Import a previously exported `drawingqa-human-reviews.json` back into the report. Currently reviews live only in `localStorage` or the export.
- Add a CLI command, for example `drawingqa apply-reviews`, that turns human reviews into human-verified accuracy and localization metrics. Then `summary["evidence"]["localization_accuracy"]` is no longer always `None`.

### 2.9 Reduce requests in tile mode (Cost, 25%)

`make_views` in `src/drawingqa/imaging.py` creates one request per tile, plus the overview.

- Coarse-to-fine: run the overview first, and request tiles only if the overview is incomplete or uncertain, or only for the regions containing uncertain boxes.
- Alternatively, send several tiles in one request with explicit view IDs, if the model handles multiple images well.
- Record views per question in `Usage`, so the cost of tiling is visible in the report.

---

## 3. Correctness and robustness

### 3.1 Evidence fusion heuristics (`fuse_evidence`, `pipeline.py`)

- **Partial boxes at tile edges.** An object cut by a tile boundary produces a small partial box. Its IoU against the full overview box is low, so it is not merged and gets counted twice. Use intersection over the smaller box's area. Also treat a box touching a tile edge (not a page edge) as a partial observation that should be merged rather than counted.
- **The "moderate overlap" pass is too broad.** The loop that marks `0.2 < IoU` pairs as uncertain does not check `kind(...)` and does not exclude boxes from the same view. Two adjacent sinks or windows from one view can mark each other uncertain, which pushes the answer to `needs_review` (2.5).
- **Repeated warnings.** The overlap warning is appended once per pair. It is deduplicated at the end, but it can also short-circuit after the first match.
- **Quadratic loop.** Fusion compares each item with all fused items. This is fine at current sizes, but grouping by `(page, kind)` first is cheap.

### 3.2 Inconsistent count definitions

The inventory prompt (`prompts.py`) says: "Set answer to the number of counted=true, uncertain=false candidates". The stated-count check in `pipeline.py` uses the same definition. However, the application's own proposed count is `sum(e.counted for e in result.evidence)`, which also includes uncertain items. Choose one definition, probably "counted, including uncertain, as the proposal, with the uncertain ones flagged". Then align the prompt, the check and the docs.

### 3.3 Direct mode with a null answer becomes an error

In direct mode, if any view returns `answer: null`, the pipeline raises `ValueError("Direct count must return a nonnegative JSON integer")` and the row becomes `status="error"`. A model that abstains is not an operational failure. Map this case to `needs_review`, keeping the localized evidence.

### 3.4 Text answer scoring is very strict (`evaluation.py`)

`exact_match` compares strings after only `strip()`. It is case-sensitive, and `"7"` does not match `7`. General questions are likely to return strings such as `"Yes"`, `"7 m"` or `"Kitchen"`.

- Add a normalization step: casefold, parse numeric strings, map yes/no and kyllä/ei to booleans, and strip units when the expected `unit` is known.
- Keep the raw strict match as an additional metric, so the change is transparent.
- Mirror the logic in the report's JavaScript `equal`, so the "matches" badge agrees with `summary.json`.

### 3.5 The extraction cache key is broader than necessary

The per-view cache key in `Pipeline.predict` hashes `model_config.model_dump()`. That includes `timeout_seconds`, `max_retries`, `api_key_env`, `input_cost_per_million` and `output_cost_per_million`, so adding a price invalidates every cached extraction. Hash only the fields that change the model output: `model`, `base_url`, `max_tokens`, `temperature`, `json_mode` and `extra_body`. `prediction_fingerprint` can stay broad, because it identifies the run, not the extraction.

### 3.6 Repeated work per question

- `render_pages` (`imaging.py`) fully rasterizes the PDF page or decodes the image before checking `if not target.exists()`. With 66 questions over 4 images, every source is decoded on every question. Check for the cached PNG first. Page dimensions can be read from the cached file or stored alongside it.
- `sha256_file(view.path)` is recomputed for every view of every question. Memoize it by path and modification time, or derive the view hash from the source hash plus the view parameters.
- `sha256_file` of the source is computed in both `run_evaluation` and `Pipeline.predict`.

### 3.7 No HTTP connection reuse

`Pipeline.predict` creates and closes a new `OpenAICompatibleProvider`, and therefore a new `httpx.Client`, for every question. Create one provider per model alias for the duration of a run. `httpx.Client` is thread-safe, so the existing thread pool can share it.

### 3.8 Unexpected exceptions in `run_evaluation` make the run cost unknown

When a worker raises an unexpected exception, `cli.py` creates a fallback `Prediction` with the default `Usage()`, whose `estimated_cost_usd` is `None`. `_usage_metrics` then reports the whole run's cost as unknown. Use `Usage(estimated_cost_usd=0, usage_reported=False)`, or keep the usage already known, so a single crash does not hide all cost data. Also consider setting `category` from `resolve_intent(question)` instead of hard-coding `"general"`, so per-category metrics stay correct.

### 3.9 Smaller points

- `DatasetIndex` (`data.py`) indexes every file, including `.DS_Store` and other hidden files. Skip dotfiles in `scan` output and in basename matching.
- `doctor --probe` (`cli.py`) catches every `Exception`. Catch `httpx.HTTPError` and `ValueError`, so programming errors are not reported as "unreachable".
- `main` returns exit code 2 both for configuration errors and for "the run finished but some rows errored". Use a different code (for example 1) for partial failures, so scripts can tell them apart.

---

## 4. Code quality and maintainability

- **Split `Pipeline.predict`.** At about 115 lines, it mixes source resolution, the IFC route, view preparation, cached extraction, fusion and two answer policies. Suggested helpers: `_predict_ifc`, `_extract_views`, `_decide_count` and `_decide_general`. This also makes the abstention policy (2.5) easy to swap and test.
- **Consistent formatting.** `cli.py` uses a dense style (`p=sub.add_parser(name)`, `for a,b,c,d in boxes`) that differs from the other modules. Adopt `ruff format` and `ruff check` for the whole package.
- **Remove duplicated helpers.**
  - `ifc._sha256` duplicates `data.sha256_file`.
  - `cli.read_predictions` and `evaluation.read_predictions` both parse `predictions.jsonl` with different tolerance for a truncated final line. Keep one implementation with an explicit `allow_truncated_tail` flag.
  - `config.config_fingerprint` is used only in tests, while the pipeline uses `cache.digest`. They serialize differently (`mode="json"` and `allow_nan=False` against `default=str`). Pick one hashing helper.
  - `answer_ifc` builds its own fallback `question_id` with a different hash than `stable_id`. The pipeline overwrites it, but a direct caller gets inconsistent IDs.
- **Report template.** The HTML, CSS and JS in `report.py` are one minified Python string (about 20 KB). Move them into package data files (`report.html`, `report.css`, `report.js`), load them with `importlib.resources`, and add `[tool.setuptools.package-data]`. This makes the UI improvements in 2.8 much easier to review.
- **Type checking.** Many helpers are untyped (`iou`, `fuse_evidence.kind`, `View.to_page`, `_retry_delay`'s inputs). Add type hints there and run mypy or pyright in CI.
- **Named tuples for boxes.** `tuple[float, float, float, float]` is used everywhere. A small `BBox` type with `area`, `intersection` and `to_page` methods would remove the repeated index arithmetic in `pipeline.py` and `imaging.py`.

---

## 5. Tooling and repo hygiene

- **Version control.** The repository has no commits yet. Make an initial commit, and add a `.gitignore` covering at least `.venv/`, `runs/`, `.drawingqa/`, `.env`, `__pycache__/`, `*.egg-info/` and `build/`.
- **CI.** Add a GitHub Actions workflow that installs `requirements-dev.txt` and runs `python -m unittest discover -s tests` and `ruff check` on Python 3.11 and 3.12. Add an optional job with `ifcopenshell`.
- **License.** There is no `LICENSE` file. Choose one that is compatible with PyMuPDF's AGPL terms, which the README already warns about, or plan to replace the PDF renderer (for example with `pypdfium2`, which is Apache/BSD).
- **Development dependencies.** `requirements-dev.txt` contains only `-r requirements.txt`. Add a `dev` extra to `pyproject.toml` (`ruff`, `mypy`, `coverage`) and reference it from `requirements-dev.txt`.
- **Python versions.** `pyproject.toml` and `README.md` target Python 3.11, but `docs/VALIDATION.md` records validation on Python 3.12. Test both in CI, or state 3.12 as the tested version.
- **Packaging metadata.** `pyproject.toml` has no `license`, `authors`, `classifiers` or `urls`. Add them, together with the package data entry from section 4.

---

## 6. Docs and presentation

- **README structure.** The README is accurate but dense. It opens with the challenge brief and contains many caveats before the reader sees what the tool does. Suggested order:
  1. A one-paragraph pitch.
  2. A report screenshot or GIF.
  3. A results table (2.1).
  4. Quick start.
  5. How it works (link to `docs/ARCHITECTURE.md`).

  Move "Limits of this version" and "Evaluation interpretation" into a `docs/LIMITATIONS.md`, and keep a short summary in the README.
- **Example data paths.** The data layout section says the JSONL lives at the project root with drawings under `data/`. The bundled example uses `examples/questions_sample.jsonl` with bare basenames and `examples/data/`. Add the exact command for the bundled data:

  ```bash
  drawingqa evaluate --questions examples/questions_sample.jsonl --data-root examples/data \
    --models qwen --config configs/models.yaml --output runs/sample --limit 4
  ```

- **`ask` command ergonomics.** `drawingqa ask` requires `--output` and creates a full run directory, which fails on a second call unless `--resume` is given or the directory changes. For interactive use, print the answer, status and numbered evidence to stdout. Make `--output` optional, for example defaulting to a temporary or timestamped directory.
- **Model comparison fairness.** In `configs/models.yaml`, `qwen` uses `temperature: 0` and `nemotron` uses `temperature: 0.2` with `top_k: 1`. With `top_k: 1` this is effectively greedy, but state that explicitly in the README comparison section, because the README asks readers to keep generation settings consistent. Also consider evaluating Nemotron with thinking enabled on the hardest categories (floors and rooms), and report the extra tokens as a cost trade-off.

---

## Suggested order of work

1. Make an initial commit and add `.gitignore`, CI and `ruff` (section 5).
2. Get real runs on the sample (2.1), with structured decoding (2.4) and the per-model bbox format (2.3).
3. Fix the fusion and count-definition issues (3.1, 3.2), then tune the abstention policy using the real runs (2.5).
4. Broaden intent recognition (2.2) and normalize answer scoring (3.4).
5. Improve the report: zoom to evidence, review import and `apply-reviews` (2.8).
6. If time allows, add the detector and OCR (2.6, 2.7) and coarse-to-fine tiling (2.9).
7. Restructure the README around the results (section 6).
