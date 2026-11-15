# Installation and operation

DrawingQA has two separate environments:

1. **The client** reads drawings, calls models, evaluates answers and creates reports. This runs in your normal Python virtual environment on macOS, Linux or Windows.
2. **The model server** hosts the model weights. Use a compatible hosted endpoint, a separate Linux/NVIDIA GPU environment, or a rented Runpod GPU set up by `scripts/gpu/runpod_up.sh` (section 3, Option C). Installing this project's `requirements.txt` does not download model weights or install CUDA.

The commands below assume the current directory is the extracted `drawingqa` project. Use Python 3.11 or newer; the current version was validated with Python 3.13.5 (earlier with 3.12). A working model endpoint is required for VLM inference. The demo, the tests and the native DXF/IFC routes do not need one.

## 1. Install the client

macOS/Linux:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -r requirements-ifc.txt -r requirements-dxf.txt
drawingqa --help
```

The second install adds `ifcopenshell` and `ezdxf` for the native IFC and DXF routes (section 7). Without them, `.ifc` questions come back as unsupported and Swiss PDFs go to the VLM.

If your existing virtual environment already uses Python 3.11 or newer, activate it and run only the installation command. `python3.12` can replace `python3.11` throughout.

Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
drawingqa --help
```

The editable install (`-e .`) makes source changes available immediately. `requirements.txt` pins the direct client dependencies used for validation; `pyproject.toml` defines compatible ranges for packaging. For an environment-specific complete lock after installation:

```bash
python -m pip freeze > requirements.resolved.txt
```

## 2. Verify the client without downloading models

```bash
python -m unittest discover -s tests -v
drawingqa demo --output runs/demo
python -m http.server 8080 --bind 127.0.0.1 --directory runs/demo
```

Visit [http://127.0.0.1:8080/report.html](http://127.0.0.1:8080/report.html). The demo uses synthetic data and fixture responses, clearly identified in its output. It demonstrates rendering and data flow only; do not present its scores as model performance.

## 3. Choose where the models run

### Option A: Local or remote vLLM server

Use this option to compare the exact two main presets. On Apple Silicon, run only the client locally and point it at an NVIDIA GPU machine, or use another image-capable compatible server that explicitly supports the chosen model. These vLLM server instructions target Linux/NVIDIA, not macOS/MPS.

NVIDIA's Nano Omni card documents vLLM **0.20.0**. Install the server in a separate environment so GPU libraries do not change the client environment:

```bash
python3.12 -m venv .venv-vllm
source .venv-vllm/bin/activate
python -m pip install --upgrade pip
python -m pip install 'vllm==0.20.0'
```

The GPU driver/CUDA combination must be supported by the installed vLLM build. NVIDIA also documents the containers `vllm/vllm-openai:v0.20.0` and `vllm/vllm-openai:v0.20.0-cu129`; use the upstream guide for the correct image on your hardware. Record the model revision and container digest for a reproducible benchmark.

Start Qwen in its own terminal:

```bash
vllm serve Qwen/Qwen3-VL-8B-Instruct \
  --host 127.0.0.1 \
  --port 8001 \
  --max-model-len 16384 \
  --max-num-seqs 2 \
  --gpu-memory-utilization 0.90
```

Start Nemotron in another terminal/on another GPU:

```bash
vllm serve nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16 \
  --host 127.0.0.1 \
  --port 8002 \
  --max-model-len 16384 \
  --max-num-seqs 2 \
  --gpu-memory-utilization 0.90 \
  --trust-remote-code \
  --reasoning-parser nemotron_v3
```

The first start downloads weights and can take substantial time. `--trust-remote-code` allows NVIDIA's custom model code to execute; use the official repository and pin a reviewed revision for deployment. Images are sent as data URIs, so the server does not need unrestricted filesystem/media access. Audio/video-specific dependencies and flags are unnecessary for this drawing-only client.

The default Nemotron settings turn off its optional thinking mode to control token cost. `temperature: 0.2`, `top_k: 1`, and `chat_template_kwargs.enable_thinking: false` follow the model card's image/instruct example. Qwen uses temperature 0. These are recorded settings, not a claim that sampling is bit-for-bit deterministic across hardware.

Nano Omni has approximately 31B total parameters even though only about 3B are active per token. Its BF16 card lists an 80 GB H100 as the minimum reference GPU. Do not assume the 3B-active label means a small-memory model. Qwen is smaller; required free memory still depends on image tokens, context and batching. If both models cannot fit at the same time, evaluate them sequentially and then use `drawingqa compare`.

The Nano Omni FP8/NVFP4 variants can reduce memory on suitable hardware. To use one, change **both** the served model name and the `nemotron.model` entry in `configs/models.yaml` to the exact same variant. Follow NVIDIA's hardware-specific recipe rather than treating these precision formats as universally compatible. Record quantization in your experiment notes.

Verify the servers are ready:

```bash
curl http://127.0.0.1:8001/v1/models
curl http://127.0.0.1:8002/v1/models
```

When using a remote GPU machine, forward its ports over SSH or configure an authenticated HTTPS endpoint. Update `base_url` and `api_key_env` in YAML as needed. The base URL includes `/v1`; DrawingQA appends `/chat/completions`. Do not put API credentials into the URL.

### Option B: NVIDIA-hosted Nemotron

The provided `nemotron_nvidia` alias uses NVIDIA's hosted **Nano Omni** (`nvidia/nemotron-3-nano-omni-30b-a3b-reasoning`), with the same sampling settings as the local `nemotron` alias. NVIDIA controls its serving precision and revision, so record it as a separate run. It is useful for a quick hosted test without a GPU. The earlier `nvidia/nemotron-nano-12b-v2-vl` model has been retired and returns HTTP 410.

```bash
cp .env.example .env
```

Edit `.env` and set `NVIDIA_API_KEY` to your own NVIDIA API key. The value is read from the environment and is not embedded in the configuration or reports. Do not include `.env` when sharing the project.

```bash
drawingqa evaluate --questions data/questions_sample.jsonl --data-root data --models nemotron_nvidia --config configs/models.yaml --output runs/nvidia-smoke --limit 4
```

The preset disables thinking (`chat_template_kwargs.enable_thinking: false`); with thinking on, the reasoning consumes the token budget and no JSON is returned. Concurrency 4 triggered HTTP 429 rate limits in testing, so keep `--concurrency 2` and re-run with `--resume` to retry failed rows. Your account's availability, limits and pricing apply. This sends selected drawing images to the configured external endpoint. There is no hidden provider fallback. A Qwen endpoint is still needed to compare Qwen with this hosted model. You may use any compatible image-capable Qwen server by changing its YAML endpoint/model/API-key variable.

### Option C: Rented GPU on Runpod (the `gpu` profile)

One H100 80 GB serves the FP8 checkpoint (about 33 GB of weights) with room for many parallel requests. Expect about $2–3 an hour; the run itself is billed by wall time, so stop the pod as soon as the run ends.

1. **Rent the pod** in the Runpod console:
   - GPU: 1× H100 80GB (SXM or PCIe).
   - Template: the official **Runpod PyTorch** template. It ships with SSH configured for the public key saved in your Runpod account settings.
   - Public IP: the pod must have one (the "SSH over exposed TCP" connection). The proxied `ssh.runpod.io` connection cannot run the setup commands.
   - Storage: at least 80 GB of volume or container disk (weights, vLLM and the CUDA wheels).
   - If the console offers a CUDA version filter, choose 12.9 or newer to match vLLM 0.20.0's wheels.
2. **Copy the IP and port** from the pod's Connect tab, "SSH over exposed TCP" (`ssh root@<pod-ip> -p <ssh-port> -i ~/.ssh/id_ed25519`).
3. **Start the model and the tunnel** from the project root on your laptop. If the Hugging Face download asks for authentication, prefix the command with `HF_TOKEN=...`; the token is sent over stdin and stored with mode 600 on the pod only.

   ```bash
   scripts/gpu/runpod_up.sh <pod-ip> <ssh-port>             # optional third argument: path to your SSH key
   ```

   The script uploads `scripts/gpu/serve_nemotron.sh` to the pod and runs it. On the first start that script installs vLLM 0.20.0 into `/workspace/vllm-env`, downloads the weights to `/workspace/hf` and starts `vllm serve` with the model card's flags. It listens on `127.0.0.1:8000` inside the pod only, so nothing is exposed to the internet and no API key is needed. The laptop script prints the vLLM log every 20 s until the model answers, then holds an SSH tunnel from `localhost:8002`. Re-running it reuses the installed environment, the weights and a running server. Ctrl-C closes only the tunnel. `MAX_NUM_SEQS` (default 64) sets the server's parallel batch size.
4. **Check and run** from a second terminal:

   ```bash
   drawingqa doctor --probe                                  # nemotron_gpu should list the served model
   python scripts/sample_questions.py --per-folder 6 --folders cubicasa5k --output runs/samples/gpu-bench.jsonl
   drawingqa evaluate --profile gpu --questions runs/samples/gpu-bench.jsonl --data-root data --output runs/gpu-bench
   ```

   Compare `run_wall_seconds` and `estimated_gpu_cost_usd` in `runs/gpu-bench/summary.json` with the hosted run before starting a full batch. If you see timeouts, lower `--concurrency` from 48. If vLLM reports spare capacity, raise both `MAX_NUM_SEQS` and `--concurrency`.
5. **Stop the pod** in the Runpod console when the run is finished. A stopped pod can still bill for its volume storage; terminate it to stop all charges (the next start then downloads the weights again).

Set `nemotron_gpu.gpu_cost_per_hour` in `configs/models.yaml` to the price you actually pay (the default 2.69 is a placeholder within the range quoted above). The `nemotron` and `nemotron_gpu` aliases both use local port 8002; set `LOCAL_PORT` for the script and change `base_url` if you need both.

## 4. Place and validate data

The question files are `data/questions.jsonl` (127,044 rows) and `data/questions_sample.jsonl`. Source drawings stay inside `data/` with their collection subfolders. Paths in the JSONL may start with `data/`.

```bash
drawingqa scan --data-root data
```

Use nested relative paths in JSONL whenever possible:

```json
{"id":"plan-10052-windows-en","path":"cubicasa5k/colorful-10052.png","question":"How many windows are shown on this floor plan?","answer":7,"unit":"count","type":"count"}
```

To target one PDF/TIFF page, add `"page": 2` (1-based), or use `ask --page 2`. Without a page selector count questions inspect all represented pages, with configured limits enforced. Multi-page counts remain `needs_review` because the same physical element can appear on more than one sheet. General questions require a page for multi-page documents; cross-sheet synthesis is not implemented. Bare filenames such as `colorful-10052.png` work only when unique under the data root. Missing or ambiguous files produce explicit failures rather than selecting a random match.

Unsupported formats must be exported to PDF, images or supported IFC. Merely renaming an extension does not convert a file. Ground truth is optional; omit `answer` for real questions without a known label.

The included `examples/` data can be used to exercise the sample layout after an endpoint is ready:

```bash
drawingqa evaluate --questions examples/questions_sample.jsonl --data-root examples/data --models qwen --config configs/models.yaml --output runs/sample-qwen --limit 4
```

To measure each dataset folder on a small, reproducible sample (whole drawings, fixed seed; Swiss drawings only from odd site numbers, which the detector never trains on):

```bash
python scripts/sample_questions.py --per-folder 6 --output runs/samples/per-folder-6.jsonl
python scripts/sample_questions.py --per-folder 100 --folders cubicasa5k --output runs/samples/cubicasa-100.jsonl
```

Folders without questions (`wikimedia`, `habs`, `finna`, `europeana`) are skipped.

## 5. Ask a question or run a comparison

One question:

```bash
drawingqa ask --data-root data --path cubicasa5k/colorful-10052.png --question 'How many windows are shown on this floor plan?' --model qwen --config configs/models.yaml --output runs/one-question
```

`ask` and `evaluate` support `--mode direct|inventory`, `--views overview|tiles`, and `--concurrency N` to override those YAML settings for one run. Overrides become part of the saved run configuration. `--limit N` on `evaluate` selects question rows, not distinct images.

Run profiles pick the model and request strategy in one flag; later flags override them:

| Flag | Effect |
|---|---|
| `--profile normal` | `nemotron_nvidia`, one request per question (canonical counts share one per category) |
| `--profile reduced` | `nemotron_nvidia`, one shared inventory per page and group of element kinds; unrecognized questions and failed inventories fall back to per-question |
| `--profile gpu` | `nemotron_gpu` through the tunnel from Option C, shared inventories, concurrency 48 |
| `--strategy per_question\|per_drawing` | Choose the strategy without a profile |
| `--no-native-routes` | Send PDFs with a DXF or IFC source to the vision model too, for a fair "any image" measurement |

```bash
drawingqa evaluate --profile reduced --questions runs/samples/per-folder-6.jsonl --data-root data --output runs/reduced
drawingqa evaluate --profile normal --no-native-routes --questions runs/samples/native-folders-pdf.jsonl --data-root data --output runs/vlm-on-native-folders
```

The report's dataset table names the route per folder; shared inventories show as "VLM inventory".

Both models in one batch:

```bash
drawingqa evaluate --questions data/questions_sample.jsonl --data-root data --models qwen nemotron --config configs/models.yaml --output runs/comparison
```

Resume a compatible interrupted run. This also retries rows that ended in an error, such as rate-limit failures. Pass the same overrides (for example `--concurrency`) as the original run, otherwise the configuration no longer matches:

```bash
drawingqa evaluate --questions data/questions_sample.jsonl --data-root data --models qwen nemotron --config configs/models.yaml --output runs/comparison --resume
```

Use a new output directory when changing input questions, model configuration or preprocessing. The cache can still avoid repeat work if the full extraction identity matches. Preserve a run's manifest, predictions and assets together.

Two sequential runs:

```bash
drawingqa evaluate --questions data/questions_sample.jsonl --data-root data --models qwen --config configs/models.yaml --output runs/qwen
drawingqa evaluate --questions data/questions_sample.jsonl --data-root data --models nemotron --config configs/models.yaml --output runs/nemotron
drawingqa compare runs/qwen runs/nemotron --output runs/paired
```

Comparisons match question identities and warn about unequal sets. Do not compare different sample limits as if all cases were shared.

Regenerate a visual report after copying a run:

```bash
drawingqa report runs/comparison
```

The report shows a model comparison table and a dataset breakdown table (accuracy, coverage, errors and the route that answered: DXF, IFC take-off, IFC or VLM), and the question list can be filtered by dataset. It requires the `assets/` directory alongside it. It uses local JavaScript and does not fetch image assets from third-party sites. Review decisions can be exported separately as JSON; exporting review labels does not retrain a model.

## 6. Configure quality and cost

Edit a copied YAML file for each experiment; keep the original as a baseline. The full field definitions and validation live in `src/drawingqa/schemas.py`.

| Setting | Use |
|---|---|
| `pipeline.mode: inventory` | Enumerate visual instances and calculate recognized counts in Python |
| `pipeline.mode: direct` | Direct-answer baseline using the same evidence schema |
| `pipeline.views: overview` | Lowest-call default; one full-page view |
| `pipeline.views: tiles` | More detailed crops; higher calls, possible boundary duplicates |
| `max_image_side` | Resolution sent to the model; larger images can preserve fine symbols but cost more |
| `tile_size`, `tile_overlap` | Crop detail and shared context; inspect the effect on duplicate suppression |
| `max_views`, `max_pages` | Hard processing limits; exceeding scope must not silently produce a complete answer |
| `dedup_iou` | Overlap threshold used to merge repeated visual detections; tune on reviewed examples |
| `max_concurrency` | Parallel model work; reduce to 1 for a small GPU or strict rate limit |
| `max_tokens` | Output budget; increase for dense inventories or truncated responses |
| `timeout_seconds`, `max_retries` | Network/inference timeout and retry ceiling |
| `json_mode` | Optional endpoint JSON mode; leave false unless supported by your server |
| `bbox_scale` | `1000` for models that return boxes on a 0..1000 scale (Qwen3-VL, Nano Omni), `1` for 0..1 boxes |
| `pipeline.use_dxf_sibling` | Answer supported PDF questions from a same-named DXF (default true) |
| `pipeline.use_ifc_sibling` | Answer quoted storey/room/model questions on `<model>_plan-*.pdf` from `<model>.ifc` (default true) |
| `pipeline.strategy` | `per_question` (default) or `per_drawing` (shared inventories, see `inventory.py`) |
| `pipeline.inventory_max_tokens` | Output cap for shared inventories (4096); a looping list fails fast and falls back to per-question |
| `pipeline.inventory_extra_body` | Extra generation settings for inventories only; `repetition_penalty: 1.1` stops repeated-item loops |
| `profiles` | Named bundles of models and pipeline settings for `--profile` |
| `extra_body` | Backend-specific generation settings; cannot replace core messages/model fields |
| `input_cost_per_million`, `output_cost_per_million` | Your actual token prices; null means unknown, not free |
| `gpu_cost_per_hour` | Hourly price of a self-hosted GPU; runs then report `estimated_gpu_cost_usd` (wall time × rate) |
| `cache_dir` | Local cache location, relative to the working directory unless absolute |

Caching is keyed by source content, model/settings, prompt/intent and image-view settings. It must not be interpreted as a new model call. When collecting uncached speed measurements, point `cache_dir` at a fresh experiment-specific directory. Cache entries and reports contain source-derived content; manage them as part of your dataset.

For a fair baseline compare identical images and questions. Then evaluate the benefit/cost of tiles and inventory mode as separate experiments. Do not select thresholds using the final held-out set.

## 7. Native IFC and DXF routes

```bash
python -m pip install -r requirements-ifc.txt -r requirements-dxf.txt
```

No special model alias is needed: select any configured alias and the route is chosen from the files and the question. Predictions record the adapter's model ID (`ifc-takeoff-v1`, `ifc-entities-v1` or `dxf-layers-v1`). Consequently, running both Qwen and Nemotron aliases over these questions does not measure either VLM; it repeats the native result. If an optional dependency is missing, the result explains how to install it.

### IFC take-off (`ifc_takeoff.py`)

Questions that name a storey, room or model in quotes are answered from an `.ifc` file, and from `<model>.ifc` for floor plans named `<model>_plan-NN-<level>.pdf` (the `ifc_models` set). It is checked before the entity-count adapter below. Supported templates, in English, German, Dutch, Danish, Spanish, Estonian, Polish, Romanian and Italian:

- element counts on a storey ("How many doors are shown on the floor plan of 'Level 1'?") or in the whole model: doors, windows, walls, columns, stairs, beams, rooms, sanitary fixtures and light fixtures;
- per-storey sums of stored wall/slab quantities: net volume, net side area and length, converted from the model's units;
- the net floor area of a named room, and the total for a storey: the stored net area (`NetFloorArea` or an authoring-tool net-area property), otherwise the footprint of the IfcSpace geometry.

Elements belong to a storey through spatial containment or aggregation. Parts aggregated into an element of the same class (such as stair risers) are not counted separately. On all 3,062 `ifc_models` questions it scores 100%; the rules were derived on that set. Evidence lists IFC GlobalIds without boxes on the PDF. Disable the PDF route with `pipeline.use_ifc_sibling: false`.

### IFC entity counts (`ifc.py`)

Other `.ifc` questions about supported counts go to the entity-count adapter. Results retain IFC GUIDs as evidence; the review page lists them without inventing 2D image boxes. Room and fixture counts depend on the properties actually populated in the IFC file. An absent or ambiguous classification is not automatically zero. It does not render 3D geometry, measure accessibility, resolve every custom property-set vocabulary or answer arbitrary BIM questions, so the free-text `ifc_bench` questions are reported as unsupported.

### DXF sibling (`dxf.py`)

When a question points at `plan.pdf` and `plan.dxf` exists next to it, supported count and area questions are answered from the DXF's semantic layers instead of the VLM, with boxes mapped onto the rendered PDF page. It understands the Swiss layer convention (`DOOR`, `WINDOW`, `FIXTURE`, `ROOM_OUTLINE`, `ROOM_TEXT`, `UNIT_OUTLINE`) and English, German, French and Italian question wordings. Unsupported wordings still go to the selected model. On held-out Swiss sites it scores 99.0% (18,528 questions). Disable the route with `pipeline.use_dxf_sibling: false`.

## 8. Optional: train the symbol detector

The detector is experimental and not yet used by the pipeline. It runs in its own environment because Ultralytics is AGPL-3.0:

```bash
python3 -m venv .venv-detector
.venv-detector/bin/pip install ultralytics
PYTHONPATH=src .venv/bin/python scripts/detector/make_labels.py --output datasets/swiss_symbols
.venv-detector/bin/python scripts/detector/train.py --epochs 15 --imgsz 640 --batch 16
```

`make_labels.py` cuts the even-site Swiss PNGs into 1024-pixel tiles (about 3,200 tiles, 300 MB) with boxes for doors, windows, toilets, washbasins, bathtubs and showers taken from the DXFs; 10% of the even sites form the validation split. Training writes to `runs/detector/swiss_symbols/`. It defaults to CUDA or CPU: on Apple MPS the loss became NaN within the first epoch with or without mixed precision. On CPU an epoch takes about 7.5 minutes. The torch environment needs about 1.4 GB of disk.

## 9. Troubleshooting

| Symptom | Action |
|---|---|
| `drawingqa: command not found` | Activate the environment and run `python -m pip install -r requirements.txt` from the project root |
| Connection refused | Start the correct model server; check YAML host/port and server logs |
| HTTP 401/403 | Check the named API-key environment variable and provider access |
| Model not found | Match `model` to the server's `/v1/models` output or `--served-model-name` |
| Out of GPU memory | Run models sequentially; reduce server context/batching; select supported quantization or a larger GPU |
| Output truncated/invalid JSON | Increase `max_tokens`, inspect server compatibility, use a smaller view, or reduce thinking output; no count is silently invented |
| Unsupported response format | Keep `json_mode: false`; the prompt still requests JSON and the client validates it |
| Too many requests (HTTP 429) | Reduce `--concurrency` to 2; re-run with `--resume` and the same overrides to retry the failed rows |
| HTTP 410 from the hosted API | The model was retired; `nvidia/nemotron-nano-12b-v2-vl` is gone, use the configured Nano Omni ID |
| "Model output is not valid evidence JSON" | Check `bbox_scale` for the model; thinking must be disabled for the hosted Nano Omni |
| All `ifc_bench` rows unsupported | Expected: these are free-text questions the IFC adapters do not answer |
| Detector loss is `nan` | Train on CPU or CUDA instead of Apple MPS (`--device cpu`) |
| Ambiguous filename | Replace the bare filename in JSONL with its collection-relative path |
| Incorrect boxes/counts | Review source resolution, labels, grouping rules and tile boundaries; schema validation cannot correct visual errors |
| All results say unsupported | Confirm real file formats and install `requirements-ifc.txt` and `requirements-dxf.txt` |
| Blank report images | Keep `assets/` beside `report.html`, then use the local HTTP server command |
| `runpod_up.sh` asks for a password or is refused | Use the "SSH over exposed TCP" IP and port (public IP pod), and check the public key in Runpod settings and the key path |
| `vLLM exited` during GPU start-up | Read the printed log tail; common causes are a CUDA driver older than the vLLM wheels (choose a newer CUDA pod), a full disk, or a gated download (set `HF_TOKEN`) |
| Reduced mode makes more requests than expected | Inventories that loop on dense sheets fail once per drawing and group, then fall back to per-question; the run log and report warnings show which |

`drawingqa doctor --config configs/models.yaml` checks configuration/environment; add `--probe` to request `/v1/models` metadata from configured endpoints. Neither mode is an accuracy benchmark or sends drawings for inference. Run one real question to validate end-to-end model compatibility before a large batch.

## Building a distributable

For a wheel from your current source:

```bash
python -m pip wheel --no-deps . --wheel-dir dist
```

Keep configuration, documentation and your data outside the wheel as deployment inputs. To share the complete project, archive the source, configuration and docs, excluding `.env`, environments, generated caches and private drawings. Tests are executed with `python -m unittest discover -s tests -v`.

## Documentation sources

Serving settings were checked on 14 November 2026 and the hosted endpoint was exercised on 15 November 2026; GPU-driver support and provider offerings can change. The client itself has no hard dependency on a particular serving engine:

- [NVIDIA Nano Omni serving guide](https://huggingface.co/nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16) and [FP8 checkpoint](https://huggingface.co/nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-FP8)
- [Runpod: connect to a Pod with SSH](https://docs.runpod.io/pods/configuration/use-ssh)
- [Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct)
- [NVIDIA hosted models (build.nvidia.com)](https://build.nvidia.com)

