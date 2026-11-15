# PlanProof: Junction 2026 submission kit

Everything needed for the final submission of the Stoa challenge *Make Blueprints Talk*. The project is called **PlanProof** (the Python package is `drawingqa`).

| Required item | Where | Status |
|---|---|---|
| Project name, punchline, description | [SUBMISSION.md](SUBMISSION.md) | Ready to paste |
| Challenge | [SUBMISSION.md](SUBMISSION.md#challenge) | Ready |
| Technologies used | [SUBMISSION.md](SUBMISSION.md#technologies-used) | Ready |
| 2-minute video pitch | [VIDEO_SCRIPT.md](VIDEO_SCRIPT.md); draft video `PlanProof_pitch_draft.mp4` (1:43) | Record your voice (or use the draft) and upload; **paste the link** |
| Demo link | `demo/index.html` (the interactive review report, 805 real answers) | Push the repo, then use the link below |
| Source code link | <https://github.com/thierry-silbermann/Junction_2026_Finland_Make_Blueprints_Talk> | **Commit and push**; make the repo public |
| Pitch deck (PDF) | `PlanProof_pitch_deck.pdf` (12 slides, 16:9) | Ready to upload |

## Before you submit

1. **Push the code.** The repo has no commits yet. `.gitignore` already excludes `data/`, `.env`, the virtualenvs, the caches and the detector dataset. Run `git status` before the first commit and check that `.env` is not listed.
2. **Make the demo public.** Once pushed, the report opens at
   `https://raw.githack.com/thierry-silbermann/Junction_2026_Finland_Make_Blueprints_Talk/main/project_submission/demo/index.html`.
   Alternatively, enable GitHub Pages (*Settings → Pages → Deploy from branch → main, / (root)*); the URL becomes `https://thierry-silbermann.github.io/Junction_2026_Finland_Make_Blueprints_Talk/project_submission/demo/`.
3. **Video.** Record following [VIDEO_SCRIPT.md](VIDEO_SCRIPT.md), or upload the draft. Keep it at or under 1:55, make it public on YouTube or Vimeo, and paste the link into `SUBMISSION.md` and the form.
4. **Optional refresh.** Detector training is still running (mAP@50 0.89 at epoch 4). After it finishes, rerun the figure and build scripts below and update the mAP number on slide 7 and in `SUBMISSION.md` if it improved.

## About the demo report

`demo/` is a copy of `runs/per-folder-6-v2`: 805 questions from five dataset folders, answered by the DXF, IFC and Nemotron routes. Use the *Dataset* filter, click a question, and the evidence boxes appear on the page. The report shows `$0.0000` and `0.00 s` API time because this copy was rebuilt from the cache. The token figures in the deck come from the original uncached runs (~2.5k input and 0.7k output tokens per vision request). The drawings keep their licences: CubiCasa5K is CC BY-NC-SA 4.0, Swiss Dwellings is CC BY 4.0, and IFC models are licensed per file.

## Rebuilding

```bash
.venv/bin/python project_submission/build/make_figures.py          # evidence figures from the run
YOLO_CONFIG_DIR=$PWD/.ultralytics .venv-detector/bin/python project_submission/build/detector_figure.py \
  data/swiss_dwellings/site1021_bldg2101_floor9477.png data/cubicasa5k/high_quality-4017.png 700
.venv/bin/python project_submission/build/build.py                 # deck PDF + draft video (Chrome, ffmpeg, say)
```

Edit the slides in `build/deck.html`; the narration for the draft video is in `build/build.py` (`SEGMENTS`) and mirrors `VIDEO_SCRIPT.md`. The build fails if the video exceeds 1:55.
