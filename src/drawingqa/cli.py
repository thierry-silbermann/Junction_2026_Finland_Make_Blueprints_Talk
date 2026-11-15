"""Batch-friendly CLI. No credentials or inference server are bundled."""
from __future__ import annotations

import argparse
import concurrent.futures
import importlib.metadata
import json
import os
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

from . import __version__
from .cache import digest
from .config import load_config
from .data import DatasetIndex, load_questions, sha256_file, stable_id
from .dxf import ADAPTER_VERSION as DXF_ADAPTER_VERSION
from .ifc_takeoff import ADAPTER_VERSION as TAKEOFF_ADAPTER_VERSION
from .evaluation import compare_runs, summarize
from .pipeline import Pipeline, add_usage, attach_ground_truth, prediction_fingerprint
from .report import create_report
from .schemas import AppConfig, ModelConfig, ModelResponse, PipelineConfig, Prediction, Question, Usage


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def read_predictions(path):
    results = []
    if path.exists():
        raw = path.read_text(encoding="utf-8")
        lines = raw.splitlines()
        for index, line in enumerate(lines):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                # A killed process can leave only the final append truncated.
                if (index == len(lines)-1 and not raw.endswith("\n")
                        and line.lstrip().startswith("{") and not line.rstrip().endswith("}")):
                    print("Ignoring truncated final prediction; it will be retried", file=sys.stderr)
                    continue
                else:
                    raise
            # A fully written record with an invalid schema is corruption, not a
            # truncated append: never silently discard it during resume.
            results.append(Prediction.model_validate(payload))
    return results


def run_evaluation(config, questions, data_root, output, models, *, resume=False, provider_factory=None):
    if not questions or not models:
        raise ValueError("At least one question and one model are required")
    ids = [q.id or stable_id(q) for q in questions]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate question IDs; each question must have a unique identity")
    if any(alias not in config.models for alias in models):
        raise ValueError(f"Unknown model alias. Available: {', '.join(config.models)}")
    if len(set(models)) != len(models):
        raise ValueError("Model aliases must be unique")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    pipeline = Pipeline(config, Path(data_root), output, provider_factory=provider_factory)
    source_hashes = {}
    for question in questions:
        if question.path not in source_hashes:
            try:
                source_hashes[question.path] = sha256_file(pipeline.index.resolve(question.path))
            except (ValueError, OSError) as exc:
                source_hashes[question.path] = "unresolved:" + str(exc)
    specification = {"config": config.model_dump(), "models": models,
                     "questions": [q.model_dump() for q in questions], "sources": source_hashes,
                     "version": __version__}
    run_hash = digest(specification)
    manifest_path = output / "manifest.json"
    predictions_path = output / "predictions.jsonl"
    existing = []
    carried_failures = {}
    if manifest_path.exists():
        if not resume:
            raise ValueError(f"Run already exists: {output}. Choose a new directory or use --resume")
        previous = json.loads(manifest_path.read_text())
        if previous.get("run_hash") != run_hash:
            raise ValueError("Cannot resume: sources, questions, configuration or application version changed. Use a new output directory; compatible requests can still use the cache.")
        carried_failures = previous.get("pending_failed_usage", {})
        existing = read_predictions(predictions_path)
        expected = {(alias, q.id or stable_id(q)): q for alias in models for q in questions}
        seen_records = set()
        for record in existing:
            key = (record.model, record.question_id)
            if key not in expected or key in seen_records:
                raise ValueError("Cannot resume: predictions contain duplicate or unknown task identities. Use a clean run directory.")
            seen_records.add(key)
            question = expected[key]
            if (record.path, record.question, record.page) != (question.path, question.question, question.page):
                raise ValueError("Cannot resume: stored prediction question identity does not match the manifest")
            if record.status != "error":
                if record.source_sha256 != source_hashes[question.path]:
                    raise ValueError("Cannot resume: stored prediction source hash is inconsistent")
                if record.config_sha256 != prediction_fingerprint(config, record.model):
                    raise ValueError("Cannot resume: stored prediction config hash is inconsistent")
                if (not question.path.lower().endswith(".ifc")
                        and record.model_id not in {DXF_ADAPTER_VERSION, TAKEOFF_ADAPTER_VERSION}
                        and record.model_id != config.models[record.model].model):
                    raise ValueError("Cannot resume: stored prediction model ID is inconsistent")
            attach_ground_truth(record, question)
            if record.status == "error":
                carried_failures[digest(key)] = record.usage.model_dump()
    elif predictions_path.exists():
        raise ValueError("Output contains predictions without a manifest; choose a new directory")
    manifest = {"schema_version": "1.0", "drawingqa_version": __version__,
                "run_hash": run_hash, "created_at": datetime.now(timezone.utc).isoformat(),
                "data_root": str(Path(data_root).resolve()), "models": models,
                "config": config.model_dump(), "source_hashes": source_hashes,
                "question_count": len(questions), "python": sys.version.split()[0],
                "pending_failed_usage": carried_failures,
                "note": "Model outputs are predictions, not human-verified ground truth. No credentials stored."}
    write_json(manifest_path, manifest)
    order = {(alias, q.id or stable_id(q)): index for index, (alias,q) in enumerate((a,q) for a in models for q in questions)}
    # Errors can be retried; successful and review-required records remain reproducible.
    records = {(p.model,p.question_id):p for p in existing if p.status not in {"error"}}
    # Rewrite valid recovered rows first, removing any truncated append or stale error.
    with predictions_path.open("w",encoding="utf-8") as stream:
        for result in records.values():
            stream.write(result.model_dump_json()+"\n")
    total = len(order)
    completed = len(records)
    started = time.monotonic()
    with predictions_path.open("a",encoding="utf-8") as stream:
        # Models execute in sequence; each model uses bounded request concurrency.
        # This avoids benchmarks competing with each other for the same GPU.
        for alias in models:
            pending = [q for q in questions if (alias,q.id or stable_id(q)) not in records]
            with concurrent.futures.ThreadPoolExecutor(max_workers=config.pipeline.max_concurrency) as executor:
                iterator = iter(pending)
                active = {}

                def submit_next():
                    question = next(iterator, None)
                    if question is not None:
                        active[executor.submit(pipeline.predict,question,alias)] = question

                for _ in range(config.pipeline.max_concurrency):
                    submit_next()
                while active:
                    done, _ = concurrent.futures.wait(active, return_when=concurrent.futures.FIRST_COMPLETED)
                    for future in done:
                        question = active.pop(future)
                        try:
                            result = future.result()
                        except Exception as exc:
                            # Preserve batch progress without swallowing process interrupts.
                            result = Prediction(question_id=question.id or stable_id(question),path=question.path,
                                                question=question.question,page=question.page,model=alias,
                                                model_id=config.models[alias].model,status="error",category="general",
                                                error=f"Unexpected {type(exc).__name__}: {exc}")
                        attach_ground_truth(result, question)
                        failed_usage = carried_failures.get(digest((alias, result.question_id)))
                        if failed_usage is not None:
                            cumulative = Usage(estimated_cost_usd=0)
                            add_usage(cumulative, Usage.model_validate(failed_usage))
                            add_usage(cumulative, result.usage)
                            result.usage = cumulative
                            result.warnings.append("Usage includes recorded failed attempts from an earlier run before resume")
                        records[(alias,result.question_id)] = result
                        stream.write(result.model_dump_json()+"\n")
                        stream.flush()
                        completed += 1
                        print(f"[{completed}/{total}] {alias} {result.status}: {question.path}", file=sys.stderr)
                        submit_next()
    results = sorted(records.values(), key=lambda p: order.get((p.model,p.question_id),len(order)))
    summary = summarize(results)
    manifest["pending_failed_usage"] = {digest((p.model,p.question_id)):p.usage.model_dump()
                                        for p in results if p.status == "error"}
    write_json(manifest_path, manifest)
    summary["run_wall_seconds"] = time.monotonic()-started
    hourly = sum(config.models[alias].gpu_cost_per_hour or 0 for alias in models)
    summary["estimated_gpu_cost_usd"] = summary["run_wall_seconds"]/3600*hourly if hourly else None
    summary["resumed"] = resume
    write_json(output / "summary.json", summary)
    report = create_report(output)
    print(json.dumps({"output":str(output),"report":str(report),"predictions":len(results),
                      "errors":sum(p.status=="error" for p in results)},indent=2))
    return results


def demo(output: Path):
    """A clearly named synthetic fixture: exercise plumbing without model credentials."""
    from PIL import Image, ImageDraw
    if (output / "manifest.json").exists():
        raise ValueError("Choose a fresh demo output directory")
    source = output / "demo_source"
    source.mkdir(parents=True,exist_ok=True)
    image = Image.new("RGB",(1000,700),"#f3f0e9")
    draw = ImageDraw.Draw(image)
    draw.rectangle((80,80,920,620),outline="#283847",width=20)
    boxes = [(180,68,300,94),(440,68,560,94),(700,68,820,94)]
    for box in boxes:
        draw.rectangle(box,fill="#bae1ee",outline="#217b9e",width=3)
    draw.text((160,300),"SYNTHETIC DEMO — NOT A MODEL PREDICTION",fill="#283847")
    image.save(source/"synthetic_plan.png")

    class FixtureProvider:
        def __init__(self,config): pass
        def generate(self,system,user,image_paths):
            payload={"answer":3,"candidates":[{"label":"synthetic window","bbox":[a/1000,b/700,c/1000,d/700],"counted":True,"uncertain":False,"note":"Programmatically supplied test fixture"} for a,b,c,d in boxes],"complete":True,"rationale":"Synthetic fixture demonstrates numbered evidence and report export; it does not measure a model.","warnings":["DEMO ONLY: no Nemotron/Qwen model was called"]}
            return ModelResponse(content=json.dumps(payload),usage=Usage(requests=0,estimated_cost_usd=0),finish_reason="stop")
        def close(self): pass

    config=AppConfig(models={"synthetic_demo":ModelConfig(model="SYNTHETIC-FIXTURE-NOT-A-MODEL")},
                     pipeline=PipelineConfig(cache_dir=str(output/"demo_cache")))
    question=Question(path="synthetic_plan.png",question="How many windows are shown on this floor plan?",type="count",unit="count")
    run_evaluation(config,[question],source,output,["synthetic_demo"],provider_factory=FixtureProvider)


def doctor(config_path: Path, probe=False):
    config=load_config(config_path)
    details={"python":sys.version.split()[0],"models":{}}
    for alias, model in config.models.items():
        item={"model":model.model,"base_url":model.base_url,
              "credential_configured":bool(os.getenv(model.api_key_env,"")) if model.api_key_env else "not_required",
              "note":"Configuration check only; model weights are not bundled"}
        if probe:
            import httpx
            headers={}
            if model.api_key_env:
                key=os.getenv(model.api_key_env,"")
                if not key:
                    item["probe"]="missing API key"
                    details["models"][alias]=item
                    continue
                headers["Authorization"]="Bearer "+key
            try:
                response=httpx.get(model.base_url+"/models",headers=headers,timeout=10,follow_redirects=False)
                item["probe"]=f"HTTP {response.status_code}"
                if response.status_code==200:
                    ids={str(m.get("id")) for m in response.json().get("data",[])}
                    item["configured_model_listed"]=model.model in ids
            except Exception:
                item["probe"]="unreachable or invalid model-list response"
        details["models"][alias]=item
    try:
        details["ifcopenshell"]=importlib.metadata.version("ifcopenshell")
    except importlib.metadata.PackageNotFoundError:
        details["ifcopenshell"]="not installed (optional)"
    print(json.dumps(details,indent=2))


def build_parser():
    parser=argparse.ArgumentParser(description="DrawingQA: evidence-first construction drawing QA")
    parser.add_argument("--version",action="version",version=__version__)
    sub=parser.add_subparsers(dest="command",required=True)
    scan=sub.add_parser("scan",help="Inventory files and detect basename ambiguity")
    scan.add_argument("--data-root",type=Path,required=True)
    for name in ("ask","evaluate"):
        p=sub.add_parser(name)
        p.add_argument("--config",type=Path,default=Path("configs/models.yaml"))
        p.add_argument("--data-root",type=Path,required=True)
        p.add_argument("--output",type=Path,required=True)
        p.add_argument("--resume",action="store_true")
        p.add_argument("--profile",help="Named run profile from the config (e.g. normal, reduced); flags below override it")
        p.add_argument("--strategy",choices=["per_question","per_drawing"],help="Override pipeline.strategy")
        p.add_argument("--mode",choices=["direct","inventory"],help="Override pipeline.mode")
        p.add_argument("--views",choices=["overview","tiles"],help="Override pipeline.views")
        p.add_argument("--concurrency",type=int,help="Override bounded per-model worker count")
        p.add_argument("--no-native-routes",action="store_true",
                       help="Send drawings with a DXF/IFC sibling to the vision model instead (.ifc inputs stay native)")
        if name=="ask":
            p.add_argument("--path",required=True)
            p.add_argument("--question",required=True)
            p.add_argument("--model",required=True)
            p.add_argument("--page",type=int)
        else:
            p.add_argument("--questions",type=Path,required=True)
            p.add_argument("--models",nargs="+",help="Model aliases (default: the profile's models, else qwen nemotron)")
            p.add_argument("--limit",type=int,help="First N question rows (not unique images)")
    compare=sub.add_parser("compare",help="Compare saved runs on paired question IDs")
    compare.add_argument("runs",type=Path,nargs="+")
    compare.add_argument("--output",type=Path,required=True)
    report=sub.add_parser("report",help="Rebuild interactive HTML report from saved predictions")
    report.add_argument("run",type=Path)
    check=sub.add_parser("doctor",help="Check config/dependencies without inference")
    check.add_argument("--config",type=Path,default=Path("configs/models.yaml"))
    check.add_argument("--probe",action="store_true",help="GET model endpoint metadata; no image inference")
    d=sub.add_parser("demo",help="Synthetic fixture demo, no real model or accuracy results")
    d.add_argument("--output",type=Path,default=Path("runs/demo"))
    return parser


def main(argv=None):
    args=build_parser().parse_args(argv)
    try:
        if args.command=="scan":
            print(json.dumps(DatasetIndex(args.data_root).summary(),indent=2))
        elif args.command=="doctor":
            doctor(args.config,args.probe)
        elif args.command=="demo":
            demo(args.output.resolve())
        elif args.command=="report":
            print(create_report(args.run.resolve()))
        elif args.command=="compare":
            print(json.dumps(compare_runs(args.runs,args.output),indent=2))
        else:
            config=load_config(args.config)
            profile_models=config.apply_profile(args.profile) if args.profile else None
            overrides={key:getattr(args,key) for key in ("mode","views","strategy") if getattr(args,key) is not None}
            if args.concurrency is not None:
                overrides["max_concurrency"]=args.concurrency
            if args.no_native_routes:
                overrides.update(use_dxf_sibling=False,use_ifc_sibling=False)
            if overrides:
                config.pipeline=PipelineConfig.model_validate({**config.pipeline.model_dump(),**overrides})
            if args.command=="ask":
                questions=[Question(path=args.path,question=args.question,page=args.page)]
                models=[args.model]
            else:
                questions=load_questions(args.questions)
                if args.limit is not None:
                    if args.limit<1:
                        raise ValueError("--limit must be positive")
                    questions=questions[:args.limit]
                models=args.models or profile_models or ["qwen","nemotron"]
            results=run_evaluation(config,questions,args.data_root,args.output,models,resume=args.resume)
            if any(p.status=="error" for p in results):
                return 2
        return 0
    except KeyboardInterrupt:
        print("Interrupted; completed rows are saved. Resume with the same command plus --resume.",file=sys.stderr)
        return 130
    except (ValueError,OSError,KeyError) as exc:
        print(f"DrawingQA: {exc}",file=sys.stderr)
        return 2


if __name__=="__main__":
    raise SystemExit(main())
