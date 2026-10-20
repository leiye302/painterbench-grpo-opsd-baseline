"""Test the released adapted scorer before using it as an RL reward.

This is a measurement tool, not a training script. Neither labels nor raw
trajectories are changed. Derived canvases stay in --work. No API calls.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import urllib.request
import urllib.error

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".partial")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def download(url, path, expected=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        tmp = path.with_suffix(path.suffix + ".partial")
        for attempt in range(6):
            try:
                with urllib.request.urlopen(url, timeout=40) as response, tmp.open("wb") as out:
                    while chunk := response.read(8 * 1024**2):
                        out.write(chunk)
                if expected and sha256(tmp) != expected:
                    raise ValueError(f"Downloaded asset SHA mismatch: {path.name}")
                os.replace(tmp, path)
                break
            except (OSError, urllib.error.URLError) as exc:
                if (isinstance(exc, urllib.error.HTTPError) and exc.code in (401, 403, 404)) or attempt == 5:
                    raise
                time.sleep(min(2 ** attempt, 16))
    actual = sha256(path)
    if expected and actual != expected:
        raise RuntimeError(f"Public asset SHA mismatch: {path.name}")
    return actual


def prepare(work, lock):
    p = lock["painterbench"]
    base = f"https://huggingface.co/datasets/{p['repository']}/resolve/{p['revision']}/"
    assets = work / "assets"
    records = {}
    for source, name, checksum in [
        (lock["reward"]["checkpoint"], "vidra_adapted.joblib", lock["reward"]["sha256"]),
        (lock["reward"]["feature_module"], "dinov2_features.py",
         lock["reward"]["feature_module_sha256"]),
        (lock["reward"]["validation_report"], "report_adapted.json",
         lock["reward"]["validation_report_sha256"]),
    ]:
        records[name] = download(base + source, assets / name, checksum)
    dino = assets / "dinov2"
    if not dino.exists():
        dino.mkdir()
        subprocess.run(["git", "init", "--quiet", str(dino)], check=True)
        subprocess.run(["git", "-C", str(dino), "remote", "add", "origin",
                        lock["dinov2"]["repository"]], check=True)
        subprocess.run(["git", "-C", str(dino), "fetch", "--quiet", "--depth=1",
                        "origin", lock["dinov2"]["revision"]], check=True)
        subprocess.run(["git", "-C", str(dino), "checkout", "--quiet", "--detach",
                        "FETCH_HEAD"], check=True)
    revision = subprocess.check_output(
        ["git", "-C", str(dino), "rev-parse", "HEAD"], text=True).strip()
    if revision != lock["dinov2"]["revision"]:
        raise RuntimeError("DINOv2 source revision mismatch")
    records["dinov2_vitl14_pretrain.pth"] = download(
        lock["dinov2"]["weights_url"], assets / "dinov2_vitl14_pretrain.pth",
        lock["dinov2"].get("weights_sha256"))
    atomic_json(work / "asset_receipt.json", {
        "sha256": records, "dinov2_revision": revision,
        "reward_sha256": lock["reward"]["sha256"],
    })
    return assets


def collect_cases(trajectories, work, limit):
    """Choose by ID, never by score. Score unfinished last canvases too."""
    from PIL import Image, ImageDraw
    import numpy as np

    paths = sorted(trajectories.glob("*/complete.json"),
                   key=lambda p: hashlib.sha256(p.parent.name.encode()).hexdigest())
    if not paths:
        raise ValueError("Expected one directory per trajectory with complete.json")
    chosen = paths[:limit] if limit else paths
    cases = []
    controls = work / "controls"
    controls.mkdir(parents=True, exist_ok=True)
    for meta_path in chosen:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        directory = meta_path.parent
        final = directory / ("final_painting.png" if meta.get("finished") else
                             "unfinished_painting.png")
        if not final.exists():
            raise FileNotFoundError(f"No last canvas for {directory.name}")
        image = Image.open(final).convert("RGB")
        common = {"trajectory_id": directory.name, "stimulus": meta["stimulus"],
                  "finished": meta["finished"], "turns": meta["turns"]}
        variants = {"original": image}
        # 5px black strokes are legal native drawing operations. Controlled
        # degradation adds no semantic object, no labels and no title.
        for count in (10, 50):
            corrupted = image.copy()
            draw = ImageDraw.Draw(corrupted)
            rng = random.Random(2026)
            for _ in range(count):
                draw.line(tuple(rng.randrange(10, 390) for _ in range(4)),
                          fill="black", width=5)
            variants[f"add_random_{count}_lines"] = corrupted
        # A simple geometry-only pattern probes elaboration without a depicted
        # scene. Pixel coordinates, width and original stimulus are retained.
        grid = image.copy()
        draw = ImageDraw.Draw(grid)
        for x in range(20, 400, 20):
            draw.line((x, 0, x, 399), fill="black", width=5)
            draw.line((0, x, 399, x), fill="black", width=5)
        variants["add_grid"] = grid
        variants["duplicate_pixels"] = image.copy()
        for kind, img in variants.items():
            path = controls / f"{directory.name}__{kind}.png"
            img.save(path)
            cases.append({**common, "variant": kind, "path": str(path),
                          "image_sha256": sha256(path),
                          "ink_fraction": float((np.asarray(img).min(axis=2) < 128).mean())})
    # Extra density controls have no stimulus: diagnostic only, never mixed
    # into reward training/evaluation as if they were legal rollouts.
    for kind, color in [("blank", "white"), ("solid_black", "black")]:
        path = controls / f"diagnostic_{kind}.png"
        Image.new("RGB", (400, 400), color).save(path)
        cases.append({"trajectory_id": "diagnostic", "stimulus": None,
                      "finished": None, "turns": None, "variant": kind,
                      "path": str(path), "image_sha256": sha256(path),
                      "ink_fraction": float(color == "black")})
    return cases


def public_examples(work, lock):
    """A clean checkout can run the probe without private collection files."""
    p = lock["painterbench"]
    base = f"https://huggingface.co/datasets/{p['repository']}/resolve/{p['revision']}/"
    folder = work / "public_examples"
    for model in ("Qwen_Qwen3.5-9B", "gpt-6-astra"):
        for stimulus in ("is01", "ot01"):
            destination = folder / f"{model}_{stimulus}_seed1"
            destination.mkdir(parents=True, exist_ok=True)
            source = f"results/study/run/{model}/{stimulus}/seed_1/final_painting.png"
            image_sha = download(base + source, destination / "final_painting.png")
            atomic_json(destination / "complete.json", {
                "stimulus": stimulus, "finished": True, "turns": None,
                "source": base + source, "image_sha256": image_sha,
                "note": "Public final-canvas integration fixture, not held-out reward validation.",
            })
    return folder


def score_cases(cases, assets, batch_size, device_name, cpu_threads):
    import joblib
    import numpy as np
    import sklearn
    import torch

    if sklearn.__version__ != "1.8.0":
        raise RuntimeError("The released head requires scikit-learn==1.8.0")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("No CUDA device. Do not silently fall back to a small CPU container.")
    if device_name == "cpu":
        memory_limit = Path("/sys/fs/cgroup/memory.max")
        if memory_limit.exists():
            limit = memory_limit.read_text().strip()
            if limit.isdigit() and int(limit) < 8 * 1024**3:
                raise RuntimeError("CPU scoring requires at least 8 GiB container RAM")
    torch.set_num_threads(cpu_threads)
    spec = importlib.util.spec_from_file_location("painterbench_d2f", assets / "dinov2_features.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    checkpoint = joblib.load(assets / "vidra_adapted.joblib")
    if checkpoint["model_type"] != "dinov2-ridge":
        raise ValueError("Unexpected official reward architecture")
    config = module.config_from_stored(checkpoint.get("feature_config"))
    print(json.dumps({"feature_config": config.to_dict(), "head": checkpoint.get("head")}), flush=True)
    device = torch.device(device_name)
    # Same official architecture, exact released weights, deterministic FP32.
    # Do not call torch.hub's unpinned GitHub main or silently change precision.
    model = torch.hub.load(str(assets / "dinov2"), config.variant,
                          source="local", pretrained=False, verbose=False)
    state = torch.load(assets / "dinov2_vitl14_pretrain.pth",
                       map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval().requires_grad_(False).to(device)
    x = module.extract_features_for_paths(
        [Path(c["path"]) for c in cases], model, module.build_transform(config),
        device, batch_size=batch_size, config=config,
        progress=lambda done, total: print(f"features {done}/{total}", flush=True))
    if x.shape[1] != checkpoint["pipeline"].n_features_in_:
        raise ValueError("Reward feature width mismatch")
    scores = checkpoint["pipeline"].predict(x)
    if not np.isfinite(scores).all():
        raise FloatingPointError("Nonfinite reward")
    for case, score in zip(cases, scores):
        case["score"] = float(score)
    return config.to_dict()


def summarize(cases):
    import numpy as np
    from scipy.stats import spearmanr

    originals = {c["trajectory_id"]: c for c in cases if c["variant"] == "original"}
    comparisons = {}
    for kind in ("add_random_10_lines", "add_random_50_lines", "add_grid", "duplicate_pixels"):
        deltas = [c["score"] - originals[c["trajectory_id"]]["score"]
                  for c in cases if c["variant"] == kind]
        comparisons[kind] = {"count": len(deltas), "mean_delta": float(np.mean(deltas)),
                             "min_delta": float(min(deltas)), "max_delta": float(max(deltas)),
                             "score_increased_count": sum(d > 1e-5 for d in deltas),
                             "score_increased_over_0_05_count": sum(d > .05 for d in deltas)}
    values = list(originals.values())
    return {
        "original_count": len(values),
        "original_mean": float(np.mean([x["score"] for x in values])),
        "original_min": min(x["score"] for x in values),
        "original_max": max(x["score"] for x in values),
        "score_ink_spearman": float(spearmanr([x["score"] for x in values],
                                           [x["ink_fraction"] for x in values]).statistic),
        "paired_controls": comparisons,
        "out_of_range_count": sum(not 0 <= x["score"] <= 1 for x in cases),
        "rl_reward_approved": False,
        "interpretation": "Stress tests are diagnostic, not human creativity ground truth. "
                          "A correlation on a passive dataset does not certify an optimized RL reward.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, required=True)
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument("--trajectories", type=Path)
    inputs.add_argument("--public-examples", action="store_true")
    parser.add_argument("--limit", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    args.work.mkdir(parents=True, exist_ok=True)
    lock = json.loads((ROOT / "sources.lock.json").read_text(encoding="utf-8"))
    assets = prepare(args.work, lock)
    if args.prepare_only:
        return
    if args.public_examples:
        args.trajectories = public_examples(args.work, lock)
    if args.trajectories is None:
        parser.error("--trajectories or --public-examples is required unless --prepare-only")
    started = time.time()
    cases = collect_cases(args.trajectories, args.work, args.limit)
    atomic_json(args.work / "cases_before_scoring.json", cases)
    config = score_cases(cases, assets, args.batch_size, args.device, args.cpu_threads)
    report = {**summarize(cases), "feature_config": config,
              "runtime_seconds": time.time() - started, "cases": cases}
    atomic_json(args.work / "reward_probe.json", report)
    print(json.dumps({k: v for k, v in report.items() if k != "cases"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
