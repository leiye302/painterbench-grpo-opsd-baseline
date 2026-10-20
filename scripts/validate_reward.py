"""Recompute the official adapted scorer's test split, not its training fit.

Public recorded predictions are independently joined to ratings by drawing ID.
Optionally re-score a hash-selected image subset through the actual checkpoint.
No model or reward coefficient is fit on these test rows.
"""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.reward_probe import download, prepare, score_cases, atomic_json


def get_rows(work, base, path):
    target = work / "tables" / path.replace("/", "__")
    checksum = download(base + path, target)
    return list(csv.DictReader(io.StringIO(target.read_text(encoding="utf-8")))), checksum


def metrics(rows, score_key):
    import numpy as np
    from scipy.stats import pearsonr, spearmanr
    y = np.array([float(r["rated_creativity"]) for r in rows])
    p = np.array([float(r[score_key]) for r in rows])
    if not (np.isfinite(y).all() and np.isfinite(p).all()):
        raise ValueError("Nonfinite public rating or score")
    # Compare within the same initial stimulus, never arbitrary different tasks.
    pairs = [(i, j) for i in range(len(rows)) for j in range(i)
             if rows[i]["task"] == rows[j]["task"] and abs(y[i] - y[j]) >= .1]
    return {"n": len(rows), "pearson_r": float(pearsonr(y, p).statistic),
            "spearman_rho": float(spearmanr(y, p).statistic),
            "mse": float(np.mean((y - p)**2)),
            "same_stimulus_pairs_rating_gap_at_least_0_1": len(pairs),
            "pairwise_agreement": (sum((p[i] - p[j]) * (y[i] - y[j]) > 0
                                      for i, j in pairs) / len(pairs) if pairs else None)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--rescore", type=int, default=0)
    ap.add_argument("--assets-work", type=Path)
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    ap.add_argument("--download-only", action="store_true")
    args = ap.parse_args()
    lock = json.loads((ROOT / "sources.lock.json").read_text(encoding="utf-8"))
    p = lock["painterbench"]
    base = f"https://huggingface.co/datasets/{p['repository']}/resolve/{p['revision']}/"
    tables, hashes = {}, {}
    for name, path in {
        "split": "vidra/creativity_scorer_adapted/splits.csv",
        "ratings": "results/transfer/transfer_scores.csv",
        "index": "cloudresearch/index.csv",
        "scores": "results/vidra_adapted_scores.csv",
    }.items():
        tables[name], hashes[path] = get_rows(args.work, base, path)
    split = {r["drawing_id"]: r["split"] for r in tables["split"]}
    index = {r["drawing_id"]: r for r in tables["index"]}
    scores = {r["drawing_path"]: r["vidra_score"] for r in tables["scores"]}
    rows, all_rows, missing = [], [], []
    for r in tables["ratings"]:
        if split.get(r["drawing_id"]) != "test":
            continue
        path = index[r["drawing_id"]]["drawing_path"]
        all_rows.append({**r, "drawing_path": path,
                         "public_image": "cloudresearch/data/" + index[r["drawing_id"]]["filename"],
                         "adapted": scores.get(path)})
        if path not in scores:
            missing.append(r["drawing_id"])
            continue
        rows.append({**r, "drawing_path": path, "adapted": scores[path]})
    expected = sum(s == "test" for s in split.values())
    if len(all_rows) != expected:
        raise RuntimeError(f"Incomplete ratings/test join: {len(all_rows)}/{expected}")
    result = {"source_revision": p["revision"], "source_sha256": hashes,
              "scope": "Public recorded predictions on the available agent subset; not the full 480-row test.",
              "full_test_n": expected, "public_prediction_missing_n": len(missing),
              "adapted": metrics(rows, "adapted"), "unadapted": metrics(rows, "vidra"),
              "ink": metrics(rows, "ink"),
              "rl_robustness_certified": False}
    if args.rescore:
        selected = sorted(all_rows, key=lambda r: hashlib.sha256(
            ("reward-check-v1:" + r["drawing_id"]).encode()).hexdigest())[:args.rescore]
        cases = []
        def fetch_image(r):
            path = args.work / "images" / Path(r["public_image"]).name
            image_sha = download(base + r["public_image"], path)
            return {"path": str(path), "drawing_id": r["drawing_id"], "image_sha256": image_sha}
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=6) as pool:
            cases = list(pool.map(fetch_image, selected))
        if args.download_only:
            atomic_json(args.work / "reward_validation.json", result)
            print(json.dumps({"downloaded": len(cases), **result}, indent=2))
            return
        assets = prepare(args.assets_work or args.work / "scorer", lock)
        score_cases(cases, assets, 2, args.device, 6)
        deltas = [abs(c["score"] - float(r["adapted"])) for c, r in zip(cases, selected)
                  if r["adapted"] is not None]
        # Stored public predictions are rounded to six decimals.
        result["live_rescore"] = {"n": len(cases), "max_abs_difference": max(deltas) if deltas else None,
                                 "recorded_overlap_n": len(deltas),
                                 "all_within_1e_4": bool(deltas) and max(deltas) < 1e-4,
                                 "metrics": metrics([{**r, "live": c["score"]}
                                                     for r, c in zip(selected, cases)], "live"),
                                 "cases": cases}
    atomic_json(args.work / "reward_validation.json", result)
    print(json.dumps({k: v for k, v in result.items() if k != "source_sha256"}, indent=2))


if __name__ == "__main__":
    main()
