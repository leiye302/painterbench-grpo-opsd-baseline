"""Atomic full checkpoints and bounded retention of verified same-run saves."""
import json
import os
from pathlib import Path
import random
import re
import shutil

import numpy as np
import torch
from scripts.reward_probe import atomic_json, sha256


def save_checkpoint(directory, model, processor, optimizer, update, contract, *, restore_probe=None):
    directory = Path(directory)
    partial = directory.with_name(directory.name + ".partial")
    if directory.exists() or partial.exists():
        raise FileExistsError("Checkpoint exists; preserve it and choose a new destination")
    directory.parent.mkdir(parents=True, exist_ok=True)
    need = int(sum(p.numel()*p.element_size() for p in model.parameters()) * 3.1)
    if shutil.disk_usage(directory.parent).free < need:
        raise RuntimeError("Insufficient disk space for a complete resumable checkpoint")
    partial.mkdir(parents=True)
    model.save_pretrained(partial / "model", safe_serialization=True, max_shard_size="2GB")
    processor.save_pretrained(partial / "model")
    torch.save({"optimizer": optimizer.state_dict(), "update": update,
                "python_rng": random.getstate(), "numpy_rng": np.random.get_state(),
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []},
               partial / "training_state.pt")
    if restore_probe is not None:
        # One small scoring fixture per checkpoint, not every rollout input.
        torch.save(restore_probe, partial / "restore_probe.pt")
    files = {str(p.relative_to(partial)).replace("\\", "/"): sha256(p)
             for p in sorted(partial.rglob("*")) if p.is_file()}
    atomic_json(partial / "manifest.json", {"update": update, "contract": contract,
                                          "files": files, "complete": True})
    # Flush file contents before the final directory commit.
    for p in partial.rglob("*"):
        if p.is_file():
            with p.open("rb+") as handle:
                os.fsync(handle.fileno())
    if os.name == "posix":
        directories = sorted((p for p in partial.rglob("*") if p.is_dir()),
                             key=lambda p: len(p.parts), reverse=True) + [partial]
        for p in directories:
            fd = os.open(p, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
    os.replace(partial, directory)
    if os.name == "posix":
        fd = os.open(directory.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    return directory


def verify_checkpoint(directory, contract):
    directory = Path(directory)
    if directory.is_symlink() or getattr(directory, "is_junction", lambda: False)():
        raise ValueError("Checkpoint root must not be a link")
    data = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if not data["complete"] or data["contract"] != contract:
        raise ValueError("Checkpoint contract mismatch; do not resume")
    _verify_file_layout(directory, data)
    for name, checksum in data["files"].items():
        if sha256(directory / name) != checksum:
            raise ValueError(f"Checkpoint SHA mismatch: {name}")
    return data


def _verify_file_layout(directory, data):
    if directory.is_symlink() or getattr(directory, "is_junction", lambda: False)():
        raise ValueError("Checkpoint root must not be a link")
    for p in directory.rglob("*"):
        if p.is_symlink() or getattr(p, "is_junction", lambda: False)():
            raise ValueError("Checkpoint must not contain links")
    files = {str(p.relative_to(directory)).replace("\\", "/")
             for p in directory.rglob("*") if p.is_file() and p != directory / "manifest.json"}
    if files != set(data["files"]):
        raise ValueError("Checkpoint file set incomplete or modified")
    for name in data["files"]:
        path = directory / name
        if not path.resolve().is_relative_to(directory.resolve()):
            raise ValueError("Unsafe checkpoint file path")


def prune_checkpoints(directory, contract, keep_last=2):
    """Remove only older complete saves under this exact checkpoint root.

    Verify BOTH retained checkpoints before removing anything. Foreign runs,
    incomplete saves and unrelated files are never deletion candidates.
    """
    if type(keep_last) is not int or keep_last < 2:
        raise ValueError("Keep at least two complete checkpoints")
    directory = Path(directory)
    if directory.is_symlink() or getattr(directory, "is_junction", lambda: False)():
        raise ValueError("Checkpoint root must not be a link")
    root = directory.resolve(strict=True)
    candidates = []
    for path in root.iterdir():
        match = re.fullmatch(r"update_(\d{4,})", path.name)
        if (not match or not path.is_dir() or path.is_symlink()
                or getattr(path, "is_junction", lambda: False)()):
            continue
        manifest = path / "manifest.json"
        if not manifest.is_file() or manifest.is_symlink():
            continue
        try:
            data = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if data.get("contract") != contract or data.get("complete") is not True:
            continue
        if data.get("update") != int(match[1]) or path.resolve().parent != root:
            raise ValueError("Checkpoint name/update/location mismatch")
        candidates.append((data["update"], path, data))
    candidates.sort(key=lambda item: item[0])
    # A damaged newest save must never authorize deleting the older recovery copy.
    for _, path, _ in candidates[-keep_last:]:
        verify_checkpoint(path, contract)
    older = candidates[:-keep_last]
    for _, path, data in older:
        _verify_file_layout(path, data)
    removed = []
    for update, path, _ in older:
        if path.is_symlink() or path.resolve().parent != root:
            raise ValueError("Checkpoint location changed before retention")
        size = sum(p.stat().st_size for p in path.rglob("*") if p.is_file())
        shutil.rmtree(path)
        removed.append({"update": update, "directory": path.name, "bytes": size})
    if removed and os.name == "posix":
        fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    return {"keep_last": keep_last,
            "retained": [path.name for _, path, _ in candidates[-keep_last:]],
            "removed": removed}


def restore_optimizer(directory, optimizer):
    # Load only this user's own, SHA-verified checkpoint, never arbitrary pickle.
    state = torch.load(Path(directory) / "training_state.pt", map_location="cpu", weights_only=False)
    optimizer.load_state_dict(state["optimizer"])
    random.setstate(state["python_rng"])
    np.random.set_state(state["numpy_rng"])
    torch.set_rng_state(state["torch_rng"])
    if state["cuda_rng"]:
        torch.cuda.set_rng_state_all(state["cuda_rng"])
    return state["update"]
