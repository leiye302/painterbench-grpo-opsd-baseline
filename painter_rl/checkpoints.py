"""Atomic full training checkpoints. Never automatically delete user files."""
import json
import os
from pathlib import Path
import random
import shutil

import numpy as np
import torch
from scripts.reward_probe import atomic_json, sha256


def save_checkpoint(directory, model, processor, optimizer, update, contract):
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
    data = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if not data["complete"] or data["contract"] != contract:
        raise ValueError("Checkpoint contract mismatch; do not resume")
    files = {str(p.relative_to(directory)).replace("\\", "/")
             for p in directory.rglob("*") if p.is_file() and p.name != "manifest.json"}
    if files != set(data["files"]):
        raise ValueError("Checkpoint file set incomplete or modified")
    for name, checksum in data["files"].items():
        path = directory / name
        if path.is_symlink() or not path.resolve().is_relative_to(directory.resolve()):
            raise ValueError("Unsafe checkpoint file path")
        if sha256(directory / name) != checksum:
            raise ValueError(f"Checkpoint SHA mismatch: {name}")
    return data


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
