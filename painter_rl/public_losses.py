"""Load exact functions from pinned public sources; no rewritten GRPO/OPSD.

The model-specific rollout/training adapter must call this with one outcome per
trajectory, then expand the resulting advantages over that trajectory's
generated tokens. Never normalize a collection of individual turns as if those
turns were independent group samples.
"""
from __future__ import annotations

import ast
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
V = "https://raw.githubusercontent.com/JamesKrW/verl/3f55021e08b98315213e10d10b709c7f6bbe84f7/"
S = "https://raw.githubusercontent.com/ZJU-REAL/SDAR/d511f043afe9199b8576fc99ea52e12d84841f32/"
SOURCES = {
    "verl_torch.py": (V + "verl/utils/torch_functional.py",
                       ["masked_mean", "masked_sum"],
                       "cf9ecbfa6afed9981e3c14cea740509cc398da18250f7f2f990a2f56107beaae"),
    "verl_grpo.py": (V + "verl/trainer/ppo/core_algos.py",
                    ["compute_grpo_outcome_advantage", "compute_policy_loss", "kl_penalty"],
                    "3a3b1a1f351eb3ad762c8445806bedffac060486b28dde0df7f38e918dc77be5"),
    "sdar_core.py": (S + "verl/trainer/ppo/core_algos.py", ["agg_loss"],
                     "5043f97b87b00ab5c5d907022ea6cf148b935fdb71c37222a3f9c33ccfaf1dc6"),
    "sdar_loss.py": (S + "verl/trainer/ppo/sdar_utils.py", ["compute_sdar_loss"],
                     "329ee3a73e05d3ab9817dd59d69394dd6ba3df85ca9abe9ddebc0e2c2682ad68"),
}


def prepare_sources(cache):
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    receipt = {}
    for name, (url, selected, expected_sha) in SOURCES.items():
        target = cache / name
        if not target.exists():
            with urllib.request.urlopen(url, timeout=60) as response:
                raw = response.read()
            target.write_bytes(raw)
        raw = target.read_bytes()
        sha = hashlib.sha256(raw).hexdigest()
        if expected_sha and sha != expected_sha:
            raise RuntimeError(f"Pinned source SHA mismatch: {name}")
        functions = {n.name: n for n in ast.parse(raw).body if isinstance(n, ast.FunctionDef)}
        for function in selected:
            if function not in functions:
                raise ValueError(f"Missing upstream function: {function}")
        receipt[name] = {"url": url, "sha256": sha, "functions": {
            key: hashlib.sha256(ast.dump(functions[key], include_attributes=False).encode()).hexdigest()
            for key in selected}}
    (cache / "source_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    return receipt


class PublicLosses:
    def __init__(self, cache, download=False):
        import torch
        if download:
            prepare_sources(cache)
        namespace = {"torch": torch, "defaultdict": defaultdict}
        for name, (_, selected, expected_sha) in SOURCES.items():
            raw = (Path(cache) / name).read_bytes()
            if expected_sha and hashlib.sha256(raw).hexdigest() != expected_sha:
                raise RuntimeError(f"Source changed: {name}")
            nodes = [n for n in ast.parse(raw).body
                     if isinstance(n, ast.FunctionDef) and n.name in selected]
            # Compile the original function AST, including the original body.
            # Imports outside these standalone pure loss functions aren't run.
            tree = ast.Module(body=nodes, type_ignores=[])
            exec(compile(tree, f"<pinned-public:{name}>", "exec"), namespace)
            if name == "verl_torch.py":
                namespace["verl_F"] = SimpleNamespace(masked_mean=namespace["masked_mean"],
                                                       masked_sum=namespace["masked_sum"])
        self.functions = namespace

    def trajectory_advantages(self, rewards, group_ids, group_size):
        import torch
        if group_size < 2:
            raise ValueError("GRPO needs at least two trajectories per group")
        if len(rewards) != len(group_ids) or not group_ids:
            raise ValueError("One reward and group ID required per trajectory")
        if any(type(g) not in (str, int) for g in group_ids):
            raise TypeError("Group IDs must be Python str/int, not scalar tensors")
        if any(n != group_size for n in Counter(group_ids).values()):
            raise ValueError("Incomplete groups cannot be silently used")
        # The pinned upstream helper constructs CPU tensors internally.
        # Run group statistics on CPU deliberately, then move advantages when
        # assembling the token batch. Do not change the function itself.
        r = torch.as_tensor(rewards, dtype=torch.float32, device="cpu").view(-1, 1)
        if not torch.isfinite(r).all():
            raise ValueError("Nonfinite reward")
        adv, _ = self.functions["compute_grpo_outcome_advantage"](
            r, torch.ones_like(r), group_ids)
        return adv[:, 0]

    def combined(self, student, old, teacher, reference, advantages, mask,
                 entropy=None, cliprange=.2, sdar_coef=.01, ref_kl_coef=.01,
                 entropy_coef=.001, gate_beta=5.):
        import torch
        if not (student.shape == old.shape == teacher.shape == reference.shape ==
                advantages.shape == mask.shape):
            raise ValueError("Scores, mask and token advantages must align exactly")
        if mask.sum() <= 0 or not torch.all((mask == 0) | (mask == 1)):
            raise ValueError("Nonempty binary generated-token mask required")
        if not all(torch.isfinite(v).all() for v in (student, old, teacher, reference, advantages)):
            raise FloatingPointError("Nonfinite token scores or advantages")
        f = self.functions
        pg, clip, ppo_kl = f["compute_policy_loss"](
            old.detach(), student, advantages.detach(), mask, cliprange)
        sdar, stats = f["compute_sdar_loss"](
            student, teacher.detach(), mask, gate_beta, "token-mean")
        ref_kl = f["masked_mean"](
            f["kl_penalty"](student, reference.detach(), "low_var_kl"), mask)
        ent = student.new_zeros(()) if entropy is None else f["masked_mean"](entropy, mask)
        loss = pg + sdar_coef * sdar + ref_kl_coef * ref_kl - entropy_coef * ent
        return loss, {**stats, "grpo/loss": float(pg.detach()),
                      "grpo/clip_fraction": float(clip.detach()),
                      "grpo/approx_kl": float(ppo_kl.detach()),
                      "reference_kl": float(ref_kl.detach()), "entropy": float(ent.detach())}
