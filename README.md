# PainterBench GRPO + OPSD baseline preparation

**Status: reward validation and algorithm-contract preparation. Not yet a
training-ready release, and no learning result is claimed.**

This project prepares a Qwen3.5-4B baseline for PainterBench's native multi-turn
drawing tools. GRPO uses the pinned public VAGEN/VERL loss functions; gated OPSD
uses the pinned SDAR function with `gate_beta=5`. The baseline has no Route-SFT,
RETHINK, Analyzer or paid API dependency.

The current deliverable is deliberately explicit about its boundary:

| Component | Status |
| --- | --- |
| Exact public GRPO and OPSD pure-loss functions | Integrated; source SHA and AST receipts |
| Grouping, generated-token mask, gradient and microbatch tests | Implemented |
| Official ViDrA-adapted scoring and shortcut probe | Implemented |
| Public example images for standalone scoring | Downloadable from pinned public dataset |
| Qwen3.5 native multimodal training / rollout adapter | Not released yet |
| End-to-end GPU update, checkpoint and restart validation | Not completed |
| Formal RL reward approval | Pending empirical review |

Do **not** hand this preparation snapshot to a compute provider as if it already
contained a validated training launcher. The reward must be assessed first.

## Why the reward is checked first

PainterBench is open-ended creative drawing, not target-image reconstruction.
The candidate is **ViDrA-adapted**, not the unadapted ViDrA head and not
DINO/DreamSim similarity to an arbitrary target image.

The original paper reports correlation with human creativity ratings, but that
does not establish robustness when the score is directly optimized by RL.
The probe measures score changes after adding random strokes or a grid to
real drawings. Duplicate images check numerical repeatability. These controls
are diagnostics, not human creativity ground truth.

All asset revisions and SHA-256 values are in `sources.lock.json`.
The original-source checkpoint is a trusted, hash-verified pickle; do not
substitute an untrusted checkpoint.

## Standalone CPU validation

Python **3.13**, Git, at least **8 GiB RAM** (16 GiB recommended), and approximately
**4 GiB free disk** for the audit environment and DINO backbone. The first run
downloads public resources. Linux or Windows is supported for these audit
commands. CPU validation is not a GPU-training test.

```bash
python -m venv .venv
# Linux:
source .venv/bin/activate
# Windows PowerShell instead:
# .\.venv\Scripts\Activate.ps1

python -m pip install torch==2.9.1 torchvision==0.24.1 \
  --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-audit.txt
python -m unittest discover -s tests -v

# Four public final drawings; no private data or local project needed.
python scripts/reward_probe.py --work work/probe \
  --public-examples --device cpu --batch-size 2 --cpu-threads 4
python scripts/visualize_probe.py work/probe/reward_probe.json \
  --output work/probe/comparison.png
```

PowerShell users can put each command on one line instead of using Bash's `\`
line continuation. On a CUDA host, use a compatible PyTorch/torchvision build
and `--device cuda`. The scorer does not silently fall back to an undersized
CPU container.

`work/probe/reward_probe.json` contains all scores, paired deltas, ink
correlation and image hashes. Scores are not silently clipped into `[0,1]`.
The diagnostic deliberately does **not** automatically approve an RL reward.

## Score your own generated trajectories

Supply a directory of records in this small, documented interchange format:

```text
trajectories/
  trajectory_0001/
    complete.json
    final_painting.png
  trajectory_0002/
    complete.json
    unfinished_painting.png
```

`complete.json`:

```json
{"stimulus": "is01", "finished": true, "turns": 12}
```

For `finished=false`, store the last actual canvas as `unfinished_painting.png`;
do not omit the trajectory. Input images must be the native 400-by-400 canvas,
without titles, captions or coordinate overlays.

```bash
python scripts/reward_probe.py --work work/own-probe \
  --trajectories path/to/trajectories --limit 32 \
  --device cpu --batch-size 2 --cpu-threads 4
```

Selection is deterministic by ID hash, not by score. Raw inputs are never
overwritten. Derived controls are stored under `--work`.

## Method and scope

See [the baseline contract](docs/BASELINE_CONTRACT.md) for exact grouping,
masking, Teacher/reference distinction, coefficients, terminal reward handling,
evaluation scope and the required end-to-end checks.

The implementation extracts named standalone functions from pinned upstream
files rather than importing an incompatible old multimodal training runtime.
This preserves the public loss calculation; it does **not** certify that the
Qwen3.5 training adapter has already been validated.

No secret, user-specific server, private corpus, local model path or private
checkpoint is required by the published audit commands. See [NOTICE.md](NOTICE.md)
for provenance and third-party licenses.
