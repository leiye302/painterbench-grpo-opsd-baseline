# PainterBench GRPO + OPSD baseline

**Status: native training adapter implemented; final GPU checkpoint/restore
verification is being completed. No long-run learning result is claimed.**

This project prepares a Qwen3.5-4B baseline for PainterBench's native multi-turn
drawing tools. GRPO uses the pinned public VAGEN/VERL loss functions; gated OPSD
uses the pinned SDAR function with `gate_beta=5`. The baseline has no Route-SFT,
RETHINK, Analyzer or paid API dependency.

Validation boundaries:

| Component | Status |
| --- | --- |
| Exact public GRPO and OPSD pure-loss functions | Integrated; source SHA and AST receipts |
| Grouping, generated-token mask, gradient and microbatch tests | Implemented |
| Official ViDrA-adapted scoring and shortcut probe | Implemented |
| Public example images for standalone scoring | Downloadable from pinned public dataset |
| Qwen3.5 native multimodal training / rollout adapter | Implemented, single process; no stale rollout server |
| GPU rollout, reward, backward and optimizer update | Real Qwen3.5-4B integration test completed |
| Full checkpoint and fresh-process restart | Final verification pending |
| Reward's 480-image human-rated test split | Live re-score matches released results; correlation 0.8459 |
| 50-update training / improved drawing quality | Not demonstrated; do not infer from a smoke test |

Read [related implementations and reward evidence](docs/RELATED_GRPO.md) before
interpreting the score. The baseline is not an official PainterBench RL recipe.

## Standalone GPU setup

The reference runtime uses Linux, Python **3.12**, a CUDA-13-compatible driver,
Torch **2.13.0+cu130**, torchvision **0.28.0**, and the versions in
`requirements-runtime.txt`. The short real integration test uses one GPU with
approximately 84 GiB addressable VRAM. Full 32K-context memory/throughput is **not**
established by that short test. Smaller GPUs and other kernel stacks require
their own smoke test; no quantization, LoRA or frozen vision tower is silently
substituted.

The validated calculation uses **FP32 parameters, compute, gradients and Adam
states**. An initial BF16 trial showed cached-generation/full-forward
log-probability differences; FP32 resolves this in the tested path. BF16 or FLA
acceleration should only be enabled after a separate probability/gradient
equivalence test. This is a correctness-first single-GPU reference, not a claim
of optimal throughput.

Provide at least **60 GiB free persistent checkpoint storage per saved update**.
One 4.54B full FP32 model plus Adam states is about 51 GiB. Default saves at
10/20/30/40/50 retain **all five**; budget roughly **300 GiB plus model/assets and
rollout records**. The program never automatically deletes old checkpoints.
Do not use `/dev/shm` for a durable training run.

```bash
git clone https://github.com/leiye302/painterbench-grpo-opsd-baseline.git
cd painterbench-grpo-opsd-baseline
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.13.0 torchvision==0.28.0 \
  --index-url https://download.pytorch.org/whl/cu130
python -m pip install -r requirements-runtime.txt

# Public, pinned sources and model. No private corpus or credentials.
python scripts/prepare.py --work work --model-dir models/Qwen3.5-4B
python scripts/reward_probe.py --work work/reward --prepare-only
python -m unittest discover -s tests -v

# Independently reproduce the official head's full test-set result.
python scripts/validate_reward.py --work work/reward_validation \
  --assets-work work/reward --rescore 480 --device cuda
```

Preparation needs internet access to Hugging Face, GitHub and Meta's public
weight server. After preparation, copy the complete model and `work` caches to
an offline host if necessary, keeping their receipts and hashes; all paths
are supplied by arguments, not a developer's local machine.

## First: one real integration update

```bash
python scripts/train.py \
  --model models/Qwen3.5-4B --work work --reward-work work/reward \
  --config configs/smoke.json --output runs/smoke \
  --checkpoints checkpoints/smoke \
  --smoke --skip-initial-eval --skip-final-eval

python scripts/verify_restore.py \
  --checkpoint checkpoints/smoke/update_0001 --run runs/smoke --work work
```

The smoke has one group of two trajectories, two turns per trajectory,
2,048 new-token cap and one optimizer update. Both capped final canvases are
scored. This verifies wiring, not complete-drawing performance. The restore
command verifies every file hash, loads optimizer/RNG in a new process, compares
token scores and executes another native drawing rollout using restored weights.

## Baseline: 50 updates

```bash
python scripts/train.py \
  --model models/Qwen3.5-4B --work work --reward-work work/reward \
  --config configs/baseline.json --output runs/baseline \
  --checkpoints checkpoints/baseline
```

Each update samples 8 trajectories of the same initial stimulus. The next
update advances to the next training stimulus, cycling through 24. There is
one global generated-token-mean update, not one independent update per turn.
Default rollout limits are 64 turns, 16,384 new tokens per turn and 32,768
context tokens. Context overflow aborts with evidence; it does not silently
truncate or discard a member of a GRPO group.

Evaluation runs before training and every 10 updates on fixed held-out stimuli
`is03,is12,is13,is16,ot02,ot03`, two independent fixed-seed samples each.
Sampling uses temperature 1, no top-k/top-p truncation or repetition penalty.
This small deterministic task split is this baseline's protocol, not a claim
of a standard official PainterBench train/test split. Compare experimental arms
with the same tasks, seeds, budgets and reward assets.

Resume into a **new** run directory, from a verified complete checkpoint:

```bash
python scripts/train.py \
  --model models/Qwen3.5-4B --work work --reward-work work/reward \
  --config configs/baseline.json --output runs/baseline-resumed \
  --checkpoints checkpoints/baseline \
  --resume checkpoints/baseline/update_0020 --skip-initial-eval
```

Keep code, full configuration, base hashes, environment and skill unchanged.
Do not rename a checkpoint to a later update. Partial directories are never
resumed. A crash during rollout repeats that update from the previous complete
checkpoint and its recorded seed; failed-attempt evidence remains separate.

Standalone evaluation of either the base or an exported checkpoint:

```bash
python scripts/evaluate.py \
  --model checkpoints/baseline/update_0050/model \
  --work work --reward-work work/reward \
  --output runs/eval-final --config configs/baseline.json
```

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
This preserves the public loss calculation. Environment transport, native-token
parsing, checkpoint storage and the Transformers runtime are adapters maintained
here, not unmodified upstream end-to-end training code.

No secret, user-specific server, private corpus, local model path or private
checkpoint is required by the published audit commands. See [NOTICE.md](NOTICE.md)
for provenance and third-party licenses.
