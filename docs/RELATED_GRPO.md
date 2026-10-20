# Which image-agent GRPO implementation fits this task?

The optimized model here is a **language/vision controller** emitting native
drawing-tool calls, not a diffusion model or an autoregressive image tokenizer.
PainterBench's task is open-ended creativity using a given starting shape,
not reconstruction of a specified target image.

| Public work | Relevant mechanism | What transfers / what does not |
| --- | --- | --- |
| [VAGEN / VERL](https://github.com/mll-lab-nu/VAGEN) | Multi-turn visual-agent rollouts, response-token masks, group-relative advantages | Best structural fit. Reuse the pinned public GRPO functions and whole-trajectory grouping. Do not replace this task's reward with its SVG reconstruction reward or claim to reproduce VAGEN-Full. |
| [Teaching an Agent to Sketch One Part at a Time](https://arxiv.org/html/2603.19500v1) ([code](https://github.com/duxiaodan/teaching-an-agent-to-sketch-one-part-at-a-time)) | Multi-turn GRPO for sequential vector parts, with rendered feedback | Confirms that drawing-controller RL is feasible. Its part-specific DreamSim targets and process advantages require reference parts unavailable here. The released training integration uses Tinker, so it is not a drop-in standalone local-GPU backend. |
| [GenAgent](https://arxiv.org/html/2601.18543v1) | GRPO over a multimodal tool controller; judges text-condition fulfillment and improvement | Controller structure is relevant, but its text constraints, image editor, reward composition and trajectory resampling are different. Do not silently transplant them. Its paper does not supply a PainterBench reward. |
| [AR-GRPO](https://github.com/Kwai-Klear/AR-GRPO) | RL of an autoregressive image generator | Optimizes image-generation tokens rather than a sequence of tool decisions. Not a minimal runtime adaptation for this baseline. |

Selection: unchanged **public VERL GRPO + SDAR gated OPSD**, with a small native
PainterBench/model adapter. The adapter is new code and must be tested; this is
not a claim that an existing paper has already validated the entire combination.
Source commits, source-file hashes and selected functions are pinned.

Current VAGEN versions also expose separate environment, harness and training
interfaces. These are useful engineering references, but copying a newer
framework wholesale would change more components than preserving the already
pinned objective. Neither a repository's popularity nor a passing loss unit
test establishes that the Actor/runtime/reward integration is correct.

## Why this reward

[PainterBench's release](https://huggingface.co/datasets/painterbench/painterbench)
contains a task-adapted creativity predictor and human ratings. The selected
ViDrA-adapted model uses fixed DINOv2 features and a released regression head.
It consumes the final raw canvas, not an asserted title or model rationale.

Re-scoring the full released 480-image test set with the actual checkpoint gave:

| Metric | Live re-score |
| --- | ---: |
| Pearson correlation with human creativity | 0.845881 |
| Spearman correlation | 0.834583 |
| Mean squared error on normalized ratings | 0.008484 |
| Same-stimulus pairwise agreement, rating difference at least 0.1 | 0.910472 |
| Qualifying within-stimulus pairs | 2,435 |

There are recorded fixed-head predictions for 423 of these images, not all 480;
57 are human-reference drawings without that public prediction entry. The
live test includes all 480. Maximum difference on the 423 overlapping recorded
predictions was below 0.0001. This is scorer/feature-path validation, not an RL
learning result or a 91% objective-correctness claim.

The independent-source boundary matters: the scorer was fitted on other
PainterBench drawings, so an Actor held-out stimulus is not necessarily unseen
by the reward model. Report that separately.

## Remaining limitations

Synthetic stroke/grid probes reveal sensitivity to added geometry. More ink
does not consistently help, but some pattern additions raise the score. Do not
turn that observation into an invented grid penalty, tune the reward on test
images, or describe a proxy as immune to reward hacking.

For a long run, preserve generated final images and compare checkpoints with
independent blinded creativity/recognizability judgements as well as this
training score. A rising optimized score alone is insufficient evidence of
better drawing. No paid judge, secret API, arbitrary target image, or private
annotation corpus is needed to run the baseline.
