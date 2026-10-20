# Third-party provenance

This preparation repository is not affiliated with PainterBench, VAGEN, VERL,
SDAR, Qwen, or Meta.

- PainterBench environment, released drawings and ViDrA-adapted assets:
  [painterbench/painterbench](https://huggingface.co/datasets/painterbench/painterbench),
  dataset license **CC BY 4.0**, authors Shane K. A. Dalumura Hettige and
  Jonas Oppenlaender. Assets are downloaded from the pinned public dataset;
  this repository does not relicense them.
- Public loss functions: [VERL fork used by VAGEN](https://github.com/JamesKrW/verl)
  and [SDAR](https://github.com/ZJU-REAL/SDAR), Apache-2.0 source notices remain
  in the downloaded files. Only named pure functions are loaded.
- DINOv2 source and weights:
  [facebookresearch/dinov2](https://github.com/facebookresearch/dinov2),
  Apache-2.0. The official source is checked out at a fixed revision; checkpoint
  SHA-256 is pinned. No backbone retraining or random initialization is used.
- Qwen/Qwen3.5-4B is the proposed Actor initializer. See its public model card
  and license before model redistribution.

Derived random-line and grid images are generated only as reward diagnostics.
Their purpose is to test sensitivity, not to create SFT targets or benchmarks.

No private trajectory corpus, model checkpoint, authentication token, server
address, personal filesystem path or account configuration is included.
