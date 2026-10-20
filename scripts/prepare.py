"""Download public native environment and verified public loss sources."""
import argparse
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from painter_rl.upstream import prepare_environment
from painter_rl.public_losses import prepare_sources

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, default=Path("work"))
    parser.add_argument("--model-dir", type=Path,
                        help="Optionally download the public pinned Actor snapshot here")
    args = parser.parse_args()
    prepare_environment(args.work)
    prepare_sources(args.work / "public_sources")
    if args.model_dir:
        from huggingface_hub import snapshot_download
        lock = json.loads((ROOT / "sources.lock.json").read_text())["actor"]
        snapshot_download(lock["repository"], revision=lock["revision"],
                          local_dir=args.model_dir,
                          allow_patterns=["*.safetensors", "*.json", "*.jinja",
                                          "*.txt", "*.model", "README.md", "LICENSE"])
    print("Pinned native environment and unchanged public losses prepared.")
