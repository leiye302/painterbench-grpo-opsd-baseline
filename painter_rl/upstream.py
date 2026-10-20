"""Fetch the native environment from a fixed public dataset revision."""
import hashlib
import json
from pathlib import Path
import sys
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def prepare_environment(work):
    from scripts.reward_probe import download, atomic_json
    lock = json.loads((ROOT / "sources.lock.json").read_text(encoding="utf-8"))["painterbench"]
    name, rev = lock["repository"], lock["revision"]
    tree_url = f"https://huggingface.co/api/datasets/{name}/tree/{rev}/painterbench?recursive=true&limit=1000"
    tree_file = Path(work) / "environment_tree.json"
    download(tree_url, tree_file)
    tree = json.loads(tree_file.read_text(encoding="utf-8"))
    sources = [x for x in tree if x["type"] == "file" and x["path"].endswith(".py")]
    if not 15 <= len(sources) < 100:
        raise RuntimeError("Unexpected environment source manifest")
    target = Path(work) / "painterbench_source"
    hashes = {}
    for item in sources:
        path = target / item["path"]
        checksum = download(f"https://huggingface.co/datasets/{name}/resolve/{rev}/{item['path']}", path)
        raw = path.read_bytes()
        blob = hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()
        if blob != item["oid"]:
            raise RuntimeError(f"Upstream Git blob mismatch: {item['path']}")
        hashes[item["path"]] = checksum
    atomic_json(target / "environment_receipt.json", {"revision": rev, "files": hashes})
    return target


def activate_environment(work):
    source = Path(work) / "painterbench_source"
    receipt = json.loads((source / "environment_receipt.json").read_text(encoding="utf-8"))
    lock = json.loads((ROOT / "sources.lock.json").read_text(encoding="utf-8"))
    if receipt["revision"] != lock["painterbench"]["revision"]:
        raise RuntimeError("Environment revision changed")
    for name, checksum in receipt["files"].items():
        if hashlib.sha256((source / name).read_bytes()).hexdigest() != checksum:
            raise RuntimeError(f"Environment modified: {name}")
    sys.path.insert(0, str(source.resolve()))
    return receipt
