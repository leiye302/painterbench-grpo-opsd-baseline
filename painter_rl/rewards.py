"""Frozen official creativity reward; no target-image substitution or shaping."""
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


class ViDrAReward:
    def __init__(self, work, device="cuda"):
        import joblib
        import sklearn
        import torch
        from scripts.reward_probe import prepare
        if sklearn.__version__ != "1.8.0":
            raise RuntimeError("Use scikit-learn==1.8.0 for the official frozen head")
        self.device = torch.device(device)
        lock = json.loads((ROOT / "sources.lock.json").read_text(encoding="utf-8"))
        assets = prepare(Path(work), lock)
        spec = importlib.util.spec_from_file_location("painter_reward_features", assets / "dinov2_features.py")
        self.module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = self.module
        spec.loader.exec_module(self.module)
        ckpt = joblib.load(assets / "vidra_adapted.joblib")
        if not ckpt.get("adapted") or ckpt["target"] != "rated_creativity":
            raise ValueError("Not the released agent-adapted creativity head")
        self.config = self.module.config_from_stored(ckpt["feature_config"])
        self.head = ckpt["pipeline"]
        self.transform = self.module.build_transform(self.config)
        self.model = torch.hub.load(str(assets / "dinov2"), self.config.variant,
                                   source="local", pretrained=False, verbose=False)
        self.model.load_state_dict(torch.load(assets / "dinov2_vitl14_pretrain.pth",
                                              map_location="cpu", weights_only=True), strict=True)
        self.model.eval().requires_grad_(False).to(self.device)
        self.sha256 = lock["reward"]["sha256"]

    def score(self, paths, batch_size=2):
        import numpy as np
        from PIL import Image
        for path in paths:
            with Image.open(path) as image:
                if image.size != (400, 400):
                    raise ValueError("Reward must see the raw 400x400 canvas, not annotated screenshots")
        features = self.module.extract_features_for_paths(
            [Path(p) for p in paths], self.model, self.transform, self.device,
            batch_size=batch_size, config=self.config)
        scores = self.head.predict(features)
        if not np.isfinite(scores).all():
            raise FloatingPointError("Reward failure must abort the update, not become score zero")
        return [float(s) for s in scores]  # no silent clipping or ink/length/finish bonus

    def to(self, device):
        import torch
        self.device = torch.device(device)
        self.model.to(self.device)
        return self
