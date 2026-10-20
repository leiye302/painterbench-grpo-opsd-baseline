"""Check the explicit public file allowlist before creating a Git history."""
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
PUBLIC_FILES = [
    ".gitignore", ".gitattributes", "README.md", "LICENSE", "NOTICE.md", "requirements-audit.txt",
    "sources.lock.json", "docs/BASELINE_CONTRACT.md",
    "painter_rl/__init__.py", "painter_rl/public_losses.py",
    "scripts/reward_probe.py", "scripts/visualize_probe.py",
    "scripts/check_release.py", "tests/test_public_losses.py",
    "requirements-runtime.txt", "docs/RELATED_GRPO.md",
    "painter_rl/actor.py", "painter_rl/checkpoints.py", "painter_rl/environment.py",
    "painter_rl/rewards.py", "painter_rl/training.py", "painter_rl/upstream.py",
    "scripts/train.py", "scripts/prepare.py", "scripts/evaluate.py",
    "scripts/validate_reward.py", "scripts/verify_restore.py",
    "configs/baseline.json", "configs/smoke.json", "configs/static_drawing_skill.txt",
    "tests/test_runtime.py",
]
PATTERNS = [
    re.compile(r"(?:sk|oc_sk|ghp|gho|github_pat)[_-][A-Za-z0-9_-]{16,}"),
    re.compile(r"connect\.[a-z0-9.-]+\.seetacloud\.com", re.I),
    re.compile(r"[A-Za-z]:[\\/](?:Users|app|桌面资料)[\\/]", re.I),
    re.compile(r"(?:password|api_key|access_token)\s*=\s*['\"][^'\"]{8,}['\"]", re.I),
]


def main():
    for name in PUBLIC_FILES:
        p = ROOT / name
        if not p.is_file() or p.is_symlink():
            raise RuntimeError(f"Missing or linked public file: {name}")
        text = p.read_text(encoding="utf-8")
        for pattern in PATTERNS:
            # This checker contains the patterns themselves, not credentials.
            if name != "scripts/check_release.py" and pattern.search(text):
                raise RuntimeError(f"Potential private information: {name}")
    if (ROOT / ".git").exists():
        tracked = subprocess.check_output(["git", "-C", str(ROOT), "ls-files"], text=True).splitlines()
        if set(tracked) != set(PUBLIC_FILES):
            raise RuntimeError("Git index differs from explicit public allowlist")
    print(f"Checked {len(PUBLIC_FILES)} allowlisted public files; no secrets or private paths found.")


if __name__ == "__main__":
    main()
