"""Render existing probe outputs side by side; does not edit source drawings."""
import argparse
import json
from pathlib import Path
from PIL import Image, ImageDraw


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--rows", type=int, default=4)
    args = parser.parse_args()
    result = json.loads(args.report.read_text(encoding="utf-8"))
    cases = result["cases"] if isinstance(result, dict) else result
    ids = [c["trajectory_id"] for c in cases if c["variant"] == "original"]
    variants = ["original", "add_random_10_lines", "add_random_50_lines", "add_grid"]
    chosen = ids[args.start:args.start + args.rows]
    sheet = Image.new("RGB", (1200, 330 * len(chosen)), "white")
    draw = ImageDraw.Draw(sheet)
    for row, tid in enumerate(chosen):
        for col, kind in enumerate(variants):
            c = next(c for c in cases if c["trajectory_id"] == tid and c["variant"] == kind)
            img = Image.open(c["path"]).convert("RGB").resize((290, 290))
            x, y = col * 300, row * 330
            sheet.paste(img, (x, y + 36))
            text = f"{tid} / {kind}"
            if "score" in c:
                text += f"\nscore={c['score']:.4f}"
            draw.text((x + 4, y + 2), text, fill="black")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.output)


if __name__ == "__main__":
    main()
