import argparse
import json
from pathlib import Path


def infer_view_name(folder_name: str):
    name = folder_name.lower()

    if "tpv" in name:
        return "TPV"
    if "fpv" in name:
        return "FPV"
    if "top" in name:
        return "TOP"

    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", required=True)
    parser.add_argument(
        "--dynamic",
        default="hand,arm",
        help="Comma-separated dynamic tags, example: hand,arm,object",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    if not data_root.exists():
        raise FileNotFoundError(data_root)

    dynamic_tags = [x.strip() for x in args.dynamic.split(",") if x.strip()]

    found = []

    for video_dir in sorted(data_root.iterdir()):
        if not video_dir.is_dir():
            continue

        view = infer_view_name(video_dir.name)
        if view is None:
            continue

        gpt_dir = video_dir / "gpt_video"
        tags_path = gpt_dir / "tags.json"

        if tags_path.exists() and not args.overwrite:
            print(f"[SKIP] {tags_path} exists. Use --overwrite to replace.")
            continue

        gpt_dir.mkdir(parents=True, exist_ok=True)

        tags = {
            "dynamic": dynamic_tags
        }

        with open(tags_path, "w") as f:
            json.dump(tags, f, indent=2)

        found.append((view, tags_path))
        print(f"[OK] {view}: {tags_path}")

    if not found:
        print(f"[WARN] no TOP/TPV/FPV folders found under {data_root}")
    else:
        print("\nCreated tag files:")
        for view, path in found:
            print(f"  {view}: {path}")


if __name__ == "__main__":
    main()
