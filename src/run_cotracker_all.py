import argparse
import os
import subprocess
from pathlib import Path


def is_dynamic_camera(video_name: str) -> bool:
    """
    PRIN convention:
    - FPV is dynamic camera
    - TPV and TOP are static cameras because we named them cam_tpv / cam_top
    """
    name = video_name.lower()

    if "fpv" in name:
        return True

    if "cam" in name or "tpv" in name or "top" in name:
        return False

    # fallback: assume static if unknown
    return False


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset_root",
        default="/home/vrai/anilegin/visualsync/data/prin_preprocessed_20s"
    )
    parser.add_argument(
        "--track_root",
        default="/home/vrai/anilegin/visualsync/tracks/prin_tracks"
    )
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--mask_prefix", default="deva_improved")
    parser.add_argument("--skip_exist", action="store_true")

    # Static camera settings: TPV / TOP
    parser.add_argument("--static_interval", type=int, default=10)
    parser.add_argument("--static_grid_step", type=int, default=8)

    # Dynamic camera settings: FPV
    parser.add_argument("--dynamic_interval", type=int, default=20)
    parser.add_argument("--dynamic_grid_step", type=int, default=8)

    # Optional: force all videos to use one mode
    parser.add_argument(
        "--force_mode",
        choices=["auto", "static", "dynamic"],
        default="auto"
    )

    # Optional: run only one camera type
    parser.add_argument(
        "--only",
        choices=["all", "fpv", "tpv", "top", "static", "dynamic"],
        default="all"
    )

    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    track_root = Path(args.track_root)
    track_root.mkdir(parents=True, exist_ok=True)

    video_dirs = sorted([
        p for p in dataset_root.iterdir()
        if p.is_dir() and p.name.startswith("ID_")
    ])

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.gpu

    for video_dir in video_dirs:
        video_name = video_dir.name
        name_lower = video_name.lower()

        auto_dynamic = is_dynamic_camera(video_name)

        if args.force_mode == "dynamic":
            dynamic = True
        elif args.force_mode == "static":
            dynamic = False
        else:
            dynamic = auto_dynamic

        # Optional filtering
        if args.only == "fpv" and "fpv" not in name_lower:
            continue
        if args.only == "tpv" and "tpv" not in name_lower:
            continue
        if args.only == "top" and "top" not in name_lower:
            continue
        if args.only == "static" and dynamic:
            continue
        if args.only == "dynamic" and not dynamic:
            continue

        rgb_dir = video_dir / "rgb_aligned"
        mask_dir = video_dir / args.mask_prefix / "Annotations"

        if not rgb_dir.exists():
            print(f"[MISS RGB] {rgb_dir}")
            continue

        if not mask_dir.exists():
            print(f"[MISS MASK] {mask_dir}")
            continue

        out_dir = track_root / video_name
        out_path = out_dir / "tracks.pkl"

        if args.skip_exist and out_path.exists():
            print(f"[SKIP] {out_path}")
            continue

        out_dir.mkdir(parents=True, exist_ok=True)

        if dynamic:
            interval = args.dynamic_interval
            grid_step = args.dynamic_grid_step
            mode = "dynamic"
        else:
            interval = args.static_interval
            grid_step = args.static_grid_step
            mode = "static"

        print(f"\n[CONFIG] {video_name}")
        print(f"         mode={mode}, interval={interval}, grid_step={grid_step}")

        cmd = [
            "python", "src/run_cotracker_v5.py",
            "--video_dir", str(rgb_dir),
            "--mask_dir", str(mask_dir),
            "--save_dir", str(out_dir),
            "--interval", str(interval),
            "--grid_step", str(grid_step),
        ]

        print("[RUN]", " ".join(cmd))
        subprocess.run(cmd, check=True, env=env)


if __name__ == "__main__":
    main()
