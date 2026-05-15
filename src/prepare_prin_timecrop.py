import argparse
import shutil
from pathlib import Path

import cv2


def find_video_file(folder: Path):
    files = sorted(list(folder.glob("*.mp4")) + list(folder.glob("*.MP4")))
    if not files:
        raise FileNotFoundError(f"No mp4/MP4 found in {folder}")
    if len(files) > 1:
        print(f"[WARN] multiple video files in {folder}, using first: {files[0]}")
    return files[0]


def extract_timecrop_cv2(
    video_path: Path,
    out_dir: Path,
    start_sec: float,
    end_sec: float,
    out_fps: float,
    flip: bool,
):
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    src_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if src_fps <= 0:
        raise RuntimeError(f"Invalid FPS for video: {video_path}")

    print(f"[INFO] {video_path}")
    print(f"       src_fps={src_fps:.4f}, total_frames={total_frames}")

    saved = 0
    target_time = start_sec

    while target_time < end_sec:
        target_frame = int(round(target_time * src_fps))
        if target_frame >= total_frames:
            break

        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
        ok, frame = cap.read()
        if not ok:
            break

        if flip:
            frame = cv2.flip(frame, 1)

        out_path = out_dir / f"{saved:06d}.jpg"
        cv2.imwrite(str(out_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])

        saved += 1
        target_time = start_sec + saved / out_fps

    cap.release()

    print(
        f"       crop={start_sec}-{end_sec}s, "
        f"out_fps={out_fps}, saved={saved}, flip={flip}"
    )
    return saved


def parse_flip_views(value: str):
    if not value.strip():
        return set()
    return {x.strip().upper() for x in value.split(",") if x.strip()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw_root",
        default="/home/vrai/anilegin/dataset/PRIN_DATASET/Video ed Excel",
    )
    parser.add_argument("--out_root", default="data/prin_timecrop")
    parser.add_argument("--group", default="ID_0")
    parser.add_argument("--start_sec", type=float, required=True)
    parser.add_argument("--end_sec", type=float, required=True)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument(
        "--flip_views",
        default="TOP,FPV",
        help="Comma-separated original view names to horizontally flip. Example: TOP,FPV",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    raw_root = Path(args.raw_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    group_dir = raw_root / args.group
    if not group_dir.exists():
        raise FileNotFoundError(group_dir)

    expected_n = int(round((args.end_sec - args.start_sec) * args.fps))
    flip_views = parse_flip_views(args.flip_views)

    views = [
        ("TOP", "cam_top"),
        ("TPV", "cam_tpv"),
        ("FPV", "fpv"),
    ]

    for src_view, out_view in views:
        src_dir = group_dir / src_view
        video_path = find_video_file(src_dir)

        flip = src_view.upper() in flip_views

        out_name = f"{args.group}_{out_view}_000_{expected_n}"
        out_video_dir = out_root / out_name
        out_rgb_aligned = out_video_dir / "rgb_aligned"

        if out_video_dir.exists() and not args.overwrite:
            print(f"[SKIP] {out_video_dir} exists. Use --overwrite to recreate.")
            continue

        if out_video_dir.exists():
            shutil.rmtree(out_video_dir)

        extract_timecrop_cv2(
            video_path=video_path,
            out_dir=out_rgb_aligned,
            start_sec=args.start_sec,
            end_sec=args.end_sec,
            out_fps=args.fps,
            flip=flip,
        )

        print(f"[OK] {out_video_dir}")

    print("\nDone.")
    print(f"Dataset: {out_root}")


if __name__ == "__main__":
    main()
