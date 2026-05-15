import argparse
import csv
import pickle
from collections import defaultdict, deque
from pathlib import Path

import cv2
import numpy as np


def list_frames(video_dir: Path):
    rgb_dir = video_dir / "rgb_aligned"
    paths = []
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        paths.extend(rgb_dir.glob(ext))
    return sorted(paths)


def read_result_pkl(pair_dir: Path):
    candidates = sorted(pair_dir.glob("result*.pkl"), key=lambda p: p.stat().st_mtime, reverse=True)
    for p in candidates:
        try:
            with open(p, "rb") as f:
                d = pickle.load(f)
            if isinstance(d, dict) and ("pred_offset" in d or "offset_error_list" in d):
                return p, d
        except Exception:
            continue
    return None, None


def best_offset_from_result(d):
    pred = d.get("pred_offset", None)
    conf = d.get("confidence", None)

    if pred is not None:
        return float(pred), conf, "ok"

    offsets = d.get("offsets", None)
    errors = d.get("offset_error_list", None)

    if offsets is None or errors is None:
        return None, None, "no_offset_curve"

    offsets = np.asarray(offsets)
    errors = np.asarray(errors, dtype=float)
    finite = np.isfinite(errors)

    if not finite.any():
        return None, None, "all_nan"

    valid_offsets = offsets[finite]
    valid_errors = errors[finite]
    best_i = int(np.argmin(valid_errors))

    return float(valid_offsets[best_i]), float(valid_errors[best_i]), "fallback_min_energy"


def group_prefix(name: str):
    parts = name.split("_")
    if len(parts) >= 2 and parts[0] == "ID":
        return f"{parts[0]}_{parts[1]}"
    return parts[0]


def collect_pairwise(result_root: Path):
    rows = []

    for group_dir in sorted(result_root.glob("ID_*")):
        if not group_dir.is_dir():
            continue

        for pair_dir in sorted(group_dir.glob("*__*")):
            if not pair_dir.is_dir():
                continue

            try:
                video1, video2 = pair_dir.name.split("__")
            except ValueError:
                continue

            result_path, d = read_result_pkl(pair_dir)

            if d is None:
                rows.append({
                    "group": group_dir.name,
                    "video1": video1,
                    "video2": video2,
                    "pred_offset": "",
                    "confidence": "",
                    "status": "missing_result",
                    "result_path": "",
                })
                continue

            offset, conf, status = best_offset_from_result(d)

            rows.append({
                "group": group_dir.name,
                "video1": video1,
                "video2": video2,
                "pred_offset": "" if offset is None else offset,
                "confidence": "" if conf is None else conf,
                "status": status,
                "result_path": str(result_path),
            })

    return rows


def estimate_global_offsets(pair_rows, group_name, offset_sign=1.0, ignore_pairs=None):
    ignore_pairs = set(ignore_pairs or [])

    valid_statuses = {"ok", "fallback_min_energy"}
    graph = defaultdict(list)
    videos = set()

    for r in pair_rows:
        if r["group"] != group_name:
            continue

        v1 = r["video1"]
        v2 = r["video2"]
        videos.add(v1)
        videos.add(v2)

        pair_name = f"{v1}__{v2}"
        reverse_pair_name = f"{v2}__{v1}"

        if pair_name in ignore_pairs or reverse_pair_name in ignore_pairs:
            continue

        if r["status"] not in valid_statuses:
            continue
        if r["pred_offset"] == "":
            continue

        off = float(r["pred_offset"]) * float(offset_sign)

        # Convention used here:
        # offset means frame2 ≈ frame1 + offset.
        # Therefore global[v2] = global[v1] + offset.
        graph[v1].append((v2, off))
        graph[v2].append((v1, -off))

    if not videos:
        return None, {}

    # Prefer TPV as reference if present.
    reference = None
    for v in sorted(videos):
        if "tpv" in v.lower():
            reference = v
            break
    if reference is None:
        reference = sorted(videos)[0]

    global_offsets = {v: None for v in videos}
    global_offsets[reference] = 0.0

    q = deque([reference])
    while q:
        cur = q.popleft()
        for nxt, delta in graph[cur]:
            if global_offsets[nxt] is None:
                global_offsets[nxt] = global_offsets[cur] + delta
                q.append(nxt)

    return reference, global_offsets


def resize_keep_aspect(img, target_h):
    h, w = img.shape[:2]
    scale = target_h / float(h)
    new_w = max(1, int(round(w * scale)))
    return cv2.resize(img, (new_w, target_h), interpolation=cv2.INTER_AREA)


def make_panel(img, label, frame_idx, target_h, target_w):
    panel = np.zeros((target_h, target_w, 3), dtype=np.uint8)

    if img is not None:
        img = resize_keep_aspect(img, target_h)
        h, w = img.shape[:2]

        if w > target_w:
            img = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_AREA)
            h, w = img.shape[:2]

        x0 = (target_w - w) // 2
        panel[:, x0:x0 + w] = img

    cv2.rectangle(panel, (0, 0), (target_w, 42), (0, 0, 0), -1)
    cv2.putText(
        panel,
        f"{label} | frame {frame_idx}",
        (8, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return panel


def create_merged_video(dataset_root: Path, group_name: str, global_offsets: dict, out_path: Path, fps: float, max_seconds: float, panel_height: int):
    usable = {v: off for v, off in global_offsets.items() if off is not None}
    if len(usable) < 2:
        print(f"[SKIP VIDEO] {group_name}: fewer than 2 videos with offsets")
        return

    videos = []
    for name, off in sorted(usable.items()):
        video_dir = dataset_root / name
        frames = list_frames(video_dir)
        if not frames:
            continue
        videos.append((name, float(off), frames))

    if len(videos) < 2:
        print(f"[SKIP VIDEO] {group_name}: fewer than 2 videos with frames")
        return

    widths = []
    for _, _, frames in videos:
        img0 = cv2.imread(str(frames[0]))
        if img0 is None:
            continue
        h, w = img0.shape[:2]
        widths.append(int(round(w * panel_height / float(h))))

    panel_width = max(max(widths), 360) if widths else 480
    total_w = panel_width * len(videos)
    total_h = panel_height

    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (total_w, total_h),
    )

    total_frames = int(round(max_seconds * fps))

    for t in range(total_frames):
        panels = []
        for name, off, frames in videos:
            idx = int(round(t + off))
            img = None
            if 0 <= idx < len(frames):
                img = cv2.imread(str(frames[idx]))
            panels.append(make_panel(img, name, idx, total_h, panel_width))

        canvas = np.concatenate(panels, axis=1)
        writer.write(canvas)

    writer.release()
    print(f"[VIDEO] saved {out_path}")


def write_csv(path: Path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", required=True)
    parser.add_argument("--result_root", required=True)
    parser.add_argument("--group_name", default=None)
    parser.add_argument("--out_csv", default=None)
    parser.add_argument("--out_global_csv", default=None)
    parser.add_argument("--out_video_dir", default=None)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--max_seconds", type=float, default=15.0)
    parser.add_argument("--panel_height", type=int, default=480)
    parser.add_argument("--offset_sign", type=float, default=1.0)
    parser.add_argument(
        "--ignore_pair",
        action="append",
        default=[],
        help="Pair to ignore, e.g. ID_0_cam_top_000_150__ID_0_fpv_000_150. Can be repeated.",
    )
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    result_root = Path(args.result_root)

    out_csv = Path(args.out_csv) if args.out_csv else result_root / "pairwise_offsets.csv"
    out_global_csv = Path(args.out_global_csv) if args.out_global_csv else result_root / "global_offsets.csv"
    out_video_dir = Path(args.out_video_dir) if args.out_video_dir else result_root / "merged_videos"

    pair_rows = collect_pairwise(result_root)

    write_csv(
        out_csv,
        pair_rows,
        ["group", "video1", "video2", "pred_offset", "confidence", "status", "result_path"],
    )

    groups = sorted({r["group"] for r in pair_rows})
    if args.group_name:
        groups = [args.group_name]

    global_rows = []

    for group in groups:
        reference, offsets = estimate_global_offsets(
            pair_rows,
            group,
            offset_sign=args.offset_sign,
            ignore_pairs=args.ignore_pair,
        )

        print("\nGROUP:", group)
        print("reference:", reference)
        print("global offsets:")
        for video, off in sorted(offsets.items()):
            print(f"  {video}: {off}")
            global_rows.append({
                "group": group,
                "video": video,
                "global_offset_frames": "" if off is None else off,
                "reference_video": "" if reference is None else reference,
            })

        if reference is not None:
            out_video = out_video_dir / f"{group}_synced_merged.mp4"
            create_merged_video(
                dataset_root=dataset_root,
                group_name=group,
                global_offsets=offsets,
                out_path=out_video,
                fps=args.fps,
                max_seconds=args.max_seconds,
                panel_height=args.panel_height,
            )

    write_csv(
        out_global_csv,
        global_rows,
        ["group", "video", "global_offset_frames", "reference_video"],
    )

    print("\nSaved pairwise CSV:", out_csv)
    print("Saved global CSV:  ", out_global_csv)
    print("Saved videos under:", out_video_dir)


if __name__ == "__main__":
    main()
