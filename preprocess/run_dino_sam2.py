import os
import argparse
import cv2
import json
import torch
import numpy as np
import supervision as sv
from pathlib import Path
import sys
import torchvision
from tqdm import tqdm
from collections import defaultdict
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

sys.path.insert(0, "./Grounded-SAM-2")

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


parser = argparse.ArgumentParser()
parser.add_argument("--grounding-model", default="IDEA-Research/grounding-dino-base")
parser.add_argument("--workdir", required=True)
parser.add_argument("--sam2-checkpoint", default="./preprocess/pretrained/sam2.1_hiera_large.pt")
parser.add_argument("--sam2-model-config", default="configs/sam2.1/sam2.1_hiera_l.yaml")
parser.add_argument("--input-dir", default="gpt_video")
parser.add_argument("--force-cpu", action="store_true")
parser.add_argument("--use-sport-specific", action="store_true")
parser.add_argument("--skip-existing", action="store_true")
parser.add_argument("--box-threshold", type=float, default=0.4)
parser.add_argument("--text-threshold", type=float, default=0.4)
parser.add_argument("--max-area-percent", type=float, default=70.0)
args = parser.parse_args()


GROUNDING_MODEL = args.grounding_model
INPUT_DIR = Path(args.workdir)
SAM2_CHECKPOINT = args.sam2_checkpoint
SAM2_MODEL_CONFIG = args.sam2_model_config
DEVICE = "cuda" if torch.cuda.is_available() and not args.force_cpu else "cpu"
USE_SPORT_SPECIFIC = args.use_sport_specific


if DEVICE == "cuda":
    torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


print(f"Using device: {DEVICE}")

sam2_model = build_sam2(
    SAM2_MODEL_CONFIG,
    SAM2_CHECKPOINT,
    device=DEVICE,
)
sam2_predictor = SAM2ImagePredictor(sam2_model)

processor = AutoProcessor.from_pretrained(GROUNDING_MODEL)
grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(GROUNDING_MODEL).to(DEVICE)


def get_group_from_video(video_name):
    """
    PRIN names:
        ID_0_fpv_000_200
        ID_0_cam_tpv_000_200
        ID_0_cam_top_000_200

    Group should be:
        ID_0
    """
    parts = video_name.split("_")
    if len(parts) >= 2 and parts[0] == "ID":
        return f"{parts[0]}_{parts[1]}"
    return parts[0]


def get_dyn_objs_from_gpt_output(json_path):
    try:
        with open(json_path, "r") as f:
            data = json.load(f)

        if "dynamic" in data:
            return data["dynamic"]

    except Exception as e:
        print(f"[WARN] Error loading {json_path}: {e}")

    return []


def get_image_dir(work_dir):
    """
    Prefer rgb_aligned because our PRIN adapter writes frames there.
    Fall back to rgb for official VisualSync format.
    """
    rgb_aligned = work_dir / "rgb_aligned"
    rgb = work_dir / "rgb"

    if rgb_aligned.exists():
        return rgb_aligned

    if rgb.exists():
        return rgb

    return None


def filter_boxes_by_size(boxes, image_hw, max_area_percent=70.0):
    """
    image_hw: (height, width)
    boxes: [x1, y1, x2, y2]
    """
    if boxes is None or len(boxes) == 0:
        return []

    image_height, image_width = image_hw
    image_area = image_height * image_width
    max_area = (max_area_percent / 100.0) * image_area

    boxes_cpu = boxes.detach().cpu()
    box_areas = (boxes_cpu[:, 2] - boxes_cpu[:, 0]) * (boxes_cpu[:, 3] - boxes_cpu[:, 1])

    keep_indices = [
        i for i, area in enumerate(box_areas)
        if 0 < float(area) <= max_area
    ]

    return keep_indices


def save_empty_mask(image_file, image, output_ann_dir):
    """
    Save a blank single-channel annotation mask.

    Later scripts expect one mask per frame.
    Background = 0.
    Objects = positive integer IDs.
    """
    mask_name = Path(image_file).with_suffix(".png").name
    blank_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.imwrite(str(output_ann_dir / mask_name), blank_mask)


def save_instance_mask(
    image_file,
    image,
    masks,
    input_boxes,
    confidences,
    output_ann_dir,
):
    """
    Save only one VisualSync-compatible instance-ID mask:

        deva_improved/Annotations/000000.png

    Pixel values:
        0 = background
        1, 2, 3, ... = detected object instances
    """
    class_ids = np.array(list(range(len(confidences))))

    detections = sv.Detections(
        xyxy=input_boxes,
        mask=masks.astype(bool),
        class_id=class_ids,
        confidence=np.array(confidences),
    )

    if len(detections.class_id) == 0:
        save_empty_mask(image_file, image, output_ann_dir)
        return

    nms_idx = torchvision.ops.nms(
        torch.from_numpy(detections.xyxy).float(),
        torch.from_numpy(detections.confidence).float(),
        0.5,
    ).numpy().tolist()

    detections.xyxy = detections.xyxy[nms_idx]
    detections.class_id = detections.class_id[nms_idx]
    detections.confidence = detections.confidence[nms_idx]
    detections.mask = detections.mask[nms_idx]

    if len(detections.class_id) == 0:
        save_empty_mask(image_file, image, output_ann_dir)
        return

    masks = detections.mask

    # uint16 is safer if there are many instances.
    # Most scripts can read it as numeric mask.
    id_mask = np.zeros(image.shape[:2], dtype=np.uint16)

    # Render larger masks first, then smaller masks overwrite them.
    # This helps hands/arms overwrite person if overlapping.
    mask_size = [np.sum(mask) for mask in masks]
    sorted_mask_idx = np.argsort(mask_size)[::-1]

    for instance_id, idx in enumerate(sorted_mask_idx, start=1):
        mask = masks[idx]
        id_mask[mask] = instance_id

    mask_name = Path(image_file).with_suffix(".png").name
    cv2.imwrite(str(output_ann_dir / mask_name), id_mask)


# ------------------------------------------------------------
# Step 1: collect dynamic objects
# ------------------------------------------------------------

videos = sorted([
    x for x in os.listdir(INPUT_DIR)
    if (INPUT_DIR / x).is_dir()
])[::-1]

group_dyn_objects = defaultdict(set)

print("Step 1: Collecting objects that are actually moving by group...")

for video in tqdm(videos, desc="Collecting moving objects"):
    group = get_group_from_video(video)
    text_prompt_file = INPUT_DIR / video / args.input_dir / "tags.json"

    if text_prompt_file.exists():
        moving_objs = get_dyn_objs_from_gpt_output(text_prompt_file)
        group_dyn_objects[group].update(set(moving_objs))

group_dyn_objects = {
    group: sorted(list(objects))
    for group, objects in group_dyn_objects.items()
}

print(group_dyn_objects)


# ------------------------------------------------------------
# Step 2: process each video
# ------------------------------------------------------------

print("\nStep 2: Processing videos...")

for video in tqdm(videos, desc="Processing videos"):
    work_dir = INPUT_DIR / video
    image_dir_path = get_image_dir(work_dir)

    if image_dir_path is None:
        print(f"[WARN] {video}: no rgb_aligned/ or rgb/ folder, skipping")
        continue

    group = get_group_from_video(video)

    if USE_SPORT_SPECIFIC and group in group_dyn_objects and len(group_dyn_objects[group]) > 0:
        dyn_objs = group_dyn_objects[group]
        prompt_source = f"group-specific moving objects ({group})"
    else:
        text_prompt_file = work_dir / args.input_dir / "tags.json"

        if not text_prompt_file.exists():
            print(f"[WARN] {video}: missing {text_prompt_file}, skipping")
            continue

        with open(text_prompt_file, "r") as f:
            dyn_objs = json.load(f).get("dynamic", [])

        prompt_source = "video-specific moving objects"

    if len(dyn_objs) == 0:
        print(f"[WARN] {video}: empty dynamic object list, using fallback ['person']")
        dyn_objs = ["person"]

    text_input = ". ".join(dyn_objs) + "."

    print(
        f"\nProcessing {video} with {prompt_source} "
        f"prompt containing {len(dyn_objs)} objects: {dyn_objs}"
    )

    output_root = work_dir / "deva_improved"
    output_ann_dir = output_root / "Annotations"

    output_ann_dir.mkdir(parents=True, exist_ok=True)

    with open(output_root / "prompt_used.json", "w") as f:
        json.dump({
            "moving_objects": dyn_objs,
            "prompt_source": prompt_source,
            "text_input": text_input,
            "box_threshold": args.box_threshold,
            "text_threshold": args.text_threshold,
            "max_area_percent": args.max_area_percent,
        }, f, indent=4)

    image_files = sorted([
        f for f in os.listdir(image_dir_path)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])

    if len(image_files) == 0:
        print(f"[WARN] {video}: no frames found in {image_dir_path}")
        continue

    for image_file in tqdm(image_files, desc=f"{video} frames", leave=False):
        full_image_path = image_dir_path / image_file
        out_ann_file = output_ann_dir / Path(image_file).with_suffix(".png").name

        if args.skip_existing and out_ann_file.exists():
            continue

        image_pil = Image.open(full_image_path).convert("RGB")
        image = np.array(image_pil)
        h, w = image.shape[:2]

        sam2_predictor.set_image(image)

        inputs = processor(
            images=image,
            text=text_input,
            return_tensors="pt",
        ).to(DEVICE)

        with torch.no_grad():
            outputs = grounding_model(**inputs)

        results = processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=args.box_threshold,
            text_threshold=args.text_threshold,
            target_sizes=[image_pil.size[::-1]],
        )

        boxes = results[0]["boxes"]

        keep_indices = filter_boxes_by_size(
            boxes,
            image_hw=(h, w),
            max_area_percent=args.max_area_percent,
        )

        if keep_indices:
            boxes = results[0]["boxes"][keep_indices]
            scores = results[0]["scores"][keep_indices]
        else:
            boxes = results[0]["boxes"][:0]
            scores = results[0]["scores"][:0]

        input_boxes = boxes.detach().cpu().numpy()

        if input_boxes.shape[0] == 0:
            save_empty_mask(
                image_file=image_file,
                image=image,
                output_ann_dir=output_ann_dir,
            )
            continue

        masks, _, _ = sam2_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_boxes,
            multimask_output=False,
        )

        if masks.ndim == 4:
            masks = masks.squeeze(1)

        confidences = scores.detach().cpu().numpy().tolist()

        save_instance_mask(
            image_file=image_file,
            image=image,
            masks=masks,
            input_boxes=input_boxes,
            confidences=confidences,
            output_ann_dir=output_ann_dir,
        )
