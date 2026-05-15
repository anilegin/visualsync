import numpy as np
import torch
import PIL.Image
import torchvision.transforms as tvf
import torchvision.transforms.functional as F


def get_HW_resolution(image):
    """
    Return image height and width.

    Accepts:
      - numpy image H,W,C
      - PIL image
      - torch tensor C,H,W or H,W,C
    """
    if isinstance(image, PIL.Image.Image):
        W, H = image.size
        return H, W

    if isinstance(image, np.ndarray):
        return image.shape[0], image.shape[1]

    if torch.is_tensor(image):
        if image.ndim == 3:
            # C,H,W usually
            if image.shape[0] in [1, 3, 4]:
                return int(image.shape[1]), int(image.shape[2])
            return int(image.shape[0]), int(image.shape[1])
        if image.ndim == 2:
            return int(image.shape[0]), int(image.shape[1])

    raise TypeError(f"Unsupported image type: {type(image)}")


def _to_pil_rgb(image):
    """
    Convert input frame to PIL RGB.
    """
    if isinstance(image, PIL.Image.Image):
        return image.convert("RGB")

    if isinstance(image, np.ndarray):
        arr = image
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)

        if arr.ndim == 2:
            return PIL.Image.fromarray(arr).convert("RGB")

        if arr.shape[-1] == 4:
            arr = arr[..., :3]

        return PIL.Image.fromarray(arr).convert("RGB")

    if torch.is_tensor(image):
        img = image.detach().cpu()
        if img.ndim == 3 and img.shape[0] in [1, 3, 4]:
            img = img.permute(1, 2, 0)
        img = img.numpy()
        if img.max() <= 1.0:
            img = img * 255.0
        img = np.clip(img, 0, 255).astype(np.uint8)
        return PIL.Image.fromarray(img).convert("RGB")

    raise TypeError(f"Unsupported image type: {type(image)}")


def _resize_long_side_keep_aspect(W, H, maxdim, patch_size):
    """
    Resize so max side <= maxdim and both sides are divisible by patch_size.
    This keeps MASt3R/DUSt3R descriptor grid stable.
    """
    scale = min(float(maxdim) / max(W, H), 1.0)

    new_W = int(round(W * scale))
    new_H = int(round(H * scale))

    # make dimensions divisible by patch size
    new_W = max(patch_size, (new_W // patch_size) * patch_size)
    new_H = max(patch_size, (new_H // patch_size) * patch_size)

    return new_W, new_H


def _resize_mask(mask, size_wh):
    """
    Resize mask with nearest neighbor.
    size_wh = (W, H)
    """
    if mask is None:
        return None

    mask_arr = np.asarray(mask)
    mask_pil = PIL.Image.fromarray(mask_arr.astype(np.int32), mode="I")
    mask_resized = mask_pil.resize(size_wh, resample=PIL.Image.NEAREST)
    return np.array(mask_resized)


def process_image(
    image,
    maxdim=512,
    patch_size=16,
    load_mask=False,
    mask=None,
):
    """
    Minimal MASt3R/DUSt3R-compatible image preprocessing.

    Returns keys used by img_match_v4.py:
      rgb             : original PIL RGB image
      rgb_rescaled    : normalized tensor C,H,W
      valid           : original-resolution boolean mask H,W
      valid_rescaled  : resized boolean mask H,W
      to_orig         : 3x3 transform mapping resized coords to original coords
    """
    rgb = _to_pil_rgb(image)
    W, H = rgb.size

    new_W, new_H = _resize_long_side_keep_aspect(W, H, maxdim, patch_size)

    rgb_resized = rgb.resize((new_W, new_H), resample=PIL.Image.BICUBIC)

    # DUSt3R/MASt3R convention: ImageNet normalization
    transform = tvf.Compose([
        tvf.ToTensor(),
        tvf.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
    ])
    rgb_rescaled = transform(rgb_resized).float()

    if load_mask and mask is not None:
        valid = np.asarray(mask).astype(np.int32) > 0
    else:
        valid = np.ones((H, W), dtype=bool)

    valid_resized = _resize_mask(valid.astype(np.uint8), (new_W, new_H)) > 0

    # transform from resized image coordinates back to original image coords
    scale_x = W / float(new_W)
    scale_y = H / float(new_H)
    to_orig = np.array([
        [scale_x, 0.0, 0.0],
        [0.0, scale_y, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float32)

    return {
        "rgb": rgb,
        "rgb_rescaled": rgb_rescaled,
        "valid": valid,
        "valid_rescaled": torch.from_numpy(valid_resized).bool(),
        "to_orig": to_orig,
        "orig_shape": np.array([H, W], dtype=np.int32),
        "rescaled_shape": np.array([new_H, new_W], dtype=np.int32),
    }


def post_process_matches(matches, view):
    """
    Map MASt3R/DUSt3R match coordinates from resized image coordinates
    back to original image coordinates.

    Input:
      matches: N,2 array in x,y order
      view['to_orig']: 3x3 affine matrix

    Output:
      N,2 numpy array in original image x,y coordinates
    """
    if matches is None:
        return np.zeros((0, 2), dtype=np.float32)

    matches = np.asarray(matches, dtype=np.float32)

    if matches.size == 0:
        return matches.reshape(0, 2)

    if matches.ndim == 1:
        matches = matches.reshape(-1, 2)

    to_orig = view.get("to_orig", None)
    if to_orig is None:
        return matches

    ones = np.ones((matches.shape[0], 1), dtype=np.float32)
    pts_h = np.concatenate([matches, ones], axis=1)
    pts_orig = pts_h @ to_orig.T
    pts_orig = pts_orig[:, :2]

    H, W = view["valid"].shape[:2]
    pts_orig[:, 0] = np.clip(pts_orig[:, 0], 0, W - 1)
    pts_orig[:, 1] = np.clip(pts_orig[:, 1], 0, H - 1)

    return pts_orig.astype(np.float32)
