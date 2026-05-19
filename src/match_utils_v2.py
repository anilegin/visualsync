import numpy as np
import torch
from scipy.optimize import linear_sum_assignment


def tensor_to_img(x):
    """
    Convert tensor image to uint8 numpy image.
    Handles normalized or unnormalized tensor roughly.
    """
    if torch.is_tensor(x):
        x = x.detach().cpu()

        if x.ndim == 4:
            x = x[0]

        if x.ndim == 3 and x.shape[0] in [1, 3, 4]:
            x = x.permute(1, 2, 0)

        x = x.numpy()

    x = np.asarray(x)

    if x.dtype != np.uint8:
        # If normalized-ish, bring into 0..255 range safely.
        x = x.astype(np.float32)
        x = x - x.min()
        denom = x.max() - x.min()
        if denom > 1e-8:
            x = x / denom
        x = (x * 255.0).clip(0, 255).astype(np.uint8)

    if x.ndim == 3 and x.shape[-1] == 1:
        x = np.repeat(x, 3, axis=-1)

    return x


def get_mask_bbox(mask, padding=0):
    """
    Return bbox around nonzero mask pixels as x1,y1,x2,y2.
    """
    mask = np.asarray(mask)
    ys, xs = np.where(mask > 0)

    if len(xs) == 0:
        H, W = mask.shape[:2]
        return 0, 0, W, H

    H, W = mask.shape[:2]
    x1 = max(0, int(xs.min()) - padding)
    y1 = max(0, int(ys.min()) - padding)
    x2 = min(W, int(xs.max()) + 1 + padding)
    y2 = min(H, int(ys.max()) + 1 + padding)

    return x1, y1, x2, y2


def filter_segmentation_ids(mask, min_pixels=100, min_frame_ratio=None):
    """
    Return nonzero segmentation ids with at least min_pixels.
    """
    mask = np.asarray(mask)
    ids = np.unique(mask)
    ids = ids[ids > 0]

    keep = []
    for sid in ids:
        if np.sum(mask == sid) >= min_pixels:
            keep.append(int(sid))

    return np.array(keep, dtype=np.int32)


def find_max_pixel_frame(masks, seg_id=None):
    """
    Find frame index with maximum valid/nonzero mask area.
    If seg_id is given, only count that id.
    """
    masks = np.asarray(masks)

    if seg_id is None:
        counts = (masks > 0).sum(axis=(1, 2))
    else:
        counts = (masks == seg_id).sum(axis=(1, 2))

    return int(np.argmax(counts))


def optimal_segmentation_matches(correspondence_list, min_matches=5):
    """
    Match segmentation ids between two views using Hungarian assignment.

    correspondence_list: array/list of [seg_id_view1, seg_id_view2].
    Returns list of tuples: [(id1, id2), ...]
    """
    correspondence_list = np.asarray(correspondence_list)

    if correspondence_list.size == 0:
        return []

    if correspondence_list.ndim == 1:
        correspondence_list = correspondence_list.reshape(-1, 2)

    # Remove background
    valid = (correspondence_list[:, 0] > 0) & (correspondence_list[:, 1] > 0)
    correspondence_list = correspondence_list[valid]

    if len(correspondence_list) == 0:
        return []

    ids1 = np.unique(correspondence_list[:, 0])
    ids2 = np.unique(correspondence_list[:, 1])

    id1_to_i = {sid: i for i, sid in enumerate(ids1)}
    id2_to_j = {sid: j for j, sid in enumerate(ids2)}

    counts = np.zeros((len(ids1), len(ids2)), dtype=np.int32)

    for sid1, sid2 in correspondence_list:
        counts[id1_to_i[sid1], id2_to_j[sid2]] += 1

    # Hungarian minimizes, so use negative counts.
    row_ind, col_ind = linear_sum_assignment(-counts)

    pairs = []
    for r, c in zip(row_ind, col_ind):
        if counts[r, c] >= min_matches:
            pairs.append((int(ids1[r]), int(ids2[c])))

    return pairs


def post_process_matches(matches, view):
    """
    Same as match_utils.post_process_matches.
    Included only because img_match_v4 imports it from here first.
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

    valid = view.get("valid", None)
    if valid is not None:
        H, W = valid.shape[:2]
        pts_orig[:, 0] = np.clip(pts_orig[:, 0], 0, W - 1)
        pts_orig[:, 1] = np.clip(pts_orig[:, 1], 0, H - 1)

    return pts_orig.astype(np.float32)
