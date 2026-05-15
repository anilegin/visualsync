
import os
import pickle
import cv2
import torch
import torch.nn.functional as F
import argparse
import imageio
import imageio.v3 as iio
import numpy as np
from tqdm import tqdm
from scipy.ndimage import label
from cotracker.predictor import CoTrackerPredictor


# compare to run_cotracker_v4.py, not using a single frame for mask initialization, uniform 10 frames sample pts
# not downsample the video, but use grid_size=5 to sample pts within mask of each object, also add remove_redundant_query function
# also store visibility and validity (consider within the mask and visibility), only visualize valid points in the video

from cotracker.utils.visualizer import Visualizer
"""
reference: https://github.com/Stereo4d/stereo4d-code/blob/main/tracking.py 
https://github.com/Davidyao99/uni4d/blob/main/preprocess/run_cotracker.py
"""


def filter_valid_tracklets(tracklets, mask):
    """
    Checks which tracklet positions are valid based on:
    1. Being within image bounds.
    2. Falling on valid (nonzero) pixels in the provided mask.

    Args:
        tracklets (torch.Tensor): Track positions of shape (T, N, 2), where each (x, y) is a track position.
        mask (torch.Tensor): Validity mask of shape (T, H, W), where nonzero values indicate valid pixels.

    Returns:
        torch.Tensor: A boolean mask of shape (T, N, 2), indicating valid x/y positions for each point.
    """
    T, N, _ = tracklets.shape
    H, W = mask.shape[1], mask.shape[2]

    # Step 1: Check for in-bound positions
    x = tracklets[..., 0]
    y = tracklets[..., 1]

    x_valid = (x >= 0) & (x < W)
    y_valid = (y >= 0) & (y < H)
    in_bounds = x_valid & y_valid  # Shape: (T, N)

    # Step 2: Convert (x, y) to long for indexing
    x_idx = x.clamp(0, W - 1).long()
    y_idx = y.clamp(0, H - 1).long()

    # Step 3: Use the mask to check per-point validity
    t_idx = torch.arange(T, device=tracklets.device).view(-1, 1).expand_as(x_idx)
    point_valid = mask[t_idx, y_idx, x_idx] > 0  # Shape: (T, N)

    # Step 4: Combine with bounds check
    valid_mask = in_bounds & point_valid  # Shape: (T, N)

    return valid_mask


def blurry_frame_detect(videos, threshold):
    # videos: list of video frames (T, H, W, C)
    is_blurry_list  = []
    for idx in range(len(videos)):
        frame = videos[idx]         
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        # Compute Laplacian and its variance
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        if laplacian_var < threshold:
            is_blurry_list.append(True)
        else:
            is_blurry_list.append(False)
    is_blurry_list = np.array(is_blurry_list) # bool (T,)
    return is_blurry_list


def read_file_from_dir(path, read_mask=False):
    frames = []

    # Get all files in the directory
    all_files = os.listdir(path)

    # Organize into a dict with base names
    file_map = {}
    for f in all_files:
        base, ext = os.path.splitext(f)
        ext = ext.lower()
        if ext in ['.png', '.jpg', '.npy']:
            if base not in file_map:
                file_map[base] = {}
            file_map[base][ext] = f

    # Sort by base name to ensure deterministic order
    for base in sorted(file_map.keys()):
        exts = file_map[base]

        if '.npy' in exts:
            res = np.load(os.path.join(path, exts['.npy']))

        elif '.png' in exts:
            if read_mask:
                res = imageio.imread(os.path.join(path, exts['.png']), mode='F')
            else:
                res = imageio.imread(os.path.join(path, exts['.png']))

        elif '.jpg' in exts:
            if read_mask:
                res = imageio.imread(os.path.join(path, exts['.jpg']), mode='F')
            else:
                res = imageio.imread(os.path.join(path, exts['.jpg']))

        else:
            continue  # Unexpected file type

        frames.append(res)

    return frames


def downsample_video_and_tracks(
    video: torch.Tensor, 
    tracks: torch.Tensor, 
    tracks_valid: torch.Tensor,
    query_frames: torch.Tensor = None,
    target_size: int = 1920,
    max_track_points: int = None
):
    """
    Downsample the video tensor (B, T, C, H, W) so that the maximum dimension is target_size,
    preserving aspect ratio. Also downsample the track coordinates (B, T, N, 2) and the corresponding
    track validity (B, T, N) by the same scale. Additionally, if the number of track points exceeds 
    max_track_points, randomly downsample the points (and corresponding validity flags and query frames).
    
    Args:
        video (torch.Tensor): Video tensor of shape (B, T, C, H, W)
        tracks (torch.Tensor): Track coordinates of shape (B, T, N, 2)
        tracks_valid (torch.Tensor): Boolean tensor of shape (B, T, N) indicating track validity
        query_frames (torch.Tensor, optional): Tensor of shape (N,) representing a query frame for each track.
        target_size (int): Target maximum dimension size.
        max_track_points (int, optional): Maximum number of track points; if None, no downsampling of points.
        
    Returns:
        tuple: 
          If query_frames is None:
            (video_downsampled, tracks_downsampled, tracks_valid_downsampled)
          Else:
            (video_downsampled, tracks_downsampled, tracks_valid_downsampled, query_frames_downsampled)
          
          video_downsampled: Tensor of shape (B, T, C, new_H, new_W)
          tracks_downsampled: Tensor of shape (B, T, new_N, 2)
          tracks_valid_downsampled: Tensor of shape (B, T, new_N)
          query_frames_downsampled: Tensor of shape (new_N,)
    """
    B, T, C, H, W = video.shape

    # Calculate scaling factor.
    scale_factor = min(1.0, target_size / max(H, W))
    new_h = int(H * scale_factor)
    new_w = int(W * scale_factor)

    # Downsample video using interpolation.
    video_reshaped = video.view(B * T, C, H, W)
    video_resized = F.interpolate(video_reshaped, size=(new_h, new_w), mode='bilinear', align_corners=False)
    video_downsampled = video_resized.view(B, T, C, new_h, new_w)

    # Downsample tracks: adjust xy coordinates by the same scale factor.
    tracks_downsampled = tracks * scale_factor

    # If point downsampling is required and there are more points than allowed:
    if max_track_points is not None:
        _, _, N, _ = tracks.shape
        if N > max_track_points:
            # Choose a consistent set of random indices for all frames and tracks.
            indices = torch.randperm(N, device=tracks.device)[:max_track_points]
            tracks_downsampled = tracks_downsampled.index_select(dim=2, index=indices)
            tracks_valid = tracks_valid.index_select(dim=2, index=indices) # (B, T, N)
            if query_frames is not None:
                query_frames = query_frames.index_select(dim=0, index=indices) # (N, )

    # Return query_frames in the output tuple if provided.
    if query_frames is None:
        return video_downsampled, tracks_downsampled, tracks_valid
    else:
        return video_downsampled, tracks_downsampled, tracks_valid, query_frames


def sample_grid_points(
    frame_idx: int,
    height: int,
    width: int,
    stride: int = 1,
    jitter_on=False,
    device=torch.device("cpu")
):
    """Sample grid points with (time, height, width) order in PyTorch.
    
    Args:
        frame_idx (int): Frame index to use for all points
        height (int): Height of the frame
        width (int): Width of the frame
        stride (int, optional): Spacing between grid points. Defaults to 1.
        jitter_on (bool, optional): Whether to add random jitter to points. Defaults to False.
        device (torch.device, optional): Device to create tensors on. Defaults to CPU.
        
    Returns:
        torch.Tensor: Grid points of shape [out_height*out_width, 3] in (t, x, y) order
    """
    # Create ranges for grid
    y_range = torch.arange(stride // 2, height, stride, device=device)
    x_range = torch.arange(stride // 2, width, stride, device=device)
    
    # Create meshgrid (equivalent to np.mgrid)
    grid_y, grid_x = torch.meshgrid(y_range, x_range, indexing="ij")
    grid_y, grid_x = grid_y.float(), grid_x.float()
    
    # Apply jitter if enabled
    if stride > 1 and jitter_on:
        jitter_y = torch.rand_like(grid_y) * stride - stride // 2
        jitter_x = torch.rand_like(grid_x) * stride - stride // 2
        
        grid_y = grid_y + jitter_y
        grid_x = grid_x + jitter_x
        
        # Clip to valid range
        grid_y = torch.clamp(grid_y, 0, height - 1)
        grid_x = torch.clamp(grid_x, 0, width - 1)
    
    # Get output dimensions
    out_height, out_width = grid_y.shape
    
    # Stack x, y coordinates
    points_xy = torch.stack([grid_x, grid_y], dim=-1)
    
    # Create frame index tensor
    frame_idx_tensor = torch.full((out_height, out_width, 1), frame_idx, 
                                  dtype=torch.int32, device=device)
    
    # Concatenate frame index with coordinates
    points = torch.cat([frame_idx_tensor, points_xy], dim=-1)
    
    # Reshape to [out_height*out_width, 3]
    points = points.reshape(-1, 3)
    
    return points.to(dtype=torch.int32)


def get_min_distance(query_points, other_points):
    """
    Compute minimum distances from query_points to other_points using PyTorch.
    
    Args:
        query_points: Tensor of shape [n_query, 2] containing (x,y) coordinates
        other_points: Tensor of shape [n_other, 2] containing (x,y) coordinates
        
    Returns:
        Tensor of shape [n_query] containing minimum distances
    """
    # Compute pairwise distances between query_points and other_points
    # Using broadcasting to avoid excessive memory usage
    n_query = query_points.shape[0]
    n_other = other_points.shape[0]
    
    # If there are too many points, process in batches to avoid OOM
    if n_query * n_other > 10000000:  # This threshold can be adjusted based on GPU memory
        batch_size = max(1, 10000000 // n_other)
        min_distances = torch.zeros(n_query, device=query_points.device)
        
        for i in range(0, n_query, batch_size):
            end_idx = min(i + batch_size, n_query)
            batch_query = query_points[i:end_idx]
            
            # Compute squared Euclidean distances
            # reshape to [batch_size, 1, 2] and [1, n_other, 2] for broadcasting
            distances = torch.sum(
                (batch_query.unsqueeze(1) - other_points.unsqueeze(0)) ** 2, 
                dim=2
            )
            
            # Get minimum distance for each query point
            min_distances[i:end_idx] = torch.sqrt(torch.min(distances, dim=1)[0])
        
        return min_distances
    else:
        # Faster implementation for smaller point sets
        # reshape to [n_query, 1, 2] and [1, n_other, 2] for broadcasting
        distances = torch.sum(
            (query_points.unsqueeze(1) - other_points.unsqueeze(0)) ** 2, 
            dim=2
        )
        
        # Get minimum distance for each query point
        return torch.sqrt(torch.min(distances, dim=1)[0])


def remove_redundant_query(tracks, visibility, query_points, threshold):
    """
    Remove redundant query points that are too close to existing tracks.
    
    Args:
        tracks: Tensor of shape [nframe, npt, 2] with (x,y) coordinates
        visibility: Tensor of shape [nframe, npt] with visibility flags
        query_points: Tensor of shape [npt, 3] with (t, y, x) coordinates
        threshold: Float distance threshold in pixel space
        
    Returns:
        Tuple of (filtered_tracks, filtered_visibility, filtered_query_points)
    """
    # Convert to PyTorch tensors if needed
    if not isinstance(tracks, torch.Tensor):
        tracks = torch.tensor(tracks, device=query_points.device)
    if not isinstance(visibility, torch.Tensor):
        visibility = torch.tensor(visibility, device=query_points.device)
    
    nframe, npt, _ = tracks.shape
    
    # Initialize mask for good points
    good_points = torch.zeros(npt, dtype=torch.bool, device=query_points.device)
    
    # Get unique frame IDs
    frame_ids = torch.unique(query_points[:, 0])
    
    for fid in frame_ids:
        # Get query points for this frame
        frame_mask = query_points[:, 0] == fid
        query_at_fid = query_points[frame_mask][:, 1:]
        # Flip y,x to x,y for distance computation
        query_at_fid = query_at_fid.flip(dims=[1])
        query_pt_id = torch.nonzero(frame_mask, as_tuple=True)[0]
        
        # Get tracked points for this frame
        fid_int = int(fid.item())
        if fid_int >= nframe:
            # If frame ID is out of bounds, mark all query points in this frame as good
            good_points[query_pt_id] = True
            continue
        
        # Get tracked points already marked as good in this frame
        tracked_good = tracks[fid_int][good_points]  # [n_good, 2]
        tracked_visibility_good = visibility[fid_int][good_points]  # [n_good]
        
        if len(tracked_good) == 0:
            # If no good points yet, mark all query points in this frame as good
            good_points[query_pt_id] = True
            continue
        
        # Filter to only visible tracked points
        tracked_visible_mask = tracked_visibility_good > 0
        tracked_at_fid = tracked_good[tracked_visible_mask]
        
        if len(tracked_at_fid) == 0:
            # If no visible tracked points, mark all query points as good
            good_points[query_pt_id] = True
            continue
        
        # Compute minimum distances
        min_distance = get_min_distance(query_at_fid, tracked_at_fid)
        
        # Mark points with distance > threshold as good
        good_points_mask = min_distance > threshold
        good_points[query_pt_id[good_points_mask]] = True
    
    # Print removal statistics
    good_points_sum = good_points.sum().item()
    
    # Safe calculation of removal percentage with error handling
    if len(good_points) > 0:
        removal_percentage = (1 - good_points_sum / len(good_points)) * 100
        print(f'Removed {removal_percentage:.2f}% redundant points')
    else:
        print('No points to process')
    
    # Filter query points
    filtered_query_points = query_points[good_points]
    
    # Filter tracks and visibility
    filtered_tracks = tracks[:, good_points, :]
    filtered_visibility = visibility[:, good_points]
    
    return filtered_tracks, filtered_visibility, filtered_query_points


def sample_frames_with_max_valids(mask: np.ndarray, interval: int, blurry_list: np.ndarray = None) -> list:
    """
    Sample 1 frame from every 'interval' frames, choosing the frame with the maximum
    number of valid pixels in each interval. Optionally, if a blurry_list is provided, 
    frames marked as blurry will be skipped (and if all frames in an interval are blurry, 
    that interval is skipped).

    Parameters:
    -----------
    mask : np.ndarray
        Array of shape (T, H, W) where T is the number of frames.
        Each element is 0 (invalid) or 1 (valid).
    interval : int
        Number of frames in each interval to sample from.
    blurry_list : np.ndarray, optional
        Boolean array of shape (T,) where True indicates that the frame is blurry.
        If None, frames are selected solely based on the valid pixel count.

    Returns:
    --------
    list of int
        Indices of the selected frames from the original array.
    """
    T, H, W = mask.shape
    selected_indices = []
    
    # Compute the sum of valid pixels for each frame.
    valid_sums = mask.sum((1, 2))  # Shape: (T,)
    
    # Calculate the number of intervals (ceiling division)
    num_intervals = (T + interval - 1) // interval
    
    for i in range(num_intervals):
        start_idx = i * interval
        end_idx = min((i + 1) * interval, T)
        
        # Get valid sums for the current interval.
        interval_valid_sums = valid_sums[start_idx:end_idx]
        
        if blurry_list is not None:
            # Get blurry flags for this interval.
            interval_blurry = blurry_list[start_idx:end_idx]
            # Find indices of frames that are not blurry.
            non_blurry_indices = np.where(~interval_blurry)[0]
            
            if non_blurry_indices.size > 0:
                # Among non-blurry frames, choose the one with the maximum valid sum.
                best_rel_idx = non_blurry_indices[np.argmax(interval_valid_sums[non_blurry_indices])]
                selected_indices.append(start_idx + best_rel_idx)
            # If all frames in the interval are blurry, skip this interval.
        else:
            # No blurry_list provided; choose frame with maximum valid sum.
            best_rel_idx = np.argmax(interval_valid_sums)
            selected_indices.append(start_idx + best_rel_idx)
    
    return selected_indices



# def filter_segmentation_masks(masks_list, is_cam_static):
#     """
#     Filter segmentation classes from a video segmentation mask list
#     with hard-coded thresholds for filtering.
    
#     Args:
#         masks_list: numpy array of segmentation masks
#         is_cam_static: boolean indicating if camera is static
        
#     Returns:
#         unique_ids: numpy array of filtered segmentation IDs
#     """
#     # Convert masks list to int32 type
#     masks_list = np.array(masks_list).astype(np.int32)
    
#     # Get unique segmentation IDs (excluding background class 0)
#     unique_ids = np.unique(masks_list)
#     unique_ids = unique_ids[unique_ids > 0]
#     print("number of unique ids: ", len(unique_ids))
    
#     # Store original unique IDs to find max pixel count if needed
#     original_unique_ids = unique_ids.copy()
    
#     # Get video dimensions
#     T = masks_list.shape[0]  # Number of frames
    
#     # Determine if filtering should be applied
#     apply_filter = len(unique_ids) > 4
    
#     # Set thresholds based on camera state
#     if not is_cam_static:
#         min_frame_ratio = 0.3
#         min_pixel_count = 100
#     else:
#         min_frame_ratio = 0.5
#         min_pixel_count = 500
    
#     # Maximum allowed components in a frame
#     max_allowed_components = 1
    
#     # Dictionary to store pixel counts for each segmentation ID
#     id_pixel_counts = {}
    
#     # Filter segments
#     for seg_id in unique_ids:
#         if seg_id == 0:
#             continue
            
#         # Create binary mask for current class
#         binary_mask = (masks_list == seg_id)
        
#         # Calculate frame presence
#         class_presence = np.any(np.any(binary_mask, axis=2), axis=1)
#         appear_frames = np.sum(class_presence)
        
#         # Calculate ratio of frames containing this class
#         frame_ratio = appear_frames / T
        
#         # Calculate pixel counts and average
#         pixel_counts = np.sum(np.sum(binary_mask, axis=2), axis=1)
#         avg_pixels = np.mean(pixel_counts[class_presence]) if appear_frames > 0 else 0
        
#         # Store total pixel count for this ID
#         id_pixel_counts[seg_id] = np.sum(pixel_counts)
        
#         frames_with_multiple = 0
#         for t in range(T):
#             if class_presence[t]:
#                 # Get mask for current frame and label connected components
#                 frame_mask = binary_mask[t]
#                 labeled_array, num_features = label(frame_mask)
                
#                 # Count frames with too many components
#                 if num_features > max_allowed_components:
#                     frames_with_multiple += 1
        
#         # Determine if we should filter based on component criteria
#         filter_by_components = (frames_with_multiple > 0.3 * appear_frames)
        
#         if apply_filter:
#             # Filter based on conditions
#             if avg_pixels < min_pixel_count or frame_ratio < min_frame_ratio or filter_by_components:
#                 masks_list[masks_list == seg_id] = 0
    
#     # Recalculate unique IDs after filtering
#     unique_ids = np.unique(masks_list)
#     unique_ids = unique_ids[unique_ids > 0]
#     print("after filter, number of unique ids: ", len(unique_ids), unique_ids)
    
#     # If no IDs remain after filtering, return the one with max pixel count
#     if len(unique_ids) == 0 and len(original_unique_ids) > 0:
#         # Find segmentation ID with maximum pixel count
#         max_pixel_id = max(id_pixel_counts.items(), key=lambda x: x[1])[0]
#         print(f"No segments passed filtering. Returning ID {max_pixel_id} with maximum pixel count.")
#         return np.array([max_pixel_id])
    
#     return unique_ids


def filter_segmentation_masks(masks_list, is_cam_static):
    """
    Filter segmentation classes from a video segmentation mask list
    with hard-coded thresholds for filtering.
    
    Args:
        masks_list: numpy array of segmentation masks
        is_cam_static: boolean indicating if camera is static
        
    Returns:
        unique_ids: numpy array of filtered segmentation IDs
    """
    # Convert masks list to int32 type
    masks_list = np.array(masks_list).astype(np.int32)
    
    # Get unique segmentation IDs (excluding background class 0)
    unique_ids = np.unique(masks_list)
    unique_ids = unique_ids[unique_ids > 0]
    print("number of unique ids: ", len(unique_ids))
    
    # Store original unique IDs to find max pixel count if needed
    original_unique_ids = unique_ids.copy()
    
    # Get video dimensions
    T = masks_list.shape[0]  # Number of frames
    H, W = masks_list.shape[1], masks_list.shape[2]  # Height, Width
    
    # Calculate total pixels per frame
    total_pixels_per_frame = H * W
    
    # Determine if filtering should be applied
    apply_filter = len(unique_ids) > 4
    
    # Set thresholds based on camera state
    if not is_cam_static:
        min_frame_ratio = 0.3
        min_pixel_count = 100
    else:
        min_frame_ratio = 0.5
        min_pixel_count = 500
    
    # Maximum allowed components in a frame
    max_allowed_components = 1
    
    # Dictionary to store pixel counts for each segmentation ID
    id_pixel_counts = {}
    
    # Filter segments
    for seg_id in unique_ids:
        if seg_id == 0:
            continue
        
        # Create binary mask for current class
        binary_mask = (masks_list == seg_id)
        
        # Calculate frame presence
        class_presence = np.any(np.any(binary_mask, axis=2), axis=1)
        appear_frames = np.sum(class_presence)
        
        # Calculate ratio of frames containing this class
        frame_ratio = appear_frames / T
        
        # Calculate pixel counts and average
        pixel_counts = np.sum(np.sum(binary_mask, axis=2), axis=1)
        avg_pixels = np.mean(pixel_counts[class_presence]) if appear_frames > 0 else 0
        
        # Store total pixel count for this ID
        id_pixel_counts[seg_id] = np.sum(pixel_counts)
        
        # Check if any frame has this segmentation ID covering more than half the image
        half_image_threshold = total_pixels_per_frame * 0.5
        covers_too_much = np.any(pixel_counts > half_image_threshold)
        
        frames_with_multiple = 0
        for t in range(T):
            if class_presence[t]:
                # Get mask for current frame and label connected components
                frame_mask = binary_mask[t]
                labeled_array, num_features = label(frame_mask)
                
                # Count frames with too many components
                if num_features > max_allowed_components:
                    frames_with_multiple += 1
        
        # Determine if we should filter based on component criteria
        filter_by_components = (frames_with_multiple > 0.8 * appear_frames) # old 0.3 (when apply in-the-wild set smaller)
        
        if apply_filter:
            # Filter based on conditions
            if avg_pixels < min_pixel_count or frame_ratio < min_frame_ratio or filter_by_components or covers_too_much:
                masks_list[masks_list == seg_id] = 0
    
    # Recalculate unique IDs after filtering
    unique_ids = np.unique(masks_list)
    unique_ids = unique_ids[unique_ids > 0]
    print("after filter, number of unique ids: ", len(unique_ids), unique_ids)
    
    # If no IDs remain after filtering, return the one with max pixel count
    if len(unique_ids) == 0 and len(original_unique_ids) > 0:
        # Find segmentation ID with maximum pixel count
        max_pixel_id = max(id_pixel_counts.items(), key=lambda x: x[1])[0]
        print(f"No segments passed filtering. Returning ID {max_pixel_id} with maximum pixel count.")
        return np.array([max_pixel_id])
    
    return unique_ids

def is_camera_static(extrinsics, threshold=1e-6):
    """
    Determines if a camera is static by checking if all relative transformations 
    between consecutive frames are close to the identity matrix.

    Args:
        extrinsics (np.ndarray): Array of shape (T, 4, 4) containing camera extrinsics.
        threshold (float): Tolerance for determining if a transformation is close to identity.

    Returns:
        bool: True if the camera is static, False otherwise.
    """
    T = extrinsics.shape[0]
    
    # Compute relative transformations between consecutive frames
    for t in range(T - 1):
        rel_transform = np.linalg.inv(extrinsics[t]) @ extrinsics[t + 1]
        
        # Check if the relative transformation is close to the identity matrix
        if not np.allclose(rel_transform, np.eye(4), atol=threshold):
            return False  # If any relative transform is not identity, it's dynamic
    
    return True  # If all relative transforms are identity, it's static


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_dir", type=str, default=None, help="input video dir")
    parser.add_argument("--mask_dir", type=str, default=None, help="input mask dir")
    parser.add_argument("--save_dir", type=str, default="results", help="save directory")
    
    parser.add_argument("--checkpoint", default=None, help="cotracker model")
    parser.add_argument("--grid_step", type=int, default=5, help="grid step size between grid samples")
    parser.add_argument('--interval', help='interval for cotracker', default=10, type=int)
    parser.add_argument("--max_query_per_batch", type=int, default=1000, help="max query per batch")
    
    parser.add_argument("--blur_threshold", type=float, default=20.0, help="blurry frame threshold")
    parser.add_argument("--pixel_threshold", default=5.0, type=float, help="redundant pixel filtering threshold")
    parser.add_argument("--disable_blurry", action="store_true", help="disable blurry frame detection")
    
    parser.add_argument("--disable_seg_filter", action="store_true", help="apply segmentation filter")
    
    parser.add_argument("--vis_size", type=int, default=1920, help="visualization size")
    args = parser.parse_args()
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    if args.checkpoint is not None:
        model = CoTrackerPredictor(checkpoint=args.checkpoint, offline=True)
    else:
        model = torch.hub.load("/home/vrai/anilegin/visualsync/co-tracker", "cotracker3_offline", source="local")
    
    device = (
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    )

    model = model.to(device)
    video_dir = args.video_dir.rstrip('/')
    mask_dir = args.mask_dir
    video_name = os.path.basename(video_dir)
    
    video_list = read_file_from_dir(video_dir)
    masks_list = read_file_from_dir(mask_dir, read_mask=True)
    assert len(video_list) == len(masks_list)
    T = len(video_list)
    assert video_list[0].shape[:2] == masks_list[0].shape[:2] # H, W
    H, W = video_list[0].shape[:2]
    
    videos = np.array(video_list)
    if args.disable_blurry:
        is_blurry_list = np.array([False] * T, dtype=bool) # T
    else:
        is_blurry_list = blurry_frame_detect(videos, threshold=args.blur_threshold)
    
    masks = np.array(masks_list).astype(np.int32)
    if args.disable_seg_filter:
        unique_ids = np.unique(masks)
        unique_ids = unique_ids[unique_ids > 0] # Exclude background
        N = len(unique_ids)
    else:
        try:
            cam_extrinsic_path = os.path.join(video_dir, "..", "w2c.npy")
            cam_extrinsic = np.load(cam_extrinsic_path) # (T, 4, 4)
            is_cam_static = is_camera_static(cam_extrinsic, threshold=1e-3)
        except:
            is_cam_static = False
        unique_ids = filter_segmentation_masks(masks, is_cam_static=is_cam_static)
    
    video_input = torch.tensor(videos).permute(0, 3, 1, 2).float().to(device)  # T C H W
    masks_input = torch.tensor(masks).long() # T, H, W
    video_input = video_input.unsqueeze(0) # B T C H W
    
    vis = Visualizer(save_dir=args.save_dir, pad_value=120, linewidth=3)
    
    pred_tracks_all = []
    pred_visibility_all = []
    seg_ids_all = []
    query_frames_all = [] # each point stores the query frame
    pred_valid_all = []
    for seg_id in unique_ids:
        seg_mask = (masks_input == seg_id) # T H W
        T, height, width = masks_input.shape
        query_points = []
        # check is_blurry_list
        valid_ratio = is_blurry_list.sum() / T
        if valid_ratio > 0.8: # too many blurry frames, disable blurry detection
            is_blurry_list = np.array([False] * T, dtype=bool) # T
        sel_frames = sample_frames_with_max_valids(seg_mask, args.interval, is_blurry_list)
        for frame_id in sel_frames:
            query_points_tmp = sample_grid_points(frame_id, height, width, args.grid_step, jitter_on=True) # (-1, 3) # txy
            frame_mask = seg_mask[frame_id] # H W
            point_mask = frame_mask[(query_points_tmp[:, 2]).round().long(), 
                                    (query_points_tmp[:, 1]).round().long()].bool()
            query_points_tmp = query_points_tmp[point_mask, :]
            
            query_points.append(query_points_tmp)
        
        if len(query_points) == 0:
            continue
        
        queries = torch.cat(query_points, dim=0).float().to(device) # (M, 3)
        queries_input = queries[None] # 1 M 3
        
        max_query_per_batch = args.max_query_per_batch
        N_queries = queries_input.shape[1]
        if N_queries > max_query_per_batch:
            pred_tracks_list = []
            pred_visibility_list = []
            for i in tqdm(range(0, N_queries, max_query_per_batch)):
                query_sub = queries_input[:, i:i+max_query_per_batch]
                pred_tracks, pred_visibility = model(video_input, queries=query_sub, backward_tracking=False)
                pred_tracks_list.append(pred_tracks)
                pred_visibility_list.append(pred_visibility)
            pred_tracks = torch.cat(pred_tracks_list, dim=2)    
            pred_visibility = torch.cat(pred_visibility_list, dim=2)                
        else:
            pred_tracks, pred_visibility = model(video_input, queries=queries_input, backward_tracking=False) # B T N 2,  B T N 1
        
        pred_tracks = pred_tracks.squeeze(0) # T N 2
        pred_visibility = pred_visibility.squeeze(0) # T N
        
        assert pred_tracks.shape[0] == seg_mask.shape[0]
        
        filter_pred_tracks, filter_visibility, filter_queris = remove_redundant_query(pred_tracks, pred_visibility, queries, threshold=args.pixel_threshold)
        
        valid_mask = filter_valid_tracklets(filter_pred_tracks, seg_mask.to(filter_pred_tracks.device)) # (T, N)
        valid_mask = valid_mask & filter_visibility # no squeeze()
        pred_valid_all.append(valid_mask) # (T, N)
        
        seg_ids = torch.ones(filter_pred_tracks.shape[1], device=filter_pred_tracks.device) * seg_id
        query_frames = filter_queris[:, 0] # (N, )
        
        assert filter_pred_tracks.shape[1] == valid_mask.shape[1]
           
        pred_tracks_all.append(filter_pred_tracks) # T N 2
        pred_visibility_all.append(filter_visibility) # T N
        seg_ids_all.append(seg_ids) # N
        query_frames_all.append(query_frames) # N
        
    if len(pred_tracks_all) > 0:
        pred_tracks = torch.cat(pred_tracks_all, dim=1) # T N_all 2
        pred_visibility = torch.cat(pred_visibility_all, dim=1) # T N_all
        pred_valid = torch.cat(pred_valid_all, dim=1) # T N_all
        seg_ids_all = torch.cat(seg_ids_all, dim=0) # N_all
        query_frames = torch.cat(query_frames_all, dim=0) # N_all
        assert pred_tracks.shape[1] == seg_ids_all.shape[0]
        assert seg_ids_all.shape[0] == pred_tracks.shape[1]
        assert pred_tracks.shape[1] == query_frames.shape[0]
        print("num tracks: ", pred_tracks.shape[1])
        
        is_blurry_list = torch.tensor(is_blurry_list, device=pred_tracks.device) # T
        pred_valid[is_blurry_list] = False # set blurry frames to invalid
        
        result_dict = {"pred_tracks": pred_tracks.cpu().numpy(), # T N 2, 
                        "pred_visibilities": pred_visibility.cpu().numpy(), 
                        "pred_valid": pred_valid.cpu().numpy(),
                        "seg_ids": seg_ids_all.cpu().numpy(),
                        "query_frames": query_frames.cpu().numpy(),
                        }
        
        with open(os.path.join(args.save_dir, f"tracks.pkl"), "wb") as f:
            pickle.dump(result_dict, f)
            
            
        H, W = video_input.shape[-2:]
        vis_size = args.vis_size
        if max(H, W) > vis_size:
            video_vis, pred_tracks_vis, pred_valid_vis, query_frames_vis = downsample_video_and_tracks(video_input, pred_tracks[None], pred_valid[None], query_frames, target_size=vis_size, max_track_points=5000)
            query_frames_vis = query_frames_vis.long()
            print(f"downsample video and tracks to {vis_size} for visualization")
        else:
            video_vis = video_input
            pred_tracks_vis = pred_tracks[None]
            pred_valid_vis = pred_valid[None]
            query_frames_vis = query_frames.long()
        vis.visualize(
            video_vis,
            pred_tracks_vis,
            pred_valid_vis,
            query_frame=query_frames_vis.cpu(),
            save_video=True,
            filename=f"tracks_{video_name}"
        )
    print("Done!")
