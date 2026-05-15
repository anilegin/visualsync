
import os
import pickle
import cv2
import torch
from matplotlib import cm
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import glob
import argparse
import imageio
import imageio.v3 as iio
import numpy as np
from tqdm import tqdm
from scipy.optimize import linear_sum_assignment
from collections import defaultdict
from typing import List, Tuple, Dict, Union, Optional
from scipy.spatial.distance import cdist
from scipy.spatial import KDTree

# later combine with image_match_v4.py 
# reference: co-tracker/filter_corr_v1.py, match_utils_v2.py:optimal_segmentation_matches

from cotracker.utils.visualizer import Visualizer


def read_file_from_dir(path, read_mask=False):
    """
    Read files from a directory and return the loaded frames and their corresponding file names.

    The function searches for files with extensions .png, .jpg, or .npy, groups them by base name,
    and then loads them in sorted order by base name. If read_mask is True, the image is read in float mode.
    
    Parameters:
    -----------
    path : str
        Path to the directory containing the files.
    read_mask : bool, optional
        If True, images are read with mode 'F' (float). Default is False.
        
    Returns:
    --------
    tuple:
        frames : list
            A list of loaded frames (images or numpy arrays).
        filenames : list
            A list of file names corresponding to each loaded frame.
    """
    frames = []
    filenames = []

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
        file_to_read = None
        
        if '.npy' in exts:
            file_to_read = exts['.npy']
            res = np.load(os.path.join(path, file_to_read))
        elif '.png' in exts:
            file_to_read = exts['.png']
            if read_mask:
                res = imageio.imread(os.path.join(path, file_to_read), mode='F')
            else:
                res = imageio.imread(os.path.join(path, file_to_read))
        elif '.jpg' in exts:
            file_to_read = exts['.jpg']
            if read_mask:
                res = imageio.imread(os.path.join(path, file_to_read), mode='F')
            else:
                res = imageio.imread(os.path.join(path, file_to_read))
        else:
            continue  # Unexpected file type

        frames.append(res)
        filenames.append(file_to_read)

    return frames, filenames


def find_segmentation_associations(correspondence_list, min_matches=5):
    """
    Find optimal assignments between segmentation classes across two views.
    
    Parameters
    ----------
    correspondence_list : numpy.ndarray
        Array of shape (M, 2) where each row contains a pair (seg_id_view1, seg_id_view2)
        representing a correspondence between segmentation IDs in view 1 and view 2.
    min_matches : int, optional
        Minimum number of matches required to consider a segmentation ID pair valid.
        Default is 5.
    
    Returns
    -------
    list of tuples
        Each tuple (seg_id_view1, seg_id_view2) represents an associated pair of segmentation IDs.
    """
    # Count occurrences of each (view1, view2) pair
    pair_counts = defaultdict(int)
    for seg_id_view1, seg_id_view2 in correspondence_list:
        pair_counts[(seg_id_view1, seg_id_view2)] += 1
    
    # Extract unique segmentation IDs from each view
    unique_ids_view1 = np.unique(correspondence_list[:, 0])
    unique_ids_view2 = np.unique(correspondence_list[:, 1])
    
    # Create a cost matrix for the assignment problem
    # Initialize with zeros (will be filled with negative counts to convert max problem to min problem)
    cost_matrix = np.zeros((len(unique_ids_view1), len(unique_ids_view2)))
    
    # Build a mapping from ID to index for quick lookup
    view1_id_to_idx = {id_val: idx for idx, id_val in enumerate(unique_ids_view1)}
    view2_id_to_idx = {id_val: idx for idx, id_val in enumerate(unique_ids_view2)}
    
    # Fill the cost matrix with negative counts (to convert max problem to min problem)
    for (seg_id_view1, seg_id_view2), count in pair_counts.items():
        # Get the indices in our cost matrix
        idx1 = view1_id_to_idx.get(seg_id_view1)
        idx2 = view2_id_to_idx.get(seg_id_view2)
        
        if idx1 is not None and idx2 is not None:
            # Using negative count because linear_sum_assignment minimizes cost
            cost_matrix[idx1, idx2] = -count
    
    # Solve the assignment problem
    row_indices, col_indices = linear_sum_assignment(cost_matrix)
    
    # Create the optimal assignments
    optimal_assignments = []
    for row_idx, col_idx in zip(row_indices, col_indices):
        seg_id_view1 = unique_ids_view1[row_idx]
        seg_id_view2 = unique_ids_view2[col_idx]
        
        # Only include assignments with at least min_matches
        # Note: cost_matrix has negative values, so we negate it back
        if -cost_matrix[row_idx, col_idx] >= min_matches:
            optimal_assignments.append((seg_id_view1, seg_id_view2))
    
    return optimal_assignments


def visualize_segmentation_matches(image1, mask1, image2, mask2, paired_seg_ids, alpha=0.5):
    """
    Visualize segmentation masks overlaid on images with matched segmentation classes having the same color.
    
    Parameters:
    -----------
    image1 : numpy.ndarray
        First image (H, W, 3) in RGB format
    mask1 : numpy.ndarray
        First segmentation mask (H, W) with integer segmentation IDs
    image2 : numpy.ndarray
        Second image (H, W, 3) in RGB format
    mask2 : numpy.ndarray
        Second segmentation mask (H, W) with integer segmentation IDs
    paired_seg_ids : list of tuples
        List of paired segmentation IDs between the two views [(seg_id_view1, seg_id_view2), ...]
    alpha : float, optional
        Transparency of the mask overlay (0.0 to 1.0)
        
    Returns:
    --------
    vis_image1 : numpy.ndarray
        First image with segmentation mask overlay
    vis_image2 : numpy.ndarray
        Second image with segmentation mask overlay
    """
    # Ensure images are RGB and float32 in range [0, 1]
    image1 = np.asarray(image1, dtype=np.float32) / 255.0
    image2 = np.asarray(image2, dtype=np.float32) / 255.0
    
    # Create colormap using jet for the segment classes
    n_classes = len(paired_seg_ids)
    cmap_colors = cm.jet(np.linspace(0, 1, n_classes))
    
    # Create empty visualization images
    vis_image1 = image1.copy()
    vis_image2 = image2.copy()
    
    # Create colored masks
    colored_mask1 = np.zeros((*mask1.shape, 3), dtype=np.float32)
    colored_mask2 = np.zeros((*mask2.shape, 3), dtype=np.float32)
    
    # Create binary masks to track which pixels are covered by any segment
    mask1_covered = np.zeros_like(mask1, dtype=bool)
    mask2_covered = np.zeros_like(mask2, dtype=bool)
    
    # Assign colors to each paired segment
    for i, (seg_id_view1, seg_id_view2) in enumerate(paired_seg_ids):
        # Get binary masks for this segment in each view
        segment_mask1 = mask1 == seg_id_view1
        segment_mask2 = mask2 == seg_id_view2
        
        # Skip if segment doesn't exist in either view
        if not np.any(segment_mask1) and not np.any(segment_mask2):
            continue
        
        # Get color for this class pair
        color = cmap_colors[i][:3]  # Get RGB values (exclude alpha)
        
        # Apply color to this segment in both masks
        for c in range(3):  # RGB channels
            colored_mask1[:, :, c][segment_mask1] = color[c]
            colored_mask2[:, :, c][segment_mask2] = color[c]
        
        # Update covered areas
        mask1_covered |= segment_mask1
        mask2_covered |= segment_mask2
    
    # Apply colored masks to the original images using alpha blending
    # Only apply to pixels that are covered by any segment
    vis_image1[mask1_covered] = (1-alpha) * image1[mask1_covered] + alpha * colored_mask1[mask1_covered]
    vis_image2[mask2_covered] = (1-alpha) * image2[mask2_covered] + alpha * colored_mask2[mask2_covered]
    
    # Convert back to uint8 for visualization
    vis_image1 = (vis_image1 * 255).astype(np.uint8)
    vis_image2 = (vis_image2 * 255).astype(np.uint8)
    
    return vis_image1, vis_image2


def visualize_correspondences(
    img1: np.ndarray,
    img2: np.ndarray,
    corr_pts1: np.ndarray,
    corr_pts2: np.ndarray,
    n_viz: int = 100,
) -> np.ndarray:
    """
    Visualize correspondences between two images using OpenCV.
    
    Parameters
    ----------
    img1 : np.ndarray
        First image (H0, W0, 3) RGB format
    img2 : np.ndarray
        Second image (H1, W1, 3) RGB format
    corr_pts1 : np.ndarray
        Correspondence points in first image (M, 2), where each row is [x, y]
    corr_pts2 : np.ndarray
        Correspondence points in second image (M, 2), where each row is [x, y]
    n_viz : int, optional
        Number of correspondences to visualize, by default 100
    save_path : str, optional
        Path to save the visualization, by default None
    title : str, optional
        Title for the visualization, by default "Image Correspondences"
    
    Returns
    -------
    np.ndarray
        Visualization image with correspondences drawn
    """
    # Make sure images are in BGR for OpenCV (if they're in RGB)
    img1_cv = cv2.cvtColor(img1, cv2.COLOR_RGB2BGR) if img1.shape[2] == 3 else img1
    img2_cv = cv2.cvtColor(img2, cv2.COLOR_RGB2BGR) if img2.shape[2] == 3 else img2
    
    # Get image dimensions
    H0, W0 = img1_cv.shape[:2]
    H1, W1 = img2_cv.shape[:2]
    
    # Pad images to have the same height
    img1_padded = np.pad(img1_cv, ((0, max(H1 - H0, 0)), (0, 0), (0, 0)), 'constant', constant_values=0)
    img2_padded = np.pad(img2_cv, ((0, max(H0 - H1, 0)), (0, 0), (0, 0)), 'constant', constant_values=0)
    
    # Concatenate images horizontally
    concat_img = np.concatenate((img1_padded, img2_padded), axis=1)
    
    # Ensure we have correspondences to visualize
    num_matches = min(len(corr_pts1), len(corr_pts2))
    if num_matches == 0:
        print("No correspondences to visualize")
        return concat_img
    
    # Limit the number of visualized matches
    n_viz = min(n_viz, num_matches)
    
    # Select matches to visualize (evenly spaced)
    match_idx_to_viz = np.round(np.linspace(0, num_matches - 1, n_viz)).astype(int)
    viz_matches_im_view1 = corr_pts1[match_idx_to_viz]
    viz_matches_im_view2 = corr_pts2[match_idx_to_viz]
    
    # Create a copy of the concatenated image for drawing
    vis_img = concat_img.copy()
    
    # Draw correspondences
    for i in range(n_viz):
        x0, y0 = int(viz_matches_im_view1[i][0]), int(viz_matches_im_view1[i][1])
        x1, y1 = int(viz_matches_im_view2[i][0]), int(viz_matches_im_view2[i][1])
        
        # Calculate color using HSV colormap (similar to jet in matplotlib)
        # Convert i/n_viz to a hue value (0-179 for OpenCV)
        hue = int(179 * i / (n_viz - 1)) if n_viz > 1 else 0
        color = cv2.cvtColor(np.uint8([[[hue, 255, 255]]]), cv2.COLOR_HSV2BGR)[0, 0].tolist()
        
        # Draw points and lines
        # Point in the first image
        cv2.circle(vis_img, (x0, y0), 5, color, -1)
        # Point in the second image (offset by width of first image)
        cv2.circle(vis_img, (x1 + W0, y1), 5, color, -1)
        # Line connecting the points
        cv2.line(vis_img, (x0, y0), (x1 + W0, y1), color, 2)
    
    return vis_img


def find_correspondence_indices_with_segmentation(
    query_pts1: np.ndarray,
    valid1: np.ndarray,
    segmentation1: np.ndarray,
    query_pts2: np.ndarray,
    valid2: np.ndarray,
    segmentation2: np.ndarray,
    correspondence_pts1: np.ndarray,
    correspondence_pts2: np.ndarray,
    correspondence_segmentation: np.ndarray,
    pixel_tol: float = 3.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find matching correspondence indices between two sets of query points from different views,
    considering segmentation classes and validity flags.
    
    Parameters
    ----------
    query_pts1 : np.ndarray
        Query points in view 1, shape (N1, 2)
    valid1 : np.ndarray
        Validity flags for query points in view 1, shape (N1,)
    segmentation1 : np.ndarray
        Segmentation classes for query points in view 1, shape (N1,)
    query_pts2 : np.ndarray
        Query points in view 2, shape (N2, 2)
    valid2 : np.ndarray
        Validity flags for query points in view 2, shape (N2,)
    segmentation2 : np.ndarray
        Segmentation classes for query points in view 2, shape (N2,)
    correspondence_pts1 : np.ndarray
        Correspondence points in view 1, shape (M, 2)
    correspondence_pts2 : np.ndarray
        Correspondence points in view 2, shape (M, 2)
    correspondence_segmentation : np.ndarray
        Segmentation classes for correspondences, shape (M,)
    pixel_tol : float, optional
        Pixel tolerance for matching, by default 3.0
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        matching_indices: Array of shape (P, 2) containing matching indices in query_pts1 and query_pts2.
        matching_segmentation: Array of shape (P,) containing segmentation classes for each match.
    """
    N1 = len(query_pts1)
    N2 = len(query_pts2)
    M = len(correspondence_pts1)
    
    # Verify input shapes
    assert len(valid1) == N1, "valid1 must have the same length as query_pts1"
    assert len(segmentation1) == N1, "segmentation1 must have the same length as query_pts1"
    assert len(valid2) == N2, "valid2 must have the same length as query_pts2"
    assert len(segmentation2) == N2, "segmentation2 must have the same length as query_pts2"
    assert correspondence_pts2.shape[0] == M, "correspondence_pts2 must have the same length as correspondence_pts1"
    assert correspondence_segmentation.shape[0] == M, "correspondence_segmentation must have the same length as correspondence_pts1"
    
    # Use only valid query points
    valid_mask1 = valid1.astype(bool)
    valid_mask2 = valid2.astype(bool)
    
    if not np.any(valid_mask1) or not np.any(valid_mask2):
        return np.zeros((0, 2), dtype=int), np.zeros(0, dtype=correspondence_segmentation.dtype)
    
    # Get unique segmentation classes from correspondence data
    unique_seg_classes = np.unique(correspondence_segmentation)
    
    matching_indices = []
    sq_pixel_tol = pixel_tol ** 2  # Use squared tolerance

    # Process each segmentation class separately
    for seg_class in unique_seg_classes:
        if seg_class == 0:
            continue
        # Obtain indices of correspondence points for this segmentation
        corr_seg_mask = correspondence_segmentation == seg_class
        if not np.any(corr_seg_mask):
            continue
        
        corr_pts1_seg = correspondence_pts1[corr_seg_mask]
        corr_pts2_seg = correspondence_pts2[corr_seg_mask]
        
        # Select query points in view 1 for this segmentation and that are valid
        query1_seg_mask = (segmentation1 == seg_class) & valid_mask1
        if not np.any(query1_seg_mask):
            continue
        query_pts1_seg = query_pts1[query1_seg_mask]
        query1_indices = np.nonzero(query1_seg_mask)[0]
        
        # Select query points in view 2 for this segmentation and that are valid
        query2_seg_mask = (segmentation2 == seg_class) & valid_mask2
        if not np.any(query2_seg_mask):
            continue
        query_pts2_seg = query_pts2[query2_seg_mask]
        query2_indices = np.nonzero(query2_seg_mask)[0]
        
        # --- Batch processing for view 1 ---
        # Compute squared distances between each correspondence point and each query point in view 1:
        # result shape: (num_corr, num_query1)
        diff1 = corr_pts1_seg[:, None, :] - query_pts1_seg[None, :, :]
        dists1 = np.sum(diff1**2, axis=2)
        # Create a mask for points within threshold; set distance to infinity if beyond tol.
        masked_dists1 = np.where(dists1 <= sq_pixel_tol, dists1, np.inf)
        # For each correspondence point, find the minimum distance and the index of that query point.
        min_dists1 = np.min(masked_dists1, axis=1)  # shape: (num_corr,)
        closest_idx1 = np.argmin(masked_dists1, axis=1)  # local index in query_pts1_seg
        
        # --- Batch processing for view 2 ---
        diff2 = corr_pts2_seg[:, None, :] - query_pts2_seg[None, :, :]
        dists2 = np.sum(diff2**2, axis=2)
        masked_dists2 = np.where(dists2 <= sq_pixel_tol, dists2, np.inf)
        min_dists2 = np.min(masked_dists2, axis=1)
        closest_idx2 = np.argmin(masked_dists2, axis=1)
        
        # Identify correspondence points with valid matches in both views (where a distance under tol was found)
        valid_matches = (min_dists1 != np.inf) & (min_dists2 != np.inf)
        if not np.any(valid_matches):
            continue
        
        # Map local indices back to global indices
        final_query1_indices = query1_indices[closest_idx1[valid_matches]]
        final_query2_indices = query2_indices[closest_idx2[valid_matches]]
        
        # Append matches and corresponding segmentation label
        matches = np.stack((final_query1_indices, final_query2_indices), axis=1)
        matching_indices.append(matches)
        
    # Combine results from all segmentation groups
    if matching_indices:
        matching_indices = np.concatenate(matching_indices, axis=0)
    else:
        matching_indices = np.zeros((0, 2), dtype=int)
    return matching_indices


def find_correspondence_indices_with_segmentation_torch(
    query_pts1: np.ndarray,
    valid1: np.ndarray,
    segmentation1: np.ndarray,
    query_pts2: np.ndarray,
    valid2: np.ndarray,
    segmentation2: np.ndarray,
    correspondence_pts1: np.ndarray,
    correspondence_pts2: np.ndarray,
    correspondence_segmentation: np.ndarray,
    pixel_tol: float = 3.0,
    device: str = "cuda",
    batch_size: int = 1024  # Added batch size parameter
) -> np.ndarray:
    """
    Find matching correspondence indices between two sets of query points from different views,
    considering segmentation classes and validity flags, with PyTorch CUDA acceleration.
    
    The input data is provided as NumPy arrays. They are converted to PyTorch tensors,
    moved to the specified device (default "cuda"), processed in a batched manner to avoid OOM issues,
    and the resulting matching indices are moved back to NumPy.
    
    Parameters
    ----------
    query_pts1 : np.ndarray
        Query points in view 1, shape (N1, 2).
    valid1 : np.ndarray
        Validity flags for query points in view 1, shape (N1,).
    segmentation1 : np.ndarray
        Segmentation classes for query points in view 1, shape (N1,).
    query_pts2 : np.ndarray
        Query points in view 2, shape (N2, 2).
    valid2 : np.ndarray
        Validity flags for query points in view 2, shape (N2,).
    segmentation2 : np.ndarray
        Segmentation classes for query points in view 2, shape (N2,).
    correspondence_pts1 : np.ndarray
        Correspondence points in view 1, shape (M, 2).
    correspondence_pts2 : np.ndarray
        Correspondence points in view 2, shape (M, 2).
    correspondence_segmentation : np.ndarray
        Segmentation classes for correspondences, shape (M,).
    pixel_tol : float, optional
        Pixel tolerance for matching (in pixels), by default 3.0.
    device : str, optional
        The device to use ("cuda" or "cpu"). Defaults to "cuda" if available.
    batch_size : int, optional
        Maximum batch size for distance calculations to prevent OOM issues. Defaults to 1024.
    
    Returns
    -------
    np.ndarray
        matching_indices: Array of shape (P, 2) containing matching indices in query_pts1 and query_pts2.
    """
    # Ensure that we use the desired device (fallback to CPU if CUDA is unavailable)
    device = torch.device(device if torch.cuda.is_available() and "cuda" in device else "cpu")
    
    # Convert NumPy arrays to PyTorch tensors and move to device.
    # Use appropriate dtypes: coordinates are float, segmentation indices are integer, valid flags to bool.
    query_pts1_t = torch.from_numpy(query_pts1).to(device=device, dtype=torch.float32)
    valid1_t = torch.from_numpy(valid1).to(device=device).bool()
    segmentation1_t = torch.from_numpy(segmentation1).to(device=device)
    
    query_pts2_t = torch.from_numpy(query_pts2).to(device=device, dtype=torch.float32)
    valid2_t = torch.from_numpy(valid2).to(device=device).bool()
    segmentation2_t = torch.from_numpy(segmentation2).to(device=device)
    
    correspondence_pts1_t = torch.from_numpy(correspondence_pts1).to(device=device, dtype=torch.float32)
    correspondence_pts2_t = torch.from_numpy(correspondence_pts2).to(device=device, dtype=torch.float32)
    correspondence_segmentation_t = torch.from_numpy(correspondence_segmentation).to(device=device)
    
    N1 = query_pts1_t.shape[0]
    N2 = query_pts2_t.shape[0]
    M = correspondence_pts1_t.shape[0]
    
    # Basic assertions (optional)
    assert valid1_t.shape[0] == N1, "valid1 must have the same length as query_pts1"
    assert segmentation1_t.shape[0] == N1, "segmentation1 must have the same length as query_pts1"
    assert valid2_t.shape[0] == N2, "valid2 must have the same length as query_pts2"
    assert segmentation2_t.shape[0] == N2, "segmentation2 must have the same length as query_pts2"
    assert correspondence_pts2_t.shape[0] == M, "correspondence_pts2 must have the same length as correspondence_pts1"
    assert correspondence_segmentation_t.shape[0] == M, "correspondence_segmentation must have the same length as correspondence_pts1"
    
    # Use only valid query points.
    if not valid1_t.any() or not valid2_t.any():
        return np.zeros((0, 2), dtype=int)
    
    # Precompute squared pixel tolerance.
    sq_pixel_tol = pixel_tol ** 2
    
    # Get unique segmentation classes from correspondence data.
    unique_seg_classes = torch.unique(correspondence_segmentation_t)
    
    matching_indices_list = []
    inf_tensor = torch.tensor(float('inf'), dtype=torch.float32, device=device)
    
    # Process each segmentation class.
    for seg_class in unique_seg_classes:
        if seg_class == 0:
            continue
        # Mask for correspondence points with this segmentation.
        corr_seg_mask = (correspondence_segmentation_t == seg_class)
        if not corr_seg_mask.any():
            continue
        
        corr_pts1_seg = correspondence_pts1_t[corr_seg_mask]
        corr_pts2_seg = correspondence_pts2_t[corr_seg_mask]
        
        # Select valid query points in view 1 with matching segmentation.
        query1_seg_mask = (segmentation1_t == seg_class) & valid1_t
        if not query1_seg_mask.any():
            continue
        query_pts1_seg = query_pts1_t[query1_seg_mask]
        query1_indices = torch.where(query1_seg_mask)[0]
        
        # Select valid query points in view 2 with matching segmentation.
        query2_seg_mask = (segmentation2_t == seg_class) & valid2_t
        if not query2_seg_mask.any():
            continue
        query_pts2_seg = query_pts2_t[query2_seg_mask]
        query2_indices = torch.where(query2_seg_mask)[0]
        
        # --- Batched processing for view 1 ---
        num_corr_seg = corr_pts1_seg.shape[0]
        num_query1_seg = query_pts1_seg.shape[0]
        
        # Pre-allocate tensor for results
        min_dists1 = torch.full((num_corr_seg,), float('inf'), dtype=torch.float32, device=device)
        closest_idx1 = torch.zeros((num_corr_seg,), dtype=torch.int64, device=device)
        
        # Process batches of correspondence points
        for i in range(0, num_corr_seg, batch_size):
            end_i = min(i + batch_size, num_corr_seg)
            corr_batch = corr_pts1_seg[i:end_i]
            
            # Process each query point batch for this correspondence batch
            batch_min_dists = torch.full((end_i - i, num_query1_seg), float('inf'), dtype=torch.float32, device=device)
            for j in range(0, num_query1_seg, batch_size):
                end_j = min(j + batch_size, num_query1_seg)
                query_batch = query_pts1_seg[j:end_j]
                
                # Calculate distances between this batch of correspondence points and query points
                diff1 = corr_batch.unsqueeze(1) - query_batch.unsqueeze(0)
                batch_dists = torch.sum(diff1 ** 2, dim=2)
                
                # Apply distance threshold
                masked_dists = torch.where(batch_dists <= sq_pixel_tol, batch_dists, inf_tensor)
                batch_min_dists[:, j:end_j] = masked_dists
            
            # Find minimum distances and indices for this batch
            batch_min_values, batch_min_indices = torch.min(batch_min_dists, dim=1)
            
            # Update overall minimum distances and indices
            min_dists1[i:end_i] = batch_min_values
            closest_idx1[i:end_i] = batch_min_indices
        
        # --- Batched processing for view 2 ---
        num_query2_seg = query_pts2_seg.shape[0]
        
        # Pre-allocate tensor for results
        min_dists2 = torch.full((num_corr_seg,), float('inf'), dtype=torch.float32, device=device)
        closest_idx2 = torch.zeros((num_corr_seg,), dtype=torch.int64, device=device)
        
        # Process batches of correspondence points
        for i in range(0, num_corr_seg, batch_size):
            end_i = min(i + batch_size, num_corr_seg)
            corr_batch = corr_pts2_seg[i:end_i]
            
            # Process each query point batch for this correspondence batch
            batch_min_dists = torch.full((end_i - i, num_query2_seg), float('inf'), dtype=torch.float32, device=device)
            for j in range(0, num_query2_seg, batch_size):
                end_j = min(j + batch_size, num_query2_seg)
                query_batch = query_pts2_seg[j:end_j]
                
                # Calculate distances between this batch of correspondence points and query points
                diff2 = corr_batch.unsqueeze(1) - query_batch.unsqueeze(0)
                batch_dists = torch.sum(diff2 ** 2, dim=2)
                
                # Apply distance threshold
                masked_dists = torch.where(batch_dists <= sq_pixel_tol, batch_dists, inf_tensor)
                batch_min_dists[:, j:end_j] = masked_dists
            
            # Find minimum distances and indices for this batch
            batch_min_values, batch_min_indices = torch.min(batch_min_dists, dim=1)
            
            # Update overall minimum distances and indices
            min_dists2[i:end_i] = batch_min_values
            closest_idx2[i:end_i] = batch_min_indices
        
        # Keep only correspondence points that have valid matches in both views.
        valid_matches = (min_dists1 != inf_tensor) & (min_dists2 != inf_tensor)
        if not valid_matches.any():
            continue
        
        # Map local indices back to global query point indices.
        final_query1_indices = query1_indices[closest_idx1[valid_matches]]
        final_query2_indices = query2_indices[closest_idx2[valid_matches]]
        
        # Stack indices horizontally as pairs.
        matches = torch.stack((final_query1_indices, final_query2_indices), dim=1)
        matching_indices_list.append(matches)
    
    # Combine results from all segmentation groups.
    if matching_indices_list:
        matching_indices = torch.cat(matching_indices_list, dim=0)
    else:
        matching_indices = torch.empty((0, 2), dtype=torch.int64, device=device)
    
    # Move matching indices back to CPU as a NumPy array.
    return matching_indices.cpu().numpy()



def find_correspondence_indices_torch(
    query_pts1: np.ndarray,
    valid1: np.ndarray,
    query_pts2: np.ndarray,
    valid2: np.ndarray,
    correspondence_pts1: np.ndarray,
    correspondence_pts2: np.ndarray,
    pixel_tol: float = 3.0,
    device: str = "cuda",
    batch_size: int = 1024
) -> np.ndarray:
    """
    Find matching correspondence indices between two sets of query points from different views,
    considering validity flags, with PyTorch CUDA acceleration.
    
    The input data is provided as NumPy arrays. They are converted to PyTorch tensors,
    moved to the specified device (default "cuda"), processed in a batched manner to avoid OOM issues,
    and the resulting matching indices are moved back to NumPy.
    
    Parameters
    ----------
    query_pts1 : np.ndarray
        Query points in view 1, shape (N1, 2).
    valid1 : np.ndarray
        Validity flags for query points in view 1, shape (N1,).
    query_pts2 : np.ndarray
        Query points in view 2, shape (N2, 2).
    valid2 : np.ndarray
        Validity flags for query points in view 2, shape (N2,).
    correspondence_pts1 : np.ndarray
        Correspondence points in view 1, shape (M, 2).
    correspondence_pts2 : np.ndarray
        Correspondence points in view 2, shape (M, 2).
    pixel_tol : float, optional
        Pixel tolerance for matching (in pixels), by default 3.0.
    device : str, optional
        The device to use ("cuda" or "cpu"). Defaults to "cuda" if available.
    batch_size : int, optional
        Maximum batch size for distance calculations to prevent OOM issues. Defaults to 1024.
    
    Returns
    -------
    np.ndarray
        matching_indices: Array of shape (P, 2) containing matching indices in query_pts1 and query_pts2.
    """
    # Ensure that we use the desired device (fallback to CPU if CUDA is unavailable)
    device = torch.device(device if torch.cuda.is_available() and "cuda" in device else "cpu")
    
    # Convert NumPy arrays to PyTorch tensors and move to device.
    # Use appropriate dtypes: coordinates are float, valid flags to bool.
    query_pts1_t = torch.from_numpy(query_pts1).to(device=device, dtype=torch.float32)
    valid1_t = torch.from_numpy(valid1).to(device=device).bool()
    
    query_pts2_t = torch.from_numpy(query_pts2).to(device=device, dtype=torch.float32)
    valid2_t = torch.from_numpy(valid2).to(device=device).bool()
    
    correspondence_pts1_t = torch.from_numpy(correspondence_pts1).to(device=device, dtype=torch.float32)
    correspondence_pts2_t = torch.from_numpy(correspondence_pts2).to(device=device, dtype=torch.float32)
    
    N1 = query_pts1_t.shape[0]
    N2 = query_pts2_t.shape[0]
    M = correspondence_pts1_t.shape[0]
    
    # Basic assertions (optional)
    assert valid1_t.shape[0] == N1, "valid1 must have the same length as query_pts1"
    assert valid2_t.shape[0] == N2, "valid2 must have the same length as query_pts2"
    assert correspondence_pts2_t.shape[0] == M, "correspondence_pts2 must have the same length as correspondence_pts1"
    
    # Use only valid query points.
    if not valid1_t.any() or not valid2_t.any():
        return np.zeros((0, 2), dtype=int)
    
    # Precompute squared pixel tolerance.
    sq_pixel_tol = pixel_tol ** 2
    
    # Select valid query points
    query1_mask = valid1_t
    if not query1_mask.any():
        return np.zeros((0, 2), dtype=int)
    query_pts1_valid = query_pts1_t[query1_mask]
    query1_indices = torch.where(query1_mask)[0]
    
    query2_mask = valid2_t
    if not query2_mask.any():
        return np.zeros((0, 2), dtype=int)
    query_pts2_valid = query_pts2_t[query2_mask]
    query2_indices = torch.where(query2_mask)[0]
    
    matching_indices_list = []
    inf_tensor = torch.tensor(float('inf'), dtype=torch.float32, device=device)
    
    # --- Batched processing for view 1 ---
    num_corr = correspondence_pts1_t.shape[0]
    num_query1_valid = query_pts1_valid.shape[0]
    
    # Pre-allocate tensor for results
    min_dists1 = torch.full((num_corr,), float('inf'), dtype=torch.float32, device=device)
    closest_idx1 = torch.zeros((num_corr,), dtype=torch.int64, device=device)
    
    # Process batches of correspondence points
    for i in range(0, num_corr, batch_size):
        end_i = min(i + batch_size, num_corr)
        corr_batch = correspondence_pts1_t[i:end_i]
        
        # Process each query point batch for this correspondence batch
        batch_min_dists = torch.full((end_i - i, num_query1_valid), float('inf'), dtype=torch.float32, device=device)
        for j in range(0, num_query1_valid, batch_size):
            end_j = min(j + batch_size, num_query1_valid)
            query_batch = query_pts1_valid[j:end_j]
            
            # Calculate distances between this batch of correspondence points and query points
            diff1 = corr_batch.unsqueeze(1) - query_batch.unsqueeze(0)
            batch_dists = torch.sum(diff1 ** 2, dim=2)
            
            # Apply distance threshold
            masked_dists = torch.where(batch_dists <= sq_pixel_tol, batch_dists, inf_tensor)
            batch_min_dists[:, j:end_j] = masked_dists
        
        # Find minimum distances and indices for this batch
        batch_min_values, batch_min_indices = torch.min(batch_min_dists, dim=1)
        
        # Update overall minimum distances and indices
        min_dists1[i:end_i] = batch_min_values
        closest_idx1[i:end_i] = batch_min_indices
    
    # --- Batched processing for view 2 ---
    num_query2_valid = query_pts2_valid.shape[0]
    
    # Pre-allocate tensor for results
    min_dists2 = torch.full((num_corr,), float('inf'), dtype=torch.float32, device=device)
    closest_idx2 = torch.zeros((num_corr,), dtype=torch.int64, device=device)
    
    # Process batches of correspondence points
    for i in range(0, num_corr, batch_size):
        end_i = min(i + batch_size, num_corr)
        corr_batch = correspondence_pts2_t[i:end_i]
        
        # Process each query point batch for this correspondence batch
        batch_min_dists = torch.full((end_i - i, num_query2_valid), float('inf'), dtype=torch.float32, device=device)
        for j in range(0, num_query2_valid, batch_size):
            end_j = min(j + batch_size, num_query2_valid)
            query_batch = query_pts2_valid[j:end_j]
            
            # Calculate distances between this batch of correspondence points and query points
            diff2 = corr_batch.unsqueeze(1) - query_batch.unsqueeze(0)
            batch_dists = torch.sum(diff2 ** 2, dim=2)
            
            # Apply distance threshold
            masked_dists = torch.where(batch_dists <= sq_pixel_tol, batch_dists, inf_tensor)
            batch_min_dists[:, j:end_j] = masked_dists
        
        # Find minimum distances and indices for this batch
        batch_min_values, batch_min_indices = torch.min(batch_min_dists, dim=1)
        
        # Update overall minimum distances and indices
        min_dists2[i:end_i] = batch_min_values
        closest_idx2[i:end_i] = batch_min_indices
    
    # Keep only correspondence points that have valid matches in both views.
    valid_matches = (min_dists1 != inf_tensor) & (min_dists2 != inf_tensor)
    if not valid_matches.any():
        return np.zeros((0, 2), dtype=int)
    
    # Map local indices back to global query point indices.
    final_query1_indices = query1_indices[closest_idx1[valid_matches]]
    final_query2_indices = query2_indices[closest_idx2[valid_matches]]
    
    # Stack indices horizontally as pairs.
    matches = torch.stack((final_query1_indices, final_query2_indices), dim=1)
    
    # Move matching indices back to CPU as a NumPy array.
    return matches.cpu().numpy()


def deduplicate_correspondences(correspondences1, correspondences2):
    """
    Deduplicate correspondence pairs when points are within one pixel of each other.

    Safer implementation:
    - avoids NumPy structured-view dtype issues
    - works with float32 / float64 / non-contiguous arrays
    - keeps the first occurrence and preserves original order
    """
    correspondences1 = np.asarray(correspondences1)
    correspondences2 = np.asarray(correspondences2)

    if correspondences1.size == 0 or correspondences2.size == 0:
        return np.array([], dtype=np.int64)

    combined = np.hstack((correspondences1, correspondences2))

    # Make sure shape is (N, 4)
    combined = np.asarray(combined).reshape(-1, 4)

    # Round to pixel-level coordinates.
    rounded = np.rint(combined).astype(np.int64)

    # np.unique(axis=0) is slower than structured view but much safer.
    _, unique_indices = np.unique(rounded, axis=0, return_index=True)

    # Preserve original match order.
    unique_indices = np.sort(unique_indices).astype(np.int64)

    return unique_indices


def filter_correspondences(correspondences1, correspondences2, r1, r2, M_min, tau):
    """
    Filter correspondence pairs based on spatial coherence.
    
    Parameters:
    ----------
    correspondences1 : numpy.ndarray, shape (N, 2)
        Pixel coordinates in the first view.
    correspondences2 : numpy.ndarray, shape (N, 2)
        Corresponding pixel coordinates in the second view.
    r1 : float
        Radius for neighborhood query in the first view.
    r2 : float
        Maximum allowed distance in the second view.
    M_min : int
        Minimum number of neighbors required.
    tau : float
        Minimum fraction of neighbors that must be coherent.
    
    Returns:
    -------
    numpy.ndarray
        Indices of valid correspondences.
    """
    # Create KD-tree for first set of correspondences for efficient neighbor search
    KD1 = KDTree(correspondences1)
    
    valid_indices = []
    
    # Check each correspondence pair
    for i in range(len(correspondences1)):
        # Get coordinates for current point
        u1_i = correspondences1[i]
        u2_i = correspondences2[i]
        
        # Find neighbors within radius r1 in the first view
        nbrs1 = KD1.query_ball_point(u1_i, r1)
        
        # Check if there are enough neighbors
        if len(nbrs1) < M_min:
            continue  # Discard this correspondence
        
        # Check which neighbors also land nearby in the second view
        good = []
        for j in nbrs1:
            # Calculate distance in second view
            dist = np.linalg.norm(correspondences2[j] - u2_i)
            if dist < r2:
                good.append(j)
        
        # Check if enough neighbors are coherent
        if len(good) / len(nbrs1) >= tau:
            valid_indices.append(i)
    
    return np.array(valid_indices)


def filter_correspondences_torch(correspondences1, correspondences2, r1, r2, M_min, tau, device='cpu', max_batch_size=5000):
    """
    Filter correspondence pairs based on spatial coherence using PyTorch with batched operations.
    
    Parameters:
    ----------
    correspondences1 : numpy.ndarray, shape (N, 2)
        Pixel coordinates in the first view.
    correspondences2 : numpy.ndarray, shape (N, 2)
        Corresponding pixel coordinates in the second view.
    r1 : float
        Radius for neighborhood query in the first view.
    r2 : float
        Maximum allowed distance in the second view.
    M_min : int
        Minimum number of neighbors required.
    tau : float
        Minimum fraction of neighbors that must be coherent.
    device : str
        Device to run computations on ('cpu' or 'cuda').
    max_batch_size : int
        Maximum batch size to avoid OOM errors.
    
    Returns:
    -------
    numpy.ndarray
        Indices of valid correspondences.
    """
    # Convert numpy arrays to torch tensors and move to specified device
    corr1 = torch.tensor(correspondences1, dtype=torch.float32, device=device)
    corr2 = torch.tensor(correspondences2, dtype=torch.float32, device=device)
    
    N = corr1.shape[0]
    valid_indices = []
    
    # Process data in batches to avoid OOM
    for start_idx in range(0, N, max_batch_size):
        end_idx = min(start_idx + max_batch_size, N)
        batch_size = end_idx - start_idx
        
        # Get current batch
        batch_corr1 = corr1[start_idx:end_idx]
        batch_corr2 = corr2[start_idx:end_idx]
        
        # For each point in the batch, compute distances to all other points
        # This is still efficient but avoids creating a full N×N matrix
        batch_nbrs_count = torch.zeros(batch_size, dtype=torch.int64, device=device)
        batch_good_count = torch.zeros(batch_size, dtype=torch.int64, device=device)
        
        # Process reference points in chunks as well to avoid OOM
        for ref_start in range(0, N, max_batch_size):
            ref_end = min(ref_start + max_batch_size, N)
            
            ref_corr1 = corr1[ref_start:ref_end]
            ref_corr2 = corr2[ref_start:ref_end]
            
            # Compute distances between batch points and reference points
            # Shape: [batch_size, ref_size, 2]
            diff1 = batch_corr1.unsqueeze(1) - ref_corr1.unsqueeze(0)
            dist1_squared = torch.sum(diff1 * diff1, dim=2)  # [batch_size, ref_size]
            
            diff2 = batch_corr2.unsqueeze(1) - ref_corr2.unsqueeze(0)
            dist2_squared = torch.sum(diff2 * diff2, dim=2)  # [batch_size, ref_size]
            
            # Create masks for this chunk
            nbrs_mask = dist1_squared < (r1 * r1)  # [batch_size, ref_size]
            good_mask = dist2_squared < (r2 * r2)  # [batch_size, ref_size]
            
            # Update counts
            batch_nbrs_count += torch.sum(nbrs_mask, dim=1)
            batch_good_count += torch.sum(nbrs_mask & good_mask, dim=1)
        
        # Calculate ratio of good neighbors to all neighbors
        batch_ratio = torch.zeros_like(batch_nbrs_count, dtype=torch.float32)
        valid_nbrs = batch_nbrs_count > 0
        batch_ratio[valid_nbrs] = batch_good_count[valid_nbrs].float() / batch_nbrs_count[valid_nbrs].float()
        
        # Find valid indices in this batch
        batch_valid = torch.where(
            (batch_nbrs_count >= M_min) & (batch_ratio >= tau)
        )[0]
        
        # Adjust indices to global index space and add to result
        global_indices = batch_valid + start_idx
        valid_indices.append(global_indices)
    
    # Concatenate all valid indices
    if valid_indices:
        result = torch.cat(valid_indices).cpu().numpy()
    else:
        result = np.array([], dtype=np.int64)
        
    return result


def filter_correspondence_pairs(pairs, threshold):
    """
    Filter video correspondence pairs based on frequency of unique point indices pairs.
    
    Args:
        pairs: numpy array or list of shape (N, 4) where each row contains
              [t1, t2, point_idx1, point_idx2]
        threshold: minimum frequency required to keep a point pair
    
    Returns:
        List of shape (M, 2) containing unique valid point indices pairs
    """
    import numpy as np
    from collections import Counter
    
    # Convert to numpy array if not already
    pairs = np.array(pairs)
    
    # Get the point indices (last two columns)
    point_indices = pairs[:, 2:4]
    
    # Count occurrences of each point index pair
    # We need to convert each pair to a tuple to make it hashable for Counter
    point_pairs = [tuple(pair) for pair in point_indices]
    point_counter = Counter(point_pairs)
    
    # Filter to only keep pairs that appear more than threshold times
    valid_pairs = [pair for pair, count in point_counter.items() if count >= threshold]
    
    # Convert back to numpy array of shape (M, 2)
    return np.array(valid_pairs)



def filter_temporally_consistent_correspondences_efficient_torch(tracklet1, tracklet2, valid1, valid2, correspondences, pixel_tol=5.0, min_neighbors=3, device='cuda', batch_size=1000):
    """
    Filter tracklets to get temporal-consistent correspondences using PyTorch vectorized operations with optional CUDA acceleration.
    Memory-efficient implementation using batch processing to prevent OOM errors.
    Also considers the minimum number of neighbors criterion.
    
    Args:
        tracklet1 (np.ndarray): Array of shape (T1, N1, 2) for the first tracklet.
        tracklet2 (np.ndarray): Array of shape (T2, N2, 2) for the second tracklet.
        valid1 (np.ndarray): Boolean array of shape (T1, N1) indicating valid points for tracklet1.
        valid2 (np.ndarray): Boolean array of shape (T2, N2) indicating valid points for tracklet2.
        correspondences (np.ndarray): Array of shape (M, 4) with correspondences as [t1, t2, n1, n2].
        pixel_tol (float): Distance threshold in pixels.
        min_neighbors (int): Minimum number of neighbors required to keep a point.
        device (str): Device to run computations on ('cpu' or 'cuda').
        batch_size (int): Maximum number of correspondences to process at once to limit memory usage.
    
    Returns:
        np.ndarray: Array of shape (P, 2) with filtered correspondences as [n1, n2].
    """
    
    # Convert inputs to torch tensors and move to the specified device
    tracklet1_t = torch.from_numpy(tracklet1).to(device)
    tracklet2_t = torch.from_numpy(tracklet2).to(device)
    valid1_t = torch.from_numpy(valid1).to(device)
    valid2_t = torch.from_numpy(valid2).to(device)
    
    # Get dimensions from valid masks
    T1, N1 = valid1.shape
    T2, N2 = valid2.shape
    
    # Start with all correspondences (only indices n1 and n2 are used)
    reliable = set((int(n1), int(n2)) for _, _, n1, n2 in correspondences)
    
    # Compute the squared threshold since distances are compared in squared space
    pixel_tol_sq = pixel_tol ** 2
    
    # Loop over every combination of time steps in the two tracklets
    for t1 in range(T1):
        for t2 in range(T2):
            if not reliable:
                break
            
            # Convert the current reliable set to a tensor on the chosen device
            corr_array = torch.tensor(list(reliable), dtype=torch.long, device=device)
            
            # Determine which correspondences are valid at time t1 (for tracklet1) and t2 (for tracklet2)
            valid_mask = valid1_t[t1, corr_array[:, 0]] & valid2_t[t2, corr_array[:, 1]]
            if not valid_mask.any():
                continue  # No valid correspondences at these times
            
            # Select the valid correspondences for the current time step
            valid_corr = corr_array[valid_mask]  # Shape: (L, 2)
            
            # Get the 2D positions for the valid correspondence indices
            pos1 = tracklet1_t[t1, valid_corr[:, 0]]  # Shape: (L, 2)
            pos2 = tracklet2_t[t2, valid_corr[:, 1]]  # Shape: (L, 2)
            
            L = pos1.shape[0]
            if L < 1:
                continue
                
            # Process in batches to avoid OOM
            # Initialize tensors to store results for all points
            src_keep = torch.ones(L, dtype=torch.bool, device=device)
            tgt_keep = torch.ones(L, dtype=torch.bool, device=device)
            
            # Process in batches
            for i in range(0, L, batch_size):
                batch_end = min(i + batch_size, L)
                batch_size_actual = batch_end - i
                
                # --- Source (tracklet1) consistency check for current batch ---
                # Get positions for current batch
                pos1_batch = pos1[i:batch_end]
                pos2_batch = pos2[i:batch_end]
                
                # Compute pairwise distances for all points with the current batch
                neighbors1_batch = torch.zeros(batch_size_actual, L, dtype=torch.bool, device=device)
                neighbors2_batch = torch.zeros(batch_size_actual, L, dtype=torch.bool, device=device)
                consistent_count1_batch = torch.zeros(batch_size_actual, dtype=torch.long, device=device)
                consistent_count2_batch = torch.zeros(batch_size_actual, dtype=torch.long, device=device)
                count_neighbors1_batch = torch.zeros(batch_size_actual, dtype=torch.long, device=device)
                count_neighbors2_batch = torch.zeros(batch_size_actual, dtype=torch.long, device=device)
                
                # Process in sub-batches for comparison
                for j in range(0, L, batch_size):
                    sub_batch_end = min(j + batch_size, L)
                    
                    # Compute distances between current batch and sub-batch
                    diff1 = pos1_batch.unsqueeze(1) - pos1[j:sub_batch_end].unsqueeze(0)  # Shape: (batch_size_actual, sub_batch_size, 2)
                    dists1_sq = (diff1 ** 2).sum(dim=2)  # Shape: (batch_size_actual, sub_batch_size)
                    
                    diff2 = pos2_batch.unsqueeze(1) - pos2[j:sub_batch_end].unsqueeze(0)
                    dists2_sq = (diff2 ** 2).sum(dim=2)
                    
                    # Create mask for self-comparisons within the overlap of both batches
                    if i <= j < batch_end or j <= i < sub_batch_end:
                        overlap_start = max(i, j)
                        overlap_end = min(batch_end, sub_batch_end)
                        if overlap_end > overlap_start:
                            # Calculate relative indices in the current batch matrices
                            rel_i_start = overlap_start - i
                            rel_i_end = overlap_end - i
                            rel_j_start = overlap_start - j
                            rel_j_end = overlap_end - j
                            
                            # Create the diagonal mask for the overlapping region
                            for idx_i, idx_j in zip(range(rel_i_start, rel_i_end), range(rel_j_start, rel_j_end)):
                                dists1_sq[idx_i, idx_j] = float('inf')
                                dists2_sq[idx_i, idx_j] = float('inf')
                    
                    # Find neighbors
                    neigh1 = dists1_sq <= pixel_tol_sq  # Shape: (batch_size_actual, sub_batch_size)
                    neigh2 = dists2_sq <= pixel_tol_sq
                    
                    # Update the full neighbor matrices for this batch
                    neighbors1_batch[:, j:sub_batch_end] = neigh1
                    neighbors2_batch[:, j:sub_batch_end] = neigh2
                    
                    # Update counts
                    count_neighbors1_batch += neigh1.sum(dim=1)
                    count_neighbors2_batch += neigh2.sum(dim=1)
                    
                    # Count consistent correspondences
                    consistent_count1_batch += (neigh1 & (dists2_sq <= pixel_tol_sq)).sum(dim=1)
                    consistent_count2_batch += (neigh2 & (dists1_sq <= pixel_tol_sq)).sum(dim=1)
                
                # Final decision for this batch
                # Check if the point has at least min_neighbors
                has_min_neighbors1 = count_neighbors1_batch >= min_neighbors
                has_min_neighbors2 = count_neighbors2_batch >= min_neighbors
                
                # Check consistency ratio (only for points that have enough neighbors)
                consistency_ratio1 = torch.zeros_like(count_neighbors1_batch, dtype=torch.float, device=device)
                consistency_ratio2 = torch.zeros_like(count_neighbors2_batch, dtype=torch.float, device=device)
                
                # Calculate consistency ratio only for points with neighbors
                mask1 = count_neighbors1_batch > 0
                mask2 = count_neighbors2_batch > 0
                
                if mask1.any():
                    consistency_ratio1[mask1] = consistent_count1_batch[mask1].float() / count_neighbors1_batch[mask1].float()
                
                if mask2.any():
                    consistency_ratio2[mask2] = consistent_count2_batch[mask2].float() / count_neighbors2_batch[mask2].float()
                 
                # Final criteria: has minimum number of neighbors AND has a sufficient consistency ratio
                src_keep_batch = has_min_neighbors1  & (consistency_ratio1 >= 0.5)
                tgt_keep_batch = has_min_neighbors2 & (consistency_ratio2 >= 0.5)
                
                # Update the full result arrays
                src_keep[i:batch_end] = src_keep_batch
                tgt_keep[i:batch_end] = tgt_keep_batch
            
            # Final decision: keep the correspondence if both source and target checks pass
            keep = src_keep & tgt_keep
            
            # Remove any correspondences that do not pass the consistency check
            if not torch.all(keep):
                # Convert the failing correspondences to a list on the CPU
                remove_pairs = valid_corr[~keep].cpu().tolist()
                for pair in remove_pairs:
                    reliable.discard(tuple(pair))
    
    # Convert the final reliable correspondences set back to a NumPy array
    if reliable:
        result = np.array(list(reliable), dtype=int)
    else:
        result = np.empty((0, 2), dtype=int)
        
    return result


def visualize_video_correspondences(
    video1: np.ndarray,
    video2: np.ndarray,
    tracks1: np.ndarray,
    tracks2: np.ndarray,
    pred_valid1: np.ndarray,
    pred_valid2: np.ndarray,
    n_viz: int = 20,
    output_path: str = "correspondence_video.mp4",
    fps: int = 30,
    target_resolution: tuple = (1920, 1080)
) -> str:
    """
    Visualize correspondences between two videos using tracklets.
    
    Parameters
    ----------
    video1 : np.ndarray
        First video (T, H, W, 3) RGB format
    video2 : np.ndarray
        Second video (T, H, W, 3) RGB format
    tracks1 : np.ndarray
        Correspondence points in first video (T, N, 2), where each row is [x, y]
    tracks2 : np.ndarray
        Correspondence points in second video (T, N, 2), where each row is [x, y]
    pred_valid1 : np.ndarray
        Validity flags for tracks in first video (T, N)
    pred_valid2 : np.ndarray
        Validity flags for tracks in second video (T, N)
    n_viz : int, optional
        Number of correspondences to visualize, by default 20
    output_path : str, optional
        Path to save the visualization video, by default "correspondence_video.mp4"
    fps : int, optional
        Frames per second for output video, by default 30
    target_resolution : tuple, optional
        Target resolution for output video (width, height), by default (1920, 1080)
        
    Returns
    -------
    str
        Path to the saved video file
    """
    # Get video dimensions and length
    T, H0, W0, C = video1.shape
    _, H1, W1, _ = video2.shape
    
    # Get number of tracklets
    _, N, _ = tracks1.shape
    
    # Select subset of tracklets to visualize
    if N > n_viz:
        # Select n_viz points evenly spaced
        tracklet_indices = np.round(np.linspace(0, N - 1, n_viz)).astype(int)
    else:
        tracklet_indices = np.arange(N)
        n_viz = N
    
    # Set up video writer with imageio
    video_writer = imageio.get_writer(output_path, fps=fps, quality=9)
    
    # Create a different color for each tracklet
    colors = []
    for i in range(n_viz):
        # Calculate color using HSV colormap (similar to jet in matplotlib)
        # Convert i/n_viz to a hue value (0-179 for OpenCV)
        hue = int(179 * i / (n_viz - 1)) if n_viz > 1 else 0
        color = cv2.cvtColor(np.uint8([[[hue, 255, 255]]]), cv2.COLOR_HSV2BGR)[0, 0].tolist()
        colors.append(color)
    
    # Calculate target dimensions
    target_width, target_height = target_resolution
    
    # Process each frame
    for t in range(T):
        # Convert RGB to BGR for OpenCV
        img1_cv = cv2.cvtColor(video1[t], cv2.COLOR_RGB2BGR)
        img2_cv = cv2.cvtColor(video2[t], cv2.COLOR_RGB2BGR)
        
        # Pad images to have the same height
        img1_padded = np.pad(img1_cv, ((0, max(H1 - H0, 0)), (0, 0), (0, 0)), 'constant', constant_values=0)
        img2_padded = np.pad(img2_cv, ((0, max(H0 - H1, 0)), (0, 0), (0, 0)), 'constant', constant_values=0)
        
        # Concatenate images horizontally
        concat_img = np.concatenate((img1_padded, img2_padded), axis=1)
        
        # Create a copy of the concatenated image for drawing
        vis_img = concat_img.copy()
        
        # Draw correspondences for selected tracklets
        for i, idx in enumerate(tracklet_indices):
            # Check if this tracklet is valid in both views at this time
            is_valid1 = pred_valid1[t, idx] > 0
            is_valid2 = pred_valid2[t, idx] > 0
            
            # Get positions
            x0, y0 = tracks1[t, idx]
            x1, y1 = tracks2[t, idx]
            
            # Convert to integers
            x0, y0 = int(x0), int(y0)
            x1, y1 = int(x1), int(y1)
            
            # Get color for this tracklet
            color = colors[i]
            
            # Always draw points if they're valid in their respective views (larger points)
            if is_valid1:
                cv2.circle(vis_img, (x0, y0), 8, color, -1)
            
            if is_valid2:
                cv2.circle(vis_img, (x1 + W0, y1), 8, color, -1)
            
            # Draw correspondence line only if both points are valid (thicker, more solid line)
            if is_valid1 and is_valid2:
                cv2.line(vis_img, (x0, y0), (x1 + W0, y1), color, 3)
        
        # Resize the visualization to target resolution
        vis_img_resized = cv2.resize(vis_img, (target_width, target_height), interpolation=cv2.INTER_LANCZOS4)
        
        # Write frame to video using imageio writer
        video_writer.append_data(cv2.cvtColor(vis_img_resized, cv2.COLOR_BGR2RGB))
    
    # Close the video writer
    video_writer.close()
    
    print(f"Video saved to {output_path} with resolution {target_resolution}")
    return output_path


if __name__ == "__main__":
    from imgcat import imgcat
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", type=str, default="/data11/shaowei3/datasets/egohumans/data2_preprocessed_corrected", help="dataset dir")
    parser.add_argument("--result_root", type=str, default="/data11/shaowei3/datasets/egohumans/data2_preprocessed_corrected_results_v2_improved", help="seq name")
    parser.add_argument("--track_root", type=str, default="/data11/shaowei3/datasets/egohumans/data2_preprocessed_tracks_v2_improved", help="seq name")
    parser.add_argument("--result_name1", type=str, default="volleyball_aria02_16_141")
    parser.add_argument("--result_name2", type=str, default="volleyball_aria04_32_119")
    
    parser.add_argument("--corr_radius", type=float, default=20.0, help="correspondence radius")
    parser.add_argument('--min_matches', help='minimum number of matches to consider a segmentation ID pair valid', default=100, type=int)
    parser.add_argument("--confidence_threshold", type=float, default=0., help="confidence values higher than threshold are invalid")
    parser.add_argument("--pixel_tol", type=float, default=2.0, help="pixel tolerance for matching")
    parser.add_argument("--min_neighbors", type=int, default=1, help="minimum number of neighbors for filtering")   
    
    parser.add_argument("--mask_prefix", type=str, default="deva_improved", help="mask prefix")
    parser.add_argument("--vis_seg_match", action="store_true", help="visualize the segmentation matches")
    parser.add_argument("--vis_corr", action="store_true", help="visualize the filtered correspondences")
    parser.add_argument("--vis_track_corr", action="store_true", help="visualize the filtered track correspondences")
    parser.add_argument("--vis_final_corr", action="store_true", help="visualize the final correspondences")
    parser.add_argument("--vis_final_video", action="store_true", help="visualize the final correspondences in video")
    parser.add_argument("--max_batch_size", type=int, default=4096, help="max batch size for filtering correspondences")
    parser.add_argument("--viz_matches", type=int, default=10, help="visualize matches")
    parser.add_argument("--debug", action="store_true", help="debug mode")
    
    # recommend use default values for the following arguments
    parser.add_argument("--disable_seg_match", action="store_true", help="disable segmentation matching")
    parser.add_argument("--apply_filter", action="store_true", help="apply filter")
    
    parser.add_argument("--group_prefix", type=str, default=None, help="group prefix")
    
    args = parser.parse_args()
    
    def get_group_prefix(name):
        """
        PRIN folder names look like:
            ID_0_fpv_000_200
            ID_0_cam_tpv_000_200
            ID_0_cam_top_000_200

        We want group prefix = ID_0, ID_1, ...
        For old VisualSync-style names, fallback to first token.
        """
        name = name.split("__")[0]
        parts = name.split("_")

        if len(parts) >= 2 and parts[0] == "ID":
            return f"{parts[0]}_{parts[1]}"

        return parts[0]


    if args.group_prefix is None:
        prefix1 = get_group_prefix(args.result_name1)
        prefix2 = get_group_prefix(args.result_name2)
        assert prefix1 == prefix2, f"prefix not same: {prefix1} vs {prefix2}"
    else:
        prefix1 = args.group_prefix
        prefix2 = args.group_prefix
    result_name = f"{args.result_name1}__{args.result_name2}"
    result_dir = os.path.join(args.result_root, prefix1, result_name)
    seq_dir1, seq_dir2 = os.path.basename(result_dir).split("__")
    
    image_dir1 = os.path.join(args.dataset_root, seq_dir1, 'rgb_aligned')
    image_dir2 = os.path.join(args.dataset_root, seq_dir2, 'rgb_aligned')
    
    image_files1 = [f for ext in ["*.jpg", "*.jpeg", "*.png"] for f in glob.glob(os.path.join(image_dir1, ext))]
    image_files1.sort()  
    image_files2 = [f for ext in ["*.jpg", "*.jpeg", "*.png"] for f in glob.glob(os.path.join(image_dir2, ext))]
    image_files2.sort()
    
    tracks1_path = os.path.join(args.track_root, seq_dir1, "tracks.pkl")
    tracks2_path = os.path.join(args.track_root, seq_dir2, "tracks.pkl")
    with open(tracks1_path, "rb") as f:
        result_dict1 = pickle.load(f)
    with open(tracks2_path, "rb") as f:
        result_dict2 = pickle.load(f)
    pred_tracks1 = result_dict1["pred_tracks"]# T1 N1 2
    pred_tracks2 = result_dict2["pred_tracks"] # T2 N2 2
    pred_tracks1_seg = result_dict1["seg_ids"].astype(np.int32) # N1
    assert not (pred_tracks1_seg == 0).any()
    pred_tracks2_seg = result_dict2["seg_ids"].astype(np.int32) # N2
    assert not (pred_tracks2_seg == 0).any()
    
    pred_valid1 = result_dict1["pred_valid"] # T1 N1
    pred_valid2 = result_dict2["pred_valid"]# T2 N2
    
    assert pred_tracks1.shape[0] == len(image_files1)
    assert pred_tracks2.shape[0] == len(image_files2)
    
    mask_dir1 = os.path.join(args.dataset_root, seq_dir1, args.mask_prefix, "Annotations")
    mask_dir2 = os.path.join(args.dataset_root, seq_dir2, args.mask_prefix, "Annotations")
   
    masks1_list, mask1_files = read_file_from_dir(mask_dir1, read_mask=True)
    masks2_list, mask2_files = read_file_from_dir(mask_dir2, read_mask=True)
    
    masks1 = np.array(masks1_list).astype(np.uint8) # (T, H, W)
    masks2 = np.array(masks2_list).astype(np.uint8) # (T, H, W)
    
    with open(os.path.join(result_dir, "matches.pkl"), 'rb') as f:
        corr_list = pickle.load(f)
    
    save_dict = {}
    
    if not args.disable_seg_match:
        corr_seg_list = np.empty((0, 2), dtype=int)
        for corr_result in corr_list:
            img1_path = corr_result["img1_path"]
            img2_path = corr_result["img2_path"]
            frame1 = image_files1.index(img1_path)
            frame2 = image_files2.index(img2_path)
            assert frame1 == corr_result["frame_idx1"]
            assert frame2 == corr_result["frame_idx2"]
            
            mask1 = masks1[frame1]
            mask2 = masks2[frame2]
            
            dmatches_im_view1 = corr_result["dmatches_im_view1"]
            dmatches_im_view2 = corr_result["dmatches_im_view2"]
            dmatches_confs = corr_result["dmatches_confs"]
        
            unique_indices = deduplicate_correspondences(dmatches_im_view1, dmatches_im_view2)
            
            if args.debug:
                print("{}/{} unique matches".format(len(unique_indices), len(dmatches_im_view2)))
            if len(unique_indices) < 2:
                continue
            
            dmatches_im_view1 = dmatches_im_view1[unique_indices]
            dmatches_im_view2 = dmatches_im_view2[unique_indices]
            dmatches_confs = dmatches_confs[unique_indices]
            if isinstance(dmatches_confs, float):
                dmatches_confs = np.array([dmatches_confs])
            
            conf_mask = dmatches_confs > args.confidence_threshold
            
            # if args.debug:
            #     print("{}/{} matches after confidence threshold".format(conf_mask.sum(), len(dmatches_im_view2)))
            
            dmatches_im_view1 = dmatches_im_view1[conf_mask]
            dmatches_im_view2 = dmatches_im_view2[conf_mask]
            dmatches_confs = dmatches_confs[conf_mask]
        
            corr_result.update({"dmatches_im_view1": dmatches_im_view1, "dmatches_im_view2": dmatches_im_view2, "dmatches_confs": dmatches_confs})
            
            if len(dmatches_im_view1)==0 or len(dmatches_im_view2)==0:
                continue
            
            if dmatches_im_view1.ndim == 1:
                dmatches_im_view1 = dmatches_im_view1.reshape(-1, 2) # (N, 2)
            if dmatches_im_view2.ndim == 1:
                dmatches_im_view2 = dmatches_im_view2.reshape(-1, 2) # (N, 2)
            if dmatches_confs.ndim == 0:
                dmatches_confs = np.array([dmatches_confs]) # (1,)
            
            pts1 = np.rint(dmatches_im_view1).astype(int)
            seg1 = mask1[pts1[:, 1], pts1[:, 0]] # (N,)
            
            pts2 = np.rint(dmatches_im_view2).astype(int)
            seg2 = mask2[pts2[:, 1], pts2[:, 0]] # (N,)
            
            valid_mask = (seg1 != 0) & (seg2 != 0)

            seg1 = seg1[valid_mask]
            seg2 = seg2[valid_mask]
            
            corr_seg = np.stack([seg1, seg2], axis=1) # (N, 2)
            corr_seg_list = np.concatenate([corr_seg_list, corr_seg], axis=0)
        
        # find optimal segmenation matches
        paired_seg_ids = find_segmentation_associations(corr_seg_list, min_matches=args.min_matches)
        
        # visualize the segmentation matches
        frame1_index = np.argmax(pred_valid1.sum(axis=1)) # np.random.randint(0, len(image_files1)) 
        frame2_index = np.argmax(pred_valid2.sum(axis=1)) # np.random.randint(0, len(image_files2))
        frame1 = iio.imread(image_files1[frame1_index])
        frame2 = iio.imread(image_files2[frame2_index])
        mask_frame1 = iio.imread(os.path.join(mask_dir1, mask1_files[frame1_index]), mode='F').astype(np.uint8)
        mask_frame2 = iio.imread(os.path.join(mask_dir2, mask2_files[frame2_index]), mode='F').astype(np.uint8)
    
        vis_img1, vis_img2 = visualize_segmentation_matches(frame1, mask_frame1, frame2, mask_frame2, paired_seg_ids)
        img_name1 = image_files1[frame1_index].split("/")[-1]
        save_path1 = os.path.join(result_dir, f"seg_match_{args.result_name1}_{img_name1}")
        # print(f"save segmentation match image to {save_path1}")
        iio.imwrite(save_path1, vis_img1)
        img_name1 = image_files2[frame2_index].split("/")[-1]
        save_path2 = os.path.join(result_dir, f"seg_match_{args.result_name2}_{img_name1}")
        # print(f"save segmentation match image to {save_path2}")
        iio.imwrite(save_path2, vis_img2)
        
        if args.vis_seg_match: 
            imgcat(frame1) 
            imgcat(frame2)  
            imgcat(vis_img1)
            imgcat(vis_img2)
        
        # filter wrong correspondences
        valid_seg_ids_view1 = {seg_id1 for seg_id1, _ in paired_seg_ids}
        valid_seg_ids_view2 = {seg_id2 for _, seg_id2 in paired_seg_ids}
        
        # Create mapping from original seg IDs to new consistent IDs
        seg_id1_to_new = {seg_id1: i+1 for i, (seg_id1, _) in enumerate(paired_seg_ids)}
        seg_id2_to_new = {seg_id2: i+1 for i, (_, seg_id2) in enumerate(paired_seg_ids)}
        
        # Step 1: Remap segmentation IDs with fallback to 0
        map_seg1 = np.vectorize(lambda x: seg_id1_to_new.get(x, 0))
        map_seg2 = np.vectorize(lambda x: seg_id2_to_new.get(x, 0))

        pred_tracks1_seg = map_seg1(pred_tracks1_seg)  
        pred_tracks2_seg = map_seg2(pred_tracks2_seg)

        # Step 2: Set invalid entries where remapped seg is 0
        pred_valid1[:, pred_tracks1_seg == 0] = False
        pred_valid2[:, pred_tracks2_seg == 0] = False
        
    track_corr_list = []
    for corr_result in tqdm(corr_list):
        img1_path = corr_result["img1_path"]
        img2_path = corr_result["img2_path"]
        frame1 = image_files1.index(img1_path)
        frame2 = image_files2.index(img2_path)
        mask1 = masks1[frame1]
        mask2 = masks2[frame2]
        
        dmatches_im_view1 = corr_result["dmatches_im_view1"]
        dmatches_im_view2 = corr_result["dmatches_im_view2"]
        dmatches_confs = corr_result["dmatches_confs"]
        
        if len(dmatches_im_view1)==0 or len(dmatches_im_view2)==0:
            continue
        
        if dmatches_im_view1.ndim == 1:
            dmatches_im_view1 = dmatches_im_view1.reshape(-1, 2) # (N, 2)
        if dmatches_im_view2.ndim == 1:
            dmatches_im_view2 = dmatches_im_view2.reshape(-1, 2) # (N, 2)
        if dmatches_confs.ndim == 0:
            dmatches_confs = np.array([dmatches_confs]) # (1,)
        
        pts1 = np.rint(dmatches_im_view1).astype(int)
        seg1 = mask1[pts1[:, 1], pts1[:, 0]] # (N,)
        
        pts2 = np.rint(dmatches_im_view2).astype(int)
        seg2 = mask2[pts2[:, 1], pts2[:, 0]] # (N,)
        
        valid_mask = (seg1 != 0) & (seg2 != 0)
        seg1 = seg1[valid_mask]
        seg2 = seg2[valid_mask]
        dmatches_im_view1 = dmatches_im_view1[valid_mask]
        dmatches_im_view2 = dmatches_im_view2[valid_mask]
        dmatches_confs = dmatches_confs[valid_mask]
        
        if len(seg1) == 0 or len(seg2) == 0:
            # print("seg1 or seg2 is empty, skip this pair")
            continue
        if not args.disable_seg_match:
            map_seg1 = np.vectorize(lambda x: seg_id1_to_new.get(x, 0))
            map_seg2 = np.vectorize(lambda x: seg_id2_to_new.get(x, 0))

            new_seg1 = map_seg1(seg1)
            new_seg2 = map_seg2(seg2)

            # remove correspondences not have same class
            mask = new_seg1 == new_seg2
            mask = (new_seg1 == new_seg2) & (new_seg1 != 0) & (new_seg2 != 0)
            
            filter_new_seg1 = new_seg1[mask]
            filter_new_seg2 = new_seg2[mask]
            
            if len(filter_new_seg1) == 0 or len(filter_new_seg2) == 0:
                # print("filter_new_seg1 or filter_new_seg2 is empty, skip this pair")
                continue
            
            dmatches_im_view1 = dmatches_im_view1[mask]
            dmatches_im_view2 = dmatches_im_view2[mask]
            dmatches_confs = dmatches_confs[mask]
            
            # filter correspondences by confidence threshold
            conf_mask = dmatches_confs > args.confidence_threshold
            dmatches_im_view1 = dmatches_im_view1[conf_mask]
            dmatches_im_view2 = dmatches_im_view2[conf_mask]
            dmatches_confs = dmatches_confs[conf_mask]
            
            if len(dmatches_im_view1) == 0 or len(dmatches_im_view2) == 0:
                continue
        
        if args.apply_filter:
            valid_indices = filter_correspondences_torch(dmatches_im_view1, dmatches_im_view2, r1=args.corr_radius, r2=args.corr_radius, M_min=10, tau=0.5, device="cuda", max_batch_size=args.max_batch_size)
            print("{}/{} matches after spatial filter".format(len(valid_indices), len(dmatches_im_view2)))
            if len(valid_indices) == 0:
                continue
            
            # TODO keep on torch device without shifting to numpy
            dmatches_im_view1 = dmatches_im_view1[valid_indices]
            dmatches_im_view2 = dmatches_im_view2[valid_indices]
            dmatches_confs = dmatches_confs[valid_indices]
            if not args.disable_seg_match:
                filter_new_seg1 = filter_new_seg1[valid_indices]
                filter_new_seg2 = filter_new_seg2[valid_indices]
            
        # visualize the filtered correspondences (optional)
        img1 = iio.imread(img1_path)
        img2 = iio.imread(img2_path)
        
        if args.vis_corr:
            print("vis filtered correspondences on frame1: ", frame1, " frame2: ", frame2)
            vis_img = visualize_correspondences(img1, img2, dmatches_im_view1, dmatches_im_view2, n_viz=20) 
            imgcat(vis_img[:,:, ::-1])
            
        # corr_result.update({"dmatches_im_view1": dmatches_im_view1, "dmatches_im_view2": dmatches_im_view2, "dmatches_confs": dmatches_confs})
    
        tracks1 = pred_tracks1[frame1] # N1 2
        tracks2 = pred_tracks2[frame2] # N2 2
        tracks1_valid1 = pred_valid1[frame1] # N1
        tracks2_valid2 = pred_valid2[frame2] # N2
        if not args.disable_seg_match:
            matching_indices = find_correspondence_indices_with_segmentation_torch(
                query_pts1=tracks1,
                valid1=tracks1_valid1,
                segmentation1=pred_tracks1_seg,
                query_pts2=tracks2,
                valid2=tracks2_valid2,
                segmentation2=pred_tracks2_seg,
                correspondence_pts1=dmatches_im_view1,
                correspondence_pts2=dmatches_im_view2,
                correspondence_segmentation=filter_new_seg1,
                pixel_tol=args.pixel_tol, 
                batch_size=args.max_batch_size)
        else:
            matching_indices = find_correspondence_indices_torch(
                query_pts1=tracks1,
                valid1=tracks1_valid1,
                query_pts2=tracks2,
                valid2=tracks2_valid2,
                correspondence_pts1=dmatches_im_view1,
                correspondence_pts2=dmatches_im_view2,
                pixel_tol=args.pixel_tol,
                batch_size=args.max_batch_size)
        
        if len(matching_indices) == 0:
            # print("matching_indices is empty, skip this pair")
            continue
        
        if args.vis_track_corr:
            # visualize the filtered track correspondences
            track_corr1 = tracks1[matching_indices[:, 0]]
            track_corr2 = tracks2[matching_indices[:, 1]]
            assert tracks1_valid1[matching_indices[:, 0]].all()
            assert tracks2_valid2[matching_indices[:, 1]].all()
            
            vis_img = visualize_correspondences(img1, img2, track_corr1, track_corr2, n_viz=20) 
            print("vis filtered track correspondences on frame1: ", frame1, " frame2: ", frame2)
            imgcat(vis_img[:,:, ::-1])                            
    
        t1 = np.ones_like(matching_indices[:, 0:1], dtype=int) * frame1
        t2 = np.ones_like(matching_indices[:, 1:2], dtype=int) * frame2 
        track_corr = np.concatenate([t1, t2, matching_indices], axis=1) # (N, 4)
        track_corr_list.append(track_corr)
    
    if len(track_corr_list) == 0:
        print("no filtered track correspondences")
        exit()
    
    track_corr_list = np.concatenate(track_corr_list, axis=0) # (N, 4)
    print("number of filtered track correspondences: ", len(track_corr_list))

    if len(track_corr_list) > 0:
        
        # save_dict["track_corr_list"] = track_corr_list
        
        # filtered_corr_indices = filter_temporally_consistent_correspondences_efficient_torch(
        #     pred_tracks1,
        #     pred_tracks2,
        #     pred_valid1,
        #     pred_valid2,
        #     track_corr_list,
        #     pixel_tol=args.corr_radius, # keep it small for neiborhood consistency
        #     batch_size=args.max_batch_size,
        #     min_neighbors=args.min_neighbors
        # )
        
        filtered_corr_indices = track_corr_list[:, 2:]
        
        print("number of filtered track correspondences after temporal consistency: ", len(filtered_corr_indices))
        
        if len(filtered_corr_indices) == 0:
            print("no filtered correspondences")
   
        else:
            # save results
            save_dict["seg_id1_to_new"] = seg_id1_to_new
            save_dict["seg_id2_to_new"] = seg_id2_to_new
            
            save_dict["filtered_corr_indices"] = filtered_corr_indices      
            
            save_dict["tracks1"] = {
            "pred_tracks": pred_tracks1.astype(np.float32),
            "pred_valid": pred_valid1.astype(np.bool_),
            "pred_tracks_seg": pred_tracks1_seg.astype(np.uint8)}
            
            save_dict["tracks2"] = {
                "pred_tracks": pred_tracks2.astype(np.float32),
                "pred_valid": pred_valid2.astype(np.bool_),
                "pred_tracks_seg": pred_tracks2_seg.astype(np.uint8),
            }
        
            # save_dict["filtered_tracks1"] = {
            #     "pred_tracks": pred_tracks1[:, filtered_corr_indices[:, 0]].astype(np.float32),
            #     "pred_valid": pred_valid1[:, filtered_corr_indices[:, 0]].astype(np.bool_),
            #     "pred_tracks_seg": pred_tracks1_seg[filtered_corr_indices[:, 0]].astype(np.uint8),
            # }
            # save_dict["filtered_tracks2"] = {
            #     "pred_tracks": pred_tracks2[:, filtered_corr_indices[:, 1]].astype(np.float32),
            #     "pred_valid": pred_valid2[:, filtered_corr_indices[:, 1]].astype(np.bool_),
            #     "pred_tracks_seg": pred_tracks2_seg[filtered_corr_indices[:, 1]].astype(np.uint8),
            # }
            
            if args.vis_final_video:
                track1 = pred_tracks1[:, filtered_corr_indices[:, 0]]
                track2 = pred_tracks2[:, filtered_corr_indices[:, 1]]
                valid_mask1 = pred_valid1[:, filtered_corr_indices[:, 0]]
                valid_mask2 = pred_valid2[:, filtered_corr_indices[:, 1]]
                video1, _ = read_file_from_dir(image_dir1)
                video2, _ = read_file_from_dir(image_dir2)
                video1 = np.array(video1)
                video2 = np.array(video2)
                min_length = min(len(video1), len(video2))
                video1 = video1[:min_length]
                video2 = video2[:min_length]
                track1 = track1[:min_length]
                track2 = track2[:min_length]
                valid_mask1 = valid_mask1[:min_length]
                valid_mask2 = valid_mask2[:min_length]
                video_save_path = os.path.join(result_dir, f"vis_{args.result_name1}_{args.result_name2}_corr.mp4")
                visualize_video_correspondences(video1, video2, track1, track2, valid_mask1, valid_mask2, n_viz=args.viz_matches,
                                                output_path=video_save_path, fps=10)
                print(f"save video to path {video_save_path}")
            else:
                frame1_index = np.argmax(pred_valid1.sum(axis=1))
                frame2_index = np.argmax(pred_valid2.sum(axis=1))
                img1 = iio.imread(image_files1[frame1_index])
                img2 = iio.imread(image_files2[frame2_index])
                valid_mask1 = pred_valid1[frame1_index][filtered_corr_indices[:, 0]]
                valid_mask2 = pred_valid2[frame2_index][filtered_corr_indices[:, 1]]
                valid_mask = np.logical_and(valid_mask1, valid_mask2)
                track_corr1 = pred_tracks1[frame1_index][filtered_corr_indices[:, 0]] 
                track_corr2 = pred_tracks2[frame2_index][filtered_corr_indices[:, 1]]
                track_corr1 = track_corr1[valid_mask]
                track_corr2 = track_corr2[valid_mask]
                vis_img = visualize_correspondences(img1, img2, track_corr1, track_corr2, n_viz=20) 
                if args.vis_final_corr:
                    imgcat(vis_img[:,:, ::-1])
            
                img1_name = image_files1[frame1_index].split("/")[-1].split(".")[0]
                img2_name = image_files2[frame2_index].split("/")[-1].split(".")[0]
                save_path = os.path.join(result_dir, f"final_corr_{args.result_name1}_{img1_name}_{args.result_name2}_{img2_name}.png")
                print(f"save final correspondences image to {save_path}")
                iio.imwrite(save_path, vis_img[:,:, ::-1])
           
            # with open(os.path.join(result_dir, "tracks_match_v2.pkl"), 'wb') as f:
            #     pickle.dump(save_dict, f)
            
            np.savez_compressed(os.path.join(result_dir, "tracks_match_v2.npz"), **save_dict)
            
    
    
    
    
    
    
    
