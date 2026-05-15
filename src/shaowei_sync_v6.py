# Shaowei Sync Multi-videos per sequence

import os
from rich import print
import glob
import cv2
import imageio
import imageio.v3 as iio
import torch
import json
import matplotlib.pyplot as plt
import numpy as np
import pickle
import json
import argparse
from tqdm import tqdm
try:
    from imgcat import imgcat
except ImportError:
    def imgcat(*args, **kwargs):
        return None

from sync_utils_v6 import *


# TODO: tune moving_threshold, filter out static class in bg
# TODO: store the correspondence confidence and use it in mast3r/filter_corr.py and here to avoid include bad correspondences
# example of bad correspondences: data2_preprocessed_corrected_results_v2_improved/basketball/basketball_aria02_50_100__basketball_aria04_2_104 (confidence are low)

"""
Pre compute cross-view correspodence by mast3r/filter_corr.py
Step-1: filter static class by check pairwise sampson distance per sequence (example see co-tracker/sampson_lib_test2.py, filter_cotracker_v1_2.py, filter_cotracker_v1_3.py)
Step-2: apply poission disk sampling to get correspondences
Step-3: use updated compute_sampson_distance_temporal in sampson_lib.py (example in sampson_lib_test.py)
[optional]: save local minimal 
"""
    

def skew_symmetric_batch(vectors):
    """
    Compute a batch of skew-symmetric matrices for a set of 3D vectors using PyTorch.
    
    Parameters:
        vectors: PyTorch tensor of shape (T, 3)
    
    Returns:
        skew: PyTorch tensor of shape (T, 3, 3) where each 3x3 matrix is the skew-symmetric
              matrix of the corresponding vector.
    """
    T = vectors.shape[0]
    device = vectors.device
    skew = torch.zeros((T, 3, 3), device=device)
    
    skew[:, 0, 1] = -vectors[:, 2]
    skew[:, 0, 2] = vectors[:, 1]
    skew[:, 1, 0] = vectors[:, 2]
    skew[:, 1, 2] = -vectors[:, 0]
    skew[:, 2, 0] = -vectors[:, 1]
    skew[:, 2, 1] = vectors[:, 0]
    
    return skew


def compute_fundamental_matrices(video1_w2c, video2_w2c, video1_K, video2_K):
    """
    Compute the fundamental matrix for each corresponding frame pair in two videos in batch mode.
    
    Both video1_w2c and video2_w2c are expected to be PyTorch tensors of shape (T, 4, 4) 
    representing the world-to-camera transformation for T frames.
    
    The camera intrinsics for video 1 and video 2 are given by video1_K and video2_K,
    which are PyTorch tensors of shape (3, 3) or (T, 3, 3).
    
    Returns:
        F_all: PyTorch tensor of shape (T, 3, 3) containing the fundamental matrices for each frame.
    """
    # Ensure the two videos have the same number of frames.
    T1 = video1_w2c.shape[0]
    T2 = video2_w2c.shape[0]
    if T1 != T2:
        raise ValueError("The number of frames in video1_w2c and video2_w2c must be the same.")
    
    T = T1  # Number of frames
    device = video1_w2c.device
    
    # Extract rotation (upper left 3x3) and translation (first three elements of last column) for each frame.
    R1 = video1_w2c[:, :3, :3]  # Shape: (T, 3, 3)
    t1 = video1_w2c[:, :3, 3]   # Shape: (T, 3)
    R2 = video2_w2c[:, :3, :3]
    t2 = video2_w2c[:, :3, 3]
    
    # Compute the relative rotation R_rel = R2 * R1^T for each frame.
    R1_T = R1.transpose(1, 2)  # Transpose each frame's rotation matrix.
    R_rel = torch.bmm(R2, R1_T)  # Shape: (T, 3, 3)
    
    # Compute the relative translation:
    # For each frame, t_rel = t2 - (R_rel @ t1)
    t1_transformed = torch.bmm(R_rel, t1.unsqueeze(2)).squeeze(2)
    t_rel = t2 - t1_transformed  # Shape: (T, 3)
    
    # Generate the skew-symmetric matrices for the relative translations.
    t_skew = skew_symmetric_batch(t_rel)  # Shape: (T, 3, 3)
    
    # Compute the Essential matrix for each frame: E = [t_rel]_x * R_rel.
    E = torch.bmm(t_skew, R_rel)  # Shape: (T, 3, 3)
    
    # Handle camera intrinsics based on their shape
    # Check if video1_K is per-frame (T, 3, 3) or shared (3, 3)
    if len(video1_K.shape) == 2:  # Shape: (3, 3) - shared across all frames
        invK1 = torch.inverse(video1_K).to(device)
        # Expand to match the number of frames
        invK1 = invK1.unsqueeze(0).expand(T, -1, -1)  # Shape: (T, 3, 3)
    else:  # Shape: (T, 3, 3) - per-frame intrinsics
        # Compute inverse for each frame's intrinsic matrix
        invK1 = torch.stack([torch.inverse(k) for k in video1_K]).to(device)
    
    # Check if video2_K is per-frame (T, 3, 3) or shared (3, 3)
    if len(video2_K.shape) == 2:  # Shape: (3, 3) - shared across all frames
        invK2 = torch.inverse(video2_K).to(device)
        # Expand to match the number of frames
        invK2 = invK2.unsqueeze(0).expand(T, -1, -1)  # Shape: (T, 3, 3)
    else:  # Shape: (T, 3, 3) - per-frame intrinsics
        # Compute inverse for each frame's intrinsic matrix
        invK2 = torch.stack([torch.inverse(k) for k in video2_K]).to(device)
    
    # Transpose each frame's invK2
    invK2_T = invK2.transpose(1, 2)  # Shape: (T, 3, 3)
    
    # Compute the Fundamental matrix for each frame:
    # F = inv(video2_K).T * E * inv(video1_K)
    temp = torch.bmm(E, invK1)  # Shape: (T, 3, 3)
    F_all = torch.bmm(invK2_T, temp)  # Shape: (T, 3, 3)
    
    return F_all


import torch

def compute_cosine_epipolar_error(F, pts1, pts2):
    """
    Computes the cosine‐based epipolar error for corresponding points given Fundamental matrices.
    PyTorch GPU implementation.

    Args:
        F:     Fundamental matrices with shape (T, 3, 3)
        pts1:  First set of points with shape (T, N, 2)
        pts2:  Second set of points with shape (T, N, 2)

    Returns:
        Tensor of shape (T, N) containing
            ε²_CS = (d2ᵀ F d1)² / (‖d2‖² ‖F d1‖²)
                  + (d2ᵀ F d1)² / (‖Fᵀ d2‖² ‖d1‖²)
    """
    # --- prepare inputs ---
    if not isinstance(F, torch.Tensor):
        F = torch.tensor(F, dtype=torch.float32)
    if not isinstance(pts1, torch.Tensor):
        pts1 = torch.tensor(pts1, dtype=torch.float32)
    if not isinstance(pts2, torch.Tensor):
        pts2 = torch.tensor(pts2, dtype=torch.float32)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    F    = F.to(device)
    pts1 = pts1.to(device)
    pts2 = pts2.to(device)

    T, N = pts1.shape[0], pts1.shape[1]

    # homogeneous coords
    ones    = torch.ones((T, N, 1), device=device)
    pts1_h  = torch.cat([pts1, ones], dim=2)  # (T, N, 3)
    pts2_h  = torch.cat([pts2, ones], dim=2)  # (T, N, 3)

    # compute epipolar‐plane normals
    lines2 = torch.zeros((T, N, 3), device=device)  # E d1
    lines1 = torch.zeros((T, N, 3), device=device)  # Eᵀ d2
    for t in range(T):
        lines2[t] = (F[t]   @ pts1_h[t].T).T
        lines1[t] = (F[t].T @ pts2_h[t].T).T

    # scalar product d2ᵀ F d1
    inner = torch.sum(pts2_h * lines2, dim=2)       # (T, N)
    num   = inner**2                                # numerator

    # squared norms
    norm_d2_sq      = torch.sum(pts2_h**2,    dim=2)  # ‖d2‖²
    norm_Ed1_sq     = torch.sum(lines2**2,    dim=2)  # ‖E d1‖²
    norm_Et_d2_sq   = torch.sum(lines1**2,    dim=2)  # ‖Eᵀ d2‖²
    norm_d1_sq      = torch.sum(pts1_h**2,    dim=2)  # ‖d1‖²

    # build denominators
    eps    = 1e-10
    denom1 = torch.clamp(norm_d2_sq    * norm_Ed1_sq,   min=eps)
    denom2 = torch.clamp(norm_Et_d2_sq * norm_d1_sq,    min=eps)

    return num / denom1 + num / denom2



def compute_algebraic_epipolar_error(F, pts1, pts2):
    """
    Computes the algebraic residual |d2ᵀ F d1| for corresponding points given Fundamental matrices.
    PyTorch GPU implementation.

    Args:
        F:     Fundamental matrices with shape (T, 3, 3)
        pts1:  First set of points with shape (T, N, 2)
        pts2:  Second set of points with shape (T, N, 2)

    Returns:
        Tensor of shape (T, N) containing |d2ᵀ F d1|.
    """
    # --- prepare inputs ---
    if not isinstance(F, torch.Tensor):
        F = torch.tensor(F, dtype=torch.float32)
    if not isinstance(pts1, torch.Tensor):
        pts1 = torch.tensor(pts1, dtype=torch.float32)
    if not isinstance(pts2, torch.Tensor):
        pts2 = torch.tensor(pts2, dtype=torch.float32)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    F    = F.to(device)
    pts1 = pts1.to(device)
    pts2 = pts2.to(device)

    T, N = pts1.shape[0], pts1.shape[1]

    # homogeneous coords
    ones    = torch.ones((T, N, 1), device=device)
    pts1_h  = torch.cat([pts1, ones], dim=2)
    pts2_h  = torch.cat([pts2, ones], dim=2)

    # epipolar‐plane normals
    lines2 = torch.zeros((T, N, 3), device=device)
    for t in range(T):
        lines2[t] = (F[t] @ pts1_h[t].T).T

    # scalar residual
    inner = torch.sum(pts2_h * lines2, dim=2)  # (T, N)
    return torch.abs(inner)



def compute_sampson_distance_temporal(F, pts1, pts2):
    """
    Compute Sampson distance for temporal batches of points and fundamental matrices.
    
    Args:
        F: Fundamental matrices with shape (T, 3, 3)
        pts1: First set of points with shape (T, N, 2)
        pts2: Second set of points with shape (T, N, 2)
        
    Returns:
        Sampson distances with shape (T, N)
    """
    T, N = pts1.shape[:2]
    device = F.device
    
    # Add homogeneous coordinates (T, N, 3)
    ones = torch.ones((T, N, 1), device=device)
    pts1_h = torch.cat([pts1, ones], dim=2)
    pts2_h = torch.cat([pts2, ones], dim=2)
    
    # Compute F*x1 for each time step
    # Reshape pts1_h to (T*N, 3) for batch matrix multiplication
    pts1_reshaped = pts1_h.reshape(T*N, 3).unsqueeze(1)  # (T*N, 1, 3)
    F_expanded = F.repeat_interleave(N, dim=0)  # (T*N, 3, 3)
    
    Fx1 = torch.bmm(pts1_reshaped, F_expanded.transpose(1, 2)).squeeze(1)  # (T*N, 3)
    Fx1 = Fx1.reshape(T, N, 3)  # Reshape back to (T, N, 3)
    
    # Compute F^T*x2 for each time step
    pts2_reshaped = pts2_h.reshape(T*N, 3).unsqueeze(1)  # (T*N, 1, 3)
    Ftx2 = torch.bmm(pts2_reshaped, F_expanded).squeeze(1)  # (T*N, 3)
    Ftx2 = Ftx2.reshape(T, N, 3)  # Reshape back to (T, N, 3)
    
    # Compute (x2^T F x1) using batch dot product
    x2Fx1 = torch.sum(pts2_h * Fx1, dim=2)  # (T, N)
    num = x2Fx1 ** 2  # (T, N)
    
    # Compute denominator
    denom = Fx1[..., 0] ** 2 + Fx1[..., 1] ** 2 + Ftx2[..., 0] ** 2 + Ftx2[..., 1] ** 2  # (T, N)
    
    # Handle numerical stability
    denom = torch.clamp(denom, min=1e-10)
    
    # Compute Sampson distances
    sampson_distances = num / denom  # (T, N)
    
    return sampson_distances



def compute_symmetric_epipolar_distance(F, pts1, pts2):
    """
    Computes the symmetric epipolar distance for corresponding points given Fundamental matrices.
    PyTorch GPU implementation.

    Args:
        F: Fundamental matrices with shape (T, 3, 3)
        pts1: First set of points with shape (T, N, 2)
        pts2: Second set of points with shape (T, N, 2)
        
    Returns:
        Epipolar distances with shape (T, N)
    """
    # Ensure inputs are PyTorch tensors
    if not isinstance(F, torch.Tensor):
        F = torch.tensor(F, dtype=torch.float32)
    if not isinstance(pts1, torch.Tensor):
        pts1 = torch.tensor(pts1, dtype=torch.float32)
    if not isinstance(pts2, torch.Tensor):
        pts2 = torch.tensor(pts2, dtype=torch.float32)
    
    # Move tensors to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    F = F.to(device)
    pts1 = pts1.to(device)
    pts2 = pts2.to(device)
    
    T, N = pts1.shape[0], pts1.shape[1]
    
    # Convert to homogeneous coordinates
    ones = torch.ones(T, N, 1, device=device)
    pts1_h = torch.cat([pts1, ones], dim=2)  # (T, N, 3)
    pts2_h = torch.cat([pts2, ones], dim=2)  # (T, N, 3)
    
    # Compute epipolar lines
    # We need to handle batch dimension T
    lines2 = torch.zeros(T, N, 3, device=device)
    lines1 = torch.zeros(T, N, 3, device=device)
    
    for t in range(T):
        lines2[t] = torch.matmul(F[t], pts1_h[t].t()).t()  # in image2
        lines1[t] = torch.matmul(F[t].t(), pts2_h[t].t()).t()  # in image1
    
    # Compute numerator = (d2^T F d1)^2
    num = torch.zeros(T, N, device=device)
    for t in range(T):
        num[t] = torch.sum(pts2_h[t] * lines2[t], dim=1)**2
    
    # Compute denominators = ||line_xy||^2 for each image
    denom1 = lines2[..., 0]**2 + lines2[..., 1]**2  # (T, N)
    denom2 = lines1[..., 0]**2 + lines1[..., 1]**2  # (T, N)
    
    # Avoid divide-by-zero
    eps = 1e-10
    denom1 = torch.clamp(denom1, min=eps)
    denom2 = torch.clamp(denom2, min=eps)
    
    # Compute symmetric epipolar distance
    distances = num / denom1 + num / denom2  # (T, N)
    
    return distances


def compute_seq_sampson_error(cam_extrinsics, traj, camera_K, valid_pts_mask, is_cam_static=False, return_F=False, batch_size=10):
    """
    Computes the Sampson error or Euclidean distance for a given trajectory and camera extrinsics in a batch-wise manner,
    considering only valid points as indicated by the mask. Uses fully vectorized operations for efficiency.

    Args:
        cam_extrinsics (torch.Tensor): Tensor of shape (T, 4, 4) containing camera extrinsics.
        traj (torch.Tensor): Tensor of shape (T, N, 2) containing tracked points.
        camera_K (torch.Tensor): Tensor of shape (3, 3) or (T, 3, 3) containing camera intrinsic matrices.
        valid_pts_mask (torch.Tensor): Tensor of shape (T, N) indicating whether each point is valid at each time.
        is_cam_static (bool): If True, assumes the camera is static and uses Euclidean distance instead of Sampson error.
                             If False, computes Sampson error pair-wisely for all T(T-1)/2 pairs of frames.
        return_F (bool): If True, returns the fundamental matrices as well.
        batch_size (int): Size of batches for pair-wise computation to avoid OOM.

    Returns:
        If is_cam_static:
            torch.Tensor: Error tensor of shape (N,) with median error for each point.
        If not is_cam_static:
            torch.Tensor: Error tensor of shape (N,) with median error for each point.
            torch.Tensor: (Optional if return_F=True) Tensor of shape (T*(T-1)//2, 3, 3) containing fundamental matrices.
    """
    T = traj.shape[0]
    N = traj.shape[1]
    device = traj.device
    
    if is_cam_static:
        # Implementation for static camera - consecutive frames
        # Pre-allocate the result tensor
        sdist = torch.full((T-1, N), float('nan'), device=device)
        
        # Create validity masks for consecutive frames
        valid_pairs = torch.logical_and(valid_pts_mask[:-1], valid_pts_mask[1:])
        
        # Calculate Euclidean distances in one vectorized operation
        diffs = traj[1:] - traj[:-1]  # Shape: (T-1, N, 2)
        euclidean_dists = torch.norm(diffs, dim=2)  # Shape: (T-1, N)
        
        # Apply the validity mask
        sdist = torch.where(valid_pairs, euclidean_dists, torch.tensor(float('nan'), device=device))
        
        # Compute the median error for each point
        sdist_pts = torch.nanmedian(sdist, dim=0).values  # Shape: (N,)
        sdist_pts = torch.nan_to_num(sdist_pts, nan=0.0)  # Replace NaNs with 0.0
        
        return sdist_pts
    else:
        # Generate all pairs (i,j) where i<j using vectorized operations
        indices = torch.triu_indices(T, T, offset=1, device=device)
        i, j = indices[0], indices[1]
        
        num_pairs = i.size(0)
        sdist_list = []
        F_matrices_list = [] if return_F else None
        
        # Process in batches to avoid OOM
        for batch_start in range(0, num_pairs, batch_size):
            batch_end = min(batch_start + batch_size, num_pairs)
            batch_i = i[batch_start:batch_end]
            batch_j = j[batch_start:batch_end]
            
            # Extract camera extrinsics for current batch
            cam1_extrinsics = cam_extrinsics[batch_i]  # (batch_size, 4, 4)
            cam2_extrinsics = cam_extrinsics[batch_j]  # (batch_size, 4, 4)
            
            cam1_intrinsics = camera_K[batch_i]  # (batch_size, 3, 3)
            cam2_intrinsics = camera_K[batch_j]  # (batch_size, 3, 3)
            
            # Compute fundamental matrices for current batch
            batch_F_matrices = compute_fundamental_matrices(cam1_extrinsics, cam2_extrinsics, cam1_intrinsics, cam2_intrinsics)  # (batch_size, 3, 3)
            
            # Extract trajectory points for current batch
            traj1 = traj[batch_i]  # (batch_size, N, 2)
            traj2 = traj[batch_j]  # (batch_size, N, 2)
            
            # Extract validity masks for current batch
            valid_mask1 = valid_pts_mask[batch_i]  # (batch_size, N)
            valid_mask2 = valid_pts_mask[batch_j]  # (batch_size, N)
            valid_pairs = torch.logical_and(valid_mask1, valid_mask2)  # (batch_size, N)
            
            # Compute Sampson distances for current batch
            batch_sampson_dists = compute_sampson_distance_temporal(batch_F_matrices, traj1, traj2)  # (batch_size, N)
            
            # Apply validity mask
            batch_sdist = torch.where(valid_pairs, batch_sampson_dists, torch.tensor(float('nan'), device=device))
            
            # Append results
            sdist_list.append(batch_sdist)
            if return_F:
                F_matrices_list.append(batch_F_matrices)
        
        # Concatenate results from all batches
        sdist = torch.cat(sdist_list, dim=0)  # (num_pairs, N)
        
        # Compute the median error for each point
        sdist_pts = torch.nanmedian(sdist, dim=0).values  # Shape: (N,)
        sdist_pts = torch.nan_to_num(sdist_pts, nan=0.0)  # Replace NaNs with 0.0
        
        if return_F:
            F_matrices = torch.cat(F_matrices_list, dim=0)  # (num_pairs, 3, 3)
            return sdist_pts, F_matrices
        else:
            return sdist_pts


def poisson_disk_sampling_torch_batch(
    t1, t2,  # Already tensors
    v1, v2,  # Already tensors
    threshold,
    batch_size: int = 1024,
    device: str = "cuda"
):
    """
    CUDA-accelerated Poisson Disk Sampling on two views, with chunked processing.
    
    Args:
        t1 (torch.Tensor): (T1, N, 2) points in view 1
        t2 (torch.Tensor): (T2, N, 2) points in view 2
        v1 (torch.Tensor): (T1, N) boolean mask for valid points in view 1
        v2 (torch.Tensor): (T2, N) boolean mask for valid points in view 2
        threshold (float): distance threshold
        batch_size (int): chunk size for processing
        device (str): device to use (no need to move tensors)
    
    Returns:
        np.ndarray: indices of sampled points
    """
    # Check device consistency
    device = t1.device
    assert t2.device == device and v1.device == device and v2.device == device, "All tensors must be on same device"
    
    T1, N, _ = t1.shape
    T2, N2, _ = t2.shape
    assert N == N2, "Both views must have same N"
    
    # --- Compute average pairwise distances per view with batching ---
    def compute_avg_distances_batched(pts, valid_mask):
        # Initialize results
        dist_sum = torch.zeros((N, N), device=device)
        cnt = torch.zeros((N, N), device=device)
        
        # Process each frame
        for t in range(pts.shape[0]):
            idx = torch.nonzero(valid_mask[t], as_tuple=False).view(-1)
            if idx.numel() == 0:
                continue
                
            # Process in batches to avoid OOM
            for i_start in range(0, idx.numel(), batch_size):
                i_end = min(i_start + batch_size, idx.numel())
                i_batch = idx[i_start:i_end]
                
                for j_start in range(0, idx.numel(), batch_size):
                    j_end = min(j_start + batch_size, idx.numel())
                    j_batch = idx[j_start:j_end]
                    
                    # Calculate distances for this sub-batch
                    p_i = pts[t, i_batch]  # (B1, 2)
                    p_j = pts[t, j_batch]  # (B2, 2)
                    
                    # Compute pairwise distances
                    d = torch.norm(p_i[:, None, :] - p_j[None, :, :], dim=2)  # (B1, B2)
                    
                    # Update matrices
                    ii, jj = torch.meshgrid(i_batch, j_batch, indexing='ij')
                    dist_sum[ii, jj] += d
                    cnt[ii, jj] += 1
        
        # Compute averages
        avg = dist_sum / torch.clamp(cnt, min=1)
        avg[cnt == 0] = float('inf')
        return avg, cnt
    
    # Compute average distances for both views
    avg1, cnt1 = compute_avg_distances_batched(t1, v1)
    avg2, cnt2 = compute_avg_distances_batched(t2, v2)
    
    # --- Combine views ---
    total_cnt = cnt1 + cnt2
    combined = (avg1 * cnt1 + avg2 * cnt2) / torch.clamp(total_cnt, min=1)
    combined[total_cnt == 0] = float('inf')
    combined.fill_diagonal_(float('inf'))
    
    # --- Validity score for ordering ---
    valid_score = v1.sum(dim=0) + v2.sum(dim=0)
    order = torch.argsort(-valid_score)  # descending
    
    # --- Selection loop with chunked masking ---
    selected = torch.zeros(N, dtype=torch.bool, device=device)
    available = torch.ones(N, dtype=torch.bool, device=device)
    
    for idx in order:
        if not available[idx]:
            continue
        # pick idx
        selected[idx] = True
        available[idx] = False
        
        # Process in chunks to avoid OOM
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            chunk = slice(start, end)
            # Load one chunk of distances
            dist_chunk = combined[idx, chunk]
            # Mask those too close
            mask_kill = dist_chunk < threshold
            if mask_kill.any():
                available[chunk][mask_kill] = False
    
    return selected.nonzero(as_tuple=False).view(-1).cpu().numpy()

 
def poisson_disk_sampling_torch_batch_fast(
    t1, t2,  # Already tensors
    v1, v2,  # Already tensors
    threshold,
    batch_size: int = 1024,
    device: str = "cuda",
    max_frames_to_sample: int = 10  # Limit frame sampling for speed
):
    """
    Fast approximated Poisson Disk Sampling using batched operations.
    
    Args:
        t1 (torch.Tensor): (T1, N, 2) points in view 1
        t2 (torch.Tensor): (T2, N, 2) points in view 2
        v1 (torch.Tensor): (T1, N) boolean mask for valid points in view 1
        v2 (torch.Tensor): (T2, N) boolean mask for valid points in view 2
        threshold (float): distance threshold
        batch_size (int): chunk size for processing
        device (str): device to use
        max_frames_to_sample (int): maximum number of frames to sample
    
    Returns:
        np.ndarray: indices of sampled points
    """
    T1, N, _ = t1.shape
    T2, N2, _ = t2.shape
    assert N == N2, "Both views must have same N"
    
    # --- Calculate validity score for prioritization ---
    valid_score = v1.sum(dim=0) + v2.sum(dim=0)
    
    # --- Create spatial proximity matrix using sampling ---
    # Sample a limited number of frames for efficiency
    frames_v1 = min(T1, max_frames_to_sample)
    frames_v2 = min(T2, max_frames_to_sample)
    
    sample_indices_v1 = torch.randperm(T1)[:frames_v1]
    sample_indices_v2 = torch.randperm(T2)[:frames_v2]
    
    # Create empty proximity matrix
    proximity = torch.zeros((N, N), dtype=torch.float32, device=device)
    
    # --- Process sampled frames from both views ---
    def process_frames(pts, valid_mask, frame_indices):
        for t_idx in frame_indices:
            # Get valid points in this frame
            valid_idx = torch.nonzero(valid_mask[t_idx], as_tuple=False).view(-1)
            if valid_idx.numel() <= 1:  # Need at least 2 points for comparison
                continue
                
            # Process in batches to avoid OOM
            for i_start in range(0, valid_idx.numel(), batch_size):
                i_end = min(i_start + batch_size, valid_idx.numel())
                i_batch = valid_idx[i_start:i_end]
                points_i = pts[t_idx, i_batch]  # (B, 2)
                
                # Compare against all other valid points in batches
                for j_start in range(0, valid_idx.numel(), batch_size):
                    j_end = min(j_start + batch_size, valid_idx.numel())
                    j_batch = valid_idx[j_start:j_end]
                    points_j = pts[t_idx, j_batch]  # (B, 2)
                    
                    # Compute pairwise distances efficiently
                    # points_i[:, None, :] - (B, 1, 2)
                    # points_j[None, :, :] - (1, B, 2)
                    # Result: (B, B, 2) -> distance (B, B)
                    dists = torch.sqrt(((points_i[:, None, :] - points_j[None, :, :]) ** 2).sum(dim=2))
                    
                    # Create mask for close points (below threshold)
                    close_mask = (dists < threshold) & (dists > 0)  # Exclude self-comparison
                    
                    if close_mask.any():
                        # Map local indices back to global indices
                        rows, cols = torch.nonzero(close_mask, as_tuple=True)
                        global_rows = i_batch[rows]
                        global_cols = j_batch[cols]
                        
                        # Update proximity matrix (add 1 for each close pair found)
                        proximity[global_rows, global_cols] += 1
                        
    # Process frames from both views
    process_frames(t1, v1, sample_indices_v1)
    process_frames(t2, v2, sample_indices_v2)
    
    # Ensure diagonal is excluded (can't be close to itself)
    proximity.fill_diagonal_(0)
    
    # Normalize proximity
    max_val = proximity.max()
    if max_val > 0:
        proximity = proximity / max_val
    
    # --- Create selection score ---
    # Points with high validity and low proximity to other points are preferred
    # Scale validity to [0, 1]
    valid_score_norm = valid_score / max(valid_score.max(), 1)
    
    # Combined score: validity - proximity penalty
    selection_score = valid_score_norm - (proximity.sum(dim=1) * 0.2)
    
    # --- Selection loop with batched operations ---
    selected = torch.zeros(N, dtype=torch.bool, device=device)
    available = torch.ones(N, dtype=torch.bool, device=device)
    
    # Sort points by score for greedy selection
    order = torch.argsort(-selection_score)  # Descending
    
    # Select points in batches for efficiency
    for idx_batch_start in range(0, N, batch_size):
        idx_batch_end = min(idx_batch_start + batch_size, N)
        batch_indices = order[idx_batch_start:idx_batch_end]
        
        # Filter to only available points
        valid_batch = available[batch_indices]
        valid_indices = batch_indices[valid_batch]
        
        if valid_indices.numel() == 0:
            continue
            
        # Mark these points as selected
        selected[valid_indices] = True
        available[valid_indices] = False
        
        # For each selected point, disable its neighbors
        # Process in sub-batches to avoid OOM
        for sub_start in range(0, valid_indices.numel(), 100):
            sub_end = min(sub_start + 100, valid_indices.numel())
            sub_indices = valid_indices[sub_start:sub_end]
            
            # Get proximity scores for these points
            proximity_slice = proximity[sub_indices, :]  # (sub_batch, N)
            
            # Create mask for all points too close to any in this sub-batch
            # A point is too close if its proximity score > threshold
            proximity_threshold = 0.1  # Adjust based on your needs
            too_close_mask = proximity_slice > proximity_threshold
            
            # Combine masks across the sub-batch dimension
            combined_mask = too_close_mask.any(dim=0)
            
            # Update available points
            available = available & ~combined_mask
    
    return selected.nonzero(as_tuple=False).view(-1).cpu().numpy()
  
  
def pareto_filter(indices, errors, overlaps):
    """
    From the candidate indices, filter out those dominated by others:
      j dominates i if:
        errors[j] <= errors[i] and overlaps[j] >= overlaps[i],
        and at least one strict inequality holds.
    """
    pareto = []
    # replace nan overlaps with zero so comparisons are well-defined
    overlaps = np.where(np.isnan(overlaps), 0.0, overlaps)

    for i in indices:
        err_i, ov_i = errors[i], overlaps[i]
        dominated = False
        for j in indices:
            if j == i:
                continue
            err_j, ov_j = errors[j], overlaps[j]
            if (err_j <= err_i and ov_j >= ov_i and
                (err_j < err_i or ov_j > ov_i)):
                dominated = True
                break
        if not dominated:
            pareto.append(i)
    return pareto


def estimate_offset_pareto(offset_error_list, overlap_len_list):
    # used in our finding results
    """
    1) find strict local minima in offset_error_list,
    2) drop any on the boundary or with nan neighbors,
    3) pareto-filter the rest (min error & max overlap),
    4) return None if no survivors or >2, else return the list of 1–2 indices.
    """
    errors   = np.asarray(offset_error_list, dtype=float)
    overlaps = np.asarray(overlap_len_list, dtype=float)
    N = errors.size

    # 1) strict local minima with no nan in the triplet
    # TODO: consider nan because of camera missing
    local_minima = [
        i for i in range(1, N-1)
        if not np.isnan(errors[i-1:i+2]).any()
        and errors[i] < errors[i-1]
        and errors[i] < errors[i+1]
    ]

    if not local_minima:
        return None

    survivors = pareto_filter(local_minima, errors, overlaps)

    # 4) valid only if 1 survivors
    if not survivors or len(survivors) > 1:
        return None
    return survivors


def safe_batch_invert(cam_c2w, cam_valid, cam_K):
    cam_w2c = np.zeros_like(cam_c2w)
    cam_K_new = cam_K.copy()
    new_cam_valid = cam_valid.copy()

    for t in range(cam_c2w.shape[0]):
        if cam_valid[t]:
            try:
                cam_w2c[t] = np.linalg.inv(cam_c2w[t])
            except np.linalg.LinAlgError:
                new_cam_valid[t] = False
                cam_w2c[t] = np.eye(4)
                cam_K_new[t] = np.eye(3)
                continue  # already invalid, skip cam_K checking

            try:
                _ = np.linalg.inv(cam_K[t])  # just try inversion (don't store)
            except np.linalg.LinAlgError:
                new_cam_valid[t] = False
                cam_w2c[t] = np.eye(4)
                cam_K_new[t] = np.eye(3)
        else:
            cam_w2c[t] = np.eye(4)
            cam_K_new[t] = np.eye(3)

    return cam_w2c, new_cam_valid, cam_K_new


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate the synchronization results')
    parser.add_argument('--dataset_root', default='/data11/shaowei3/datasets/egohumans/data2_preprocessed_corrected')
    parser.add_argument('--result_root', default='/data11/shaowei3/datasets/egohumans/data2_preprocessed_corrected_results_v2_improved')
    parser.add_argument('--video1_name', default='fencing_aria01_3_138')
    parser.add_argument("--video2_name", default='fencing_cam02_19_101')
    parser.add_argument('--offset_range', default=51, type=int)
    """
    blender: --offset_range 30
    panoptic: --offset_range 100 
    egohumans: --offset_range 51
    """
    parser.add_argument('--moving_threshold', default=1, type=float, help="moving pixel filter threshold")
    parser.add_argument('--pixel_threshold', default=4, type=float, help="poisson disk sampling threshold")
    parser.add_argument("--max_batch_size", type=int, default=4096, help="max batch size for filtering correspondences")
    parser.add_argument("--max_N", type=int, default=30000, help="max number of correspondences")
    # addtion options
    parser.add_argument('--debug', action='store_true', help="debug mode")
    parser.add_argument("--use_v2", action='store_true', help="use tracks_match v2") # should be true
    
    parser.add_argument("--enable_poisson", action='store_true', help="enable poisson disk sampling") # default is false
    parser.add_argument("--use_weight", action='store_true', help="use weighted error") # default is false
    
    parser.add_argument("--use_vggt", action='store_true', help="use vggt camera pose")
    parser.add_argument("--vggt_choice", default="", type=str, choices=["", "samplev1", "samplev2", "full"], help="vgg sample choice")
    parser.add_argument("--use_hloc", action='store_true', help="use hloc camera pose")
    parser.add_argument("--use_gt_intrinsic", action='store_true', help="use gt intrinsic") # default is false
    parser.add_argument("--use_gt_sample", action='store_true', help="use gt sample")
    
    parser.add_argument("--use_F_list", action='store_true', help="use F_list")
    
    # ablation study
    parser.add_argument("--use_epipolar", action='store_true', help="use epipolar distance")
    parser.add_argument("--disable_gt", action='store_true', help="disable gt")  # used for in-the-wild video
    parser.add_argument("--use_cosine", action='store_true', help="use cosine distance")
    parser.add_argument("--use_algebraic", action='store_true', help="use algebraic distance")
    
    args = parser.parse_args()
    
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    def get_group_prefix(name):
        parts = name.split("_")
        if len(parts) >= 2 and parts[0] == "ID":
            return f"{parts[0]}_{parts[1]}"
        return parts[0]

    prefix = get_group_prefix(args.video1_name)
    assert prefix == get_group_prefix(args.video2_name)
    
    result_dir = os.path.join(args.result_root, prefix, f"{args.video1_name}__{args.video2_name}")
       
    if not args.disable_gt:
        
        video1_start, video1_end = args.video1_name.split('_')[-2:]
        video1_start, video1_end = int(video1_start), int(video1_end)
        video2_start, video2_end = args.video2_name.split('_')[-2:]
        video2_start, video2_end = int(video2_start), int(video2_end)
        gt_offset = video1_start - video2_start
    
    if args.use_v2:
        result_path = os.path.join(result_dir, "tracks_match_v2.npz")
    else:
        result_path = os.path.join(result_dir, "tracks_match.npz")
    
    if not os.path.exists(result_path):
        print(f"no result found for {args.video1_name} and {args.video2_name}, exit")
        exit(0)
        
    result = np.load(result_path,  allow_pickle=True)
        # if "filtered_tracks1" not in result or "filtered_tracks2" not in result:
        #     print(f"no filtered_corr found for {result_path}, exit")
        #     exit(0)
        
    filtered_corr_indices = result["filtered_corr_indices"]
    tracks1 = result["tracks1"].item()
    tracks2 = result["tracks2"].item()
                       
    # 0 is invalid
    pred_tracks1 = tracks1['pred_tracks'] # (T, N, 2)
    pred_valid1 = tracks1['pred_valid'] # (T, N)
    seg_ids1 = tracks1['pred_tracks_seg'] # (N)
        
    pred_tracks2 = tracks2['pred_tracks']
    pred_valid2 = tracks2['pred_valid']
    seg_ids2 = tracks2['pred_tracks_seg'] # (N')
    
    # comment this line as there could be some class missing in one of the video
    # assert np.array_equal(np.unique(seg_ids1[seg_ids1 != 0]), np.unique(seg_ids2[seg_ids2 != 0])), "seg_ids1_list and seg_ids2_list do not have the same class"
        
    # load cross-view correspondences # already saved in filtered_corr_indices
    pred_tracks1 = pred_tracks1[:, filtered_corr_indices[:, 0]]
    pred_tracks2 = pred_tracks2[:, filtered_corr_indices[:, 1]]
    pred_valid1 = pred_valid1[:, filtered_corr_indices[:, 0]]
    pred_valid2 = pred_valid2[:, filtered_corr_indices[:, 1]]
    seg_ids1 = seg_ids1[filtered_corr_indices[:, 0]]
    seg_ids2 = seg_ids2[filtered_corr_indices[:, 1]]
    
    if not (args.use_vggt or args.use_hloc or args.use_gt_sample):
        cam_intrinsic_path1 = os.path.join(args.dataset_root, args.video1_name, "K.npy") # (3, 3)
        cam_intrinsic_path2 = os.path.join(args.dataset_root, args.video2_name, "K.npy") # (3, 3)
            
        cam_K1 = np.load(cam_intrinsic_path1) # (3, 3)
        cam_K2 = np.load(cam_intrinsic_path2) # (3, 3)
            
        cam_extrinsic_path1 = os.path.join(args.dataset_root, args.video1_name, "w2c.npy")
        cam_extrinsic_path2 = os.path.join(args.dataset_root, args.video2_name, "w2c.npy")
                
        cam_extrinsic1 = np.load(cam_extrinsic_path1) # (T, 4, 4)
        cam_extrinsic2 = np.load(cam_extrinsic_path2) # (T', 4, 4)
        
        cam_valid1 = np.ones((pred_valid1.shape[0]), dtype=bool) # (T)
        cam_valid2 = np.ones((pred_valid2.shape[0]), dtype=bool) # (T')
    
    else:
        if args.use_hloc:
            cam_dir_name = "hloc"
            cam_file_path = f"{cam_dir_name}/camera_parameters_no_dyn.npz"
        elif args.use_vggt:
            cam_dir_name = "vggt"
            if args.vggt_choice != "":
                cam_file_path = f"{cam_dir_name}/camera_parameters_{args.vggt_choice}.npz"
            else:
                cam_file_path = f"{cam_dir_name}/camera_parameters.npz"
        elif args.use_gt_sample:
            cam_dir_name = "oracle"
            cam_file_path = f"{cam_dir_name}/camera_parameters_samplev1.npz"
        
        cam_path1 = os.path.join(args.dataset_root, args.video1_name, cam_file_path)
        if args.use_hloc and not os.path.exists(cam_path1):
            cam_path1 = os.path.join(args.dataset_root, args.video1_name, f"{cam_dir_name}/camera_parameters.npz")
        
        cam_result1 = np.load(cam_path1, allow_pickle=True)
        cam_K1 = cam_result1["K"] # (T, 3, 3)
        cam_c2w1 = cam_result1["c2w"] # (T, 4, 4)
        cam_valid1 = cam_result1["valid"] # (T)
        # cam_extrinsic1 = np.linalg.inv(cam_c2w1) # (T, 4, 4)
        cam_extrinsic1, cam_valid1, cam_K1 = safe_batch_invert(cam_c2w1, cam_valid1, cam_K1) # (T, 4, 4)
        
        cam_path2 = os.path.join(args.dataset_root, args.video2_name, cam_file_path)
        if args.use_hloc and not os.path.exists(cam_path2):
            cam_path2 = os.path.join(args.dataset_root, args.video2_name, f"{cam_dir_name}/camera_parameters.npz")
            
        cam_result2 = np.load(cam_path2, allow_pickle=True)
        cam_K2 = cam_result2["K"] # (T, 3, 3)
        cam_c2w2 = cam_result2["c2w"] # (T, 4, 4)
        cam_valid2 = cam_result2["valid"] # (T)
        # cam_extrinsic2 = np.linalg.inv(cam_c2w2) # (T, 4, 4)
        cam_extrinsic2, cam_valid2, cam_K2 = safe_batch_invert(cam_c2w2, cam_valid2, cam_K2) # (T, 4, 4)
        
        if args.use_gt_intrinsic:
            cam_intrinsic_path1 = os.path.join(args.dataset_root, args.video1_name, "K.npy") # (3, 3)
            cam_intrinsic_path2 = os.path.join(args.dataset_root, args.video2_name, "K.npy") # (3, 3)
                
            cam_K1 = np.load(cam_intrinsic_path1) # (3, 3)
            cam_K2 = np.load(cam_intrinsic_path2) # (3, 3)
         
    if not np.all(cam_extrinsic1 == cam_extrinsic1[0]):
        assert pred_tracks1.shape[0] == cam_extrinsic1.shape[0] 
    if not np.all(cam_extrinsic2 == cam_extrinsic2[0]):
        assert pred_tracks2.shape[0] == cam_extrinsic2.shape[0]
   
    if args.use_F_list:
        F_path = os.path.join(result_dir, "F_matches.npz")
        F_results = np.load(F_path, allow_pickle=True)
        F_results_list = F_results["F_list"].item()
        valid_F_results = F_results["valid_list"].item()
                
    # check camera static or dynamic
    is_cam1_static = is_camera_static(cam_extrinsic1, threshold=1e-3)
    is_cam2_static = is_camera_static(cam_extrinsic2, threshold=1e-3)
    
    cam_extrinsic1 = torch.from_numpy(cam_extrinsic1).float().to(device)
    cam_extrinsic2 = torch.from_numpy(cam_extrinsic2).float().to(device)
    pred_tracks1 = torch.from_numpy(pred_tracks1).float().to(device)
    pred_tracks2 = torch.from_numpy(pred_tracks2).float().to(device)
    pred_valid1 = torch.from_numpy(pred_valid1).bool().to(device) # (T, N)
    pred_valid2 = torch.from_numpy(pred_valid2).bool().to(device) # (T, N)
    cam_K1 = torch.from_numpy(cam_K1).float().to(device)
    cam_K2 = torch.from_numpy(cam_K2).float().to(device)
    seg_ids1 = torch.from_numpy(seg_ids1).long().to(device)
    seg_ids2 = torch.from_numpy(seg_ids2).long().to(device)
    
    cam_valid1 = torch.from_numpy(cam_valid1).bool().to(device)
    cam_valid2 = torch.from_numpy(cam_valid2).bool().to(device)
    
    pred_valid1 = torch.logical_and(pred_valid1, cam_valid1.unsqueeze(1)) # (T, N)
    pred_valid2 = torch.logical_and(pred_valid2, cam_valid2.unsqueeze(1)) # (T, N)
    
    
    N = pred_tracks1.shape[1]
    if N > args.max_N:
        rand_indices = torch.randperm(N, device=device)[:args.max_N]
    
        # Select subset of points and validity masks
        pred_tracks1 = pred_tracks1[:, rand_indices]
        pred_valid1 = pred_valid1[:, rand_indices]
        seg_ids1 = seg_ids1[rand_indices]
        pred_tracks2 = pred_tracks2[:, rand_indices]
        pred_valid2 = pred_valid2[:, rand_indices]
        seg_ids2 = seg_ids2[rand_indices]
            
    if args.use_weight:   
        sdist1_list = compute_seq_sampson_error(cam_extrinsic1, pred_tracks1, cam_K1, pred_valid1, is_cam_static=is_cam1_static, batch_size=args.max_batch_size) # (*, N)
        sdist2_list = compute_seq_sampson_error(cam_extrinsic2, pred_tracks2, cam_K2, pred_valid2, is_cam_static=is_cam2_static, batch_size=args.max_batch_size) # (*, N)
        
        # sdist1_pts = torch.nanmedian(sdist1_list, dim=0).values
        # sdist2_pts = torch.nanmedian(sdist2_list, dim=0).values
        # sdist1_pts = torch.nan_to_num(sdist1_pts, nan=0.0)
        # sdist2_pts = torch.nan_to_num(sdist2_pts, nan=0.0)
        sdist_pts = (sdist1_list + sdist2_list) / 2.0
        sdist_pts = (sdist_pts - sdist_pts.min()) / (sdist_pts.max() - sdist_pts.min()) # normalize to [0, 1]
    
    """
    # filter out static class and static points
    for seg_id in np.unique(seg_ids1):
        if seg_id == 0:
            continue
        mask1 = seg_ids1 == seg_id
        mask2 = seg_ids2 == seg_id
        sdist1_seg = sdist1_list[:, mask1] # (*, N)
        sdist2_seg = sdist2_list[:, mask2] # (*, N)
        
        median1 = np.nanmedian(sdist1_seg)
        median2 = np.nanmedian(sdist2_seg)
        
        print("seg id", seg_id, "median1", median1, "median2", median2)
            
        if median1 < args.moving_threshold:
            pred_valid1[:, mask1] = False
            print(f"class {seg_id} in {args.video1_name} is static")
        
        if median2 < args.moving_threshold:
            pred_valid2[:, mask2] = False
            print(f"class {seg_id} in {args.video2_name} is static")
            
        pts1_median = np.nanmedian(sdist1_seg, axis=0) 
        pts2_median = np.nanmedian(sdist2_seg, axis=0)
        
        pred_valid1[:, mask1] = np.logical_and(pred_valid1[:, mask1], pts1_median > args.moving_threshold)
        pred_valid2[:, mask2] = np.logical_and(pred_valid2[:, mask2], pts2_median > args.moving_threshold)
    """
    
    # show remaining correspondences
    # poisson disk sampling
    if args.enable_poisson:
        selected_indices = poisson_disk_sampling_torch_batch_fast(pred_tracks1, pred_tracks2, pred_valid1, pred_valid2, args.pixel_threshold, 
                                                    device=device, batch_size=args.max_batch_size)
        # print("selected indices: ", selected_indices)
        selected_indices = torch.tensor(selected_indices, dtype=torch.long, device=pred_tracks1.device)
        pred_tracks1 = pred_tracks1[:, selected_indices]
        pred_valid1 = pred_valid1[:, selected_indices]
        seg_ids1 = seg_ids1[selected_indices]
        
        pred_tracks2 = pred_tracks2[:, selected_indices]
        pred_valid2 = pred_valid2[:, selected_indices]
        seg_ids2 = seg_ids2[selected_indices]
    
    if args.debug:
        from imgcat import imgcat
        image_dir1 = os.path.join(args.dataset_root, args.video1_name, 'rgb_aligned')
        image_dir2 = os.path.join(args.dataset_root, args.video2_name, 'rgb_aligned')
        
        image_files1 = [f for ext in ["*.jpg", "*.jpeg", "*.png"] for f in glob.glob(os.path.join(image_dir1, ext))]
        image_files1.sort()  
        image_files2 = [f for ext in ["*.jpg", "*.jpeg", "*.png"] for f in glob.glob(os.path.join(image_dir2, ext))]
        image_files2.sort()
        frame1_index = np.argmax(pred_valid1.sum(axis=1))
        frame2_index = np.argmax(pred_valid2.sum(axis=1))
        img1 = iio.imread(image_files1[frame1_index])
        img2 = iio.imread(image_files2[frame2_index])
        valid_mask1 = pred_valid1[frame1_index]
        valid_mask2 = pred_valid2[frame2_index]
        valid_mask = np.logical_and(valid_mask1, valid_mask2)
        track_corr1 = pred_tracks1[frame1_index, valid_mask]
        track_corr2 = pred_tracks2[frame2_index, valid_mask]
        num_valids1 = np.any(pred_valid1, axis=0).astype(np.uint8).sum()
        num_valids2 = np.any(pred_valid2, axis=0).astype(np.uint8).sum()
        print(f"num_valids corr1: {num_valids1}, num_valids corr2: {num_valids2}")
        vis_img = visualize_correspondences(img1, img2, track_corr1, track_corr2, n_viz=20) 
        imgcat(vis_img[:, :, ::-1])
    
    offsets = np.arange(-args.offset_range, args.offset_range) 
    offset_error_list = []
    overlap_len_list = [] 
    for offset in tqdm(offsets):
       
        # find overlapping
        start_idx = max(0, -offset)
        video1_len = len(pred_tracks1)
        video2_len = len(pred_tracks2)

        end_idx = min(video1_len, video2_len - offset)
        
        if start_idx >= end_idx:
            offset_error_list.append(np.nan)
            overlap_len_list.append(0)
            continue

        indices1 = torch.arange(start_idx, end_idx, device=device)
        indices2 = indices1 + offset
        
        sel_tracks1 = pred_tracks1[indices1] # (T, N, 2)
        sel_tracks2 = pred_tracks2[indices2] # (T, N, 2)
        
        extrinsics1 = cam_extrinsic1[indices1] # (T, 4, 4)
        extrinsics2 = cam_extrinsic2[indices2] # (T, 4, 4)
        
        sel_valid1 = pred_valid1[indices1] # (T, N)
        sel_valid2 = pred_valid2[indices2] # (T, N)
        sel_valid = sel_valid1 & sel_valid2 # (T, N)
        
        if len(cam_K1.shape) != 2: # (T< 3, 3)
            intrinsic1 = cam_K1[indices1] # (T, 3, 3)
            intrinsic2 = cam_K2[indices2] # (T, 3, 3)
        else:
            intrinsic1 = cam_K1
            intrinsic2 = cam_K2
        
        if not args.use_F_list:
            F_list = compute_fundamental_matrices(extrinsics1, extrinsics2, intrinsic1, intrinsic2)
        else:
            F_list = []
            valid_list = []
            if len(F_results_list) == 1:
                for key in F_results_list:
                    F_list.append(F_results_list[key])
                    assert valid_F_results[key] == True
                F_list = np.array(F_list).repeat(len(indices1), axis=0)
            else:
                for idx1, idx2 in zip(indices1, indices2):
                    F_list.append(F_results_list[str(idx1.item()) + "_"+ str(idx2.item())])
                    valid_list.append(valid_F_results[str(idx1.item()) + "_"+ str(idx2.item())])
                F_list = np.array(F_list)
                valid_list = np.array(valid_list)
                sel_valid = sel_valid & valid_list[:, None] # (T, N)
            F_list = torch.from_numpy(F_list).float().to(device)
        if args.use_epipolar:
            err_list = compute_symmetric_epipolar_distance(F_list, sel_tracks1, sel_tracks2) # (T, N)
        elif args.use_cosine:
            err_list = compute_cosine_epipolar_error(F_list, sel_tracks1, sel_tracks2)
        elif args.use_algebraic:
            err_list = compute_algebraic_epipolar_error(F_list, sel_tracks1, sel_tracks2)
        else:
            err_list = compute_sampson_distance_temporal(F_list, sel_tracks1, sel_tracks2) # (T, N)
    
        if len(err_list[sel_valid]) == 0:
            err = np.nan
        else:
            if args.use_weight:
                num = (err_list * sel_valid.float() * sdist_pts.unsqueeze(0)).sum() # (T)
                den = (sel_valid.float() * sdist_pts.unsqueeze(0)).sum()
                err = num / den
            else: 
                err = torch.nanmean(err_list[sel_valid])
        
        if isinstance(err, torch.Tensor):
            err = err.item()
        offset_error_list.append(err)
        overlap_len_list.append(len(indices1)) # len(err_list)
    
    offset_error_list = np.array(offset_error_list)
    overlap_len_list = np.array(overlap_len_list)

    assert len(offset_error_list) == len(overlap_len_list)
    assert len(offset_error_list) == len(offsets)
    
    save_dir = result_dir
    if np.all(np.isnan(offset_error_list)):
        pred_offset = None
        print(f"{args.video1_name} and {args.video2_name}: No overlapping found")
    else:
        # max_ol = np.nanmax(overlap_len_list)
        # norm_overlap = overlap_len_list / max_ol
        # weighted_errors = offset_error_list * (2 - norm_overlap) # low overlap -> high weight
    
        # pred_offset = offsets[np.nanargmin(offset_error_list)]
        pred_index = estimate_offset_pareto(offset_error_list, overlap_len_list)
        if pred_index is None:
            pred_offset = None
        else:
            pred_offset = offsets[pred_index[0]]
        
        if not args.disable_gt:
            print(f"{result_dir}: gt offset: {gt_offset}, pred offset {pred_offset}, min energy: {np.min(offset_error_list)}")
        else:
            print(f"{result_dir}: pred offset {pred_offset}, min energy: {np.min(offset_error_list)}")
        
    results = {
        "pred_offset": pred_offset,
        "overlap_len_list": overlap_len_list,
        "offset_error_list": offset_error_list, 
        "offsets": offsets,
    }
    if not args.disable_gt:
        results["gt_offset"] = gt_offset
    
    if not args.use_v2:
        save_path = os.path.join(result_dir, "result_candidates_v3.pkl")
    else:
        save_path = os.path.join(result_dir, "result_candidates_v3_v2.pkl")
    if args.use_weight:
        save_path = save_path.replace(".pkl", "_weighted.pkl")
    if args.use_vggt:
        if args.vggt_choice != "":
            save_path = save_path.replace(".pkl", "_vggt_{}.pkl".format(args.vggt_choice))
        else:
            save_path = save_path.replace(".pkl", "_vggt.pkl")
    if args.use_hloc:
        save_path = save_path.replace(".pkl", "_hloc.pkl")
    if args.use_gt_intrinsic:
        save_path = save_path.replace(".pkl", "_gt_intrinsic.pkl")
    if args.use_F_list:
        save_path = save_path.replace(".pkl", "_F_list.pkl")
    if args.use_gt_sample:
        save_path = save_path.replace(".pkl", "_gt_sample.pkl")
    if args.use_epipolar:
        save_path = save_path.replace(".pkl", "_epipolar.pkl")
    if args.use_cosine:
        save_path = save_path.replace(".pkl", "_cosine.pkl")
    if args.use_algebraic:
        save_path = save_path.replace(".pkl", "_algebraic.pkl")
    with open(save_path, 'wb') as f:
        pickle.dump(results, f)
    
    # Plot only valid errors (NaNs break the line)
    valid = ~np.isnan(offset_error_list)
    plt.figure(figsize=(16, 12))
    # Get the current axes for the error curve
    ax1 = plt.gca()
    # Plot the Sampson error curve
    ax1.plot(np.array(offsets)[valid], offset_error_list[valid]) #  marker='o', color='b', label="original error"
   
    # ax1.plot(np.array(offsets)[valid], weighted_errors[valid], marker='o', color='orange', label="overlap weighted error")
    
    ax1.scatter(np.array(offsets)[~valid], np.zeros(np.sum(~valid)), marker='x', color='b', label="No overlap")

    # Plot predicted offsets as vertical green dashed lines
    # if len(pred_offsets) > 0:
    #     for i, off in enumerate(pred_offsets):
    #         label = "Candidate Offset" if i == 0 else "_nolegend_"
    #         ax1.axvline(x=off, color='orange', linestyle='--', linewidth=1.5, label=label)
    if not args.disable_gt:
        ax1.axvline(x=gt_offset, color='g', linestyle='--', linewidth=2, label="GT Offset")
    if pred_offset is not None:
        ax1.axvline(x=pred_offset, color='r', linestyle='--', linewidth=2, label="Pred Offset")

    ax1.set_xlabel("Offset")
    ax1.set_ylabel("Sequence Error", color='b')
    ax1.tick_params(axis='y', labelcolor='b')

    # Create a second y-axis sharing the same x-axis
    ax2 = ax1.twinx()
    # Plot the overlap length as cyan bars
    bar_width = 1.0 if len(offsets) < 20 else (offsets[1] - offsets[0]) * 0.8
    ax2.bar(offsets, overlap_len_list, width=bar_width, color='cyan', alpha=0.5, label="Overlap Length")

    ax2.set_ylabel("Overlap Length", color='cyan')
    ax2.tick_params(axis='y', labelcolor='cyan')

    # Combine legends from both axes
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right')

    plt.title("Sampson Error and Overlap between Two Sequences")
    plt.tight_layout()
    # Ensure save directory exists
    save_path = os.path.join(save_dir, "offset_energy_v3.png")
    if args.use_weight:
        save_path = save_path.replace(".png", "_weighted.png")
    
    if args.use_vggt:
        if args.vggt_choice != "":
            save_path = save_path.replace(".png", "_vggt_{}.png".format(args.vggt_choice))
        else:
            save_path = save_path.replace(".png", "_vggt.png")
    elif args.use_hloc:
        save_path = save_path.replace(".png", "_hloc.png")
    elif args.use_gt_sample:
        save_path = save_path.replace(".png", "_gt_sample.png")
    elif args.use_F_list:
        save_path = save_path.replace(".png", "_F_list.png")
    
    if args.use_epipolar:
        save_path = save_path.replace(".png", "_epipolar.png")
    if args.use_gt_intrinsic:
        save_path = save_path.replace(".png", "_gt_intrinsic.png")
    if args.use_cosine:
        save_path = save_path.replace(".png", "_cosine.png")
    if args.use_algebraic:
        save_path = save_path.replace(".png", "_algebraic.png")
    # Save the figure
    plt.savefig(save_path)
    plt.close()
    imgcat(open(save_path, 'rb').read())
