import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import cv2
import torch
import imageio


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


def plot_sampson_error(offsets, offset_error_list, gt_offset, pred_offsets, overlap_len_list, save_dir):
    """
    Plots the Sampson error curve, GT offset, prediction offsets, and overlap lengths.

    Args:
        offsets (list or np.ndarray): X-axis values representing offsets.
        offset_error_list (list or np.ndarray): Sequence Sampson errors corresponding to offsets.
        gt_offset (float): Ground-truth offset value.
        pred_offsets (list or np.ndarray): List of predicted offset candidates.
        overlap_len_list (list or np.ndarray): Overlap lengths for each offset.
        save_dir (str): Directory to save the plot.
    """
    
    plt.figure(figsize=(16, 12))

    # Get the current axes for the error curve
    ax1 = plt.gca()

    # Plot the Sampson error curve
    ax1.plot(offsets, offset_error_list, 'b-o', label="Sequence Sampson Error")

    # Plot the GT offset as a vertical red dashed line
    ax1.axvline(x=gt_offset, color='g', linestyle='--', linewidth=2, label="GT Offset")

    # Plot predicted offsets as vertical green dashed lines
    for i, pred_offset in enumerate(pred_offsets):
        label = "Predicted Offset" if i == 0 else "_nolegend_"
        ax1.axvline(x=pred_offset, color='r', linestyle='--', linewidth=1.5, label=label)

    ax1.set_xlabel("Offset")
    ax1.set_ylabel("Sequence Sampson Error", color='b')
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
    save_path = os.path.join(save_dir, "offset_energy.png")

    # Save the figure
    plt.savefig(save_path)
    plt.close()


def compute_overlap(video1_start, video1_end, video2_start, video2_end):
    """
    Computes the overlapping duration and overlap ratio for two videos.

    Args:
        video1_start (float): Start time of video 1.
        video1_end (float): End time of video 1.
        video2_start (float): Start time of video 2.
        video2_end (float): End time of video 2.

    Returns:
        tuple: (overlap_length, overlap_ratio_video1, overlap_ratio_video2)
    """
    # Compute overlap
    overlap_start = max(video1_start, video2_start)
    overlap_end = min(video1_end, video2_end)
    
    # Calculate the overlapping length
    overlap_length = max(0, overlap_end - overlap_start)

    # Compute video durations
    video1_length = video1_end - video1_start
    video2_length = video2_end - video2_start

    # Compute overlap ratios
    overlap_ratio_video1 = overlap_length / video1_length if video1_length > 0 else 0
    overlap_ratio_video2 = overlap_length / video2_length if video2_length > 0 else 0

    return overlap_length, overlap_ratio_video1, overlap_ratio_video2


# reference: co-tracker/filter_utils.py:compute_seq_sampson_error_from_F_v2
def compute_seq_sampson_error(cam_extrinsics, traj, camera_K, valid_pts_mask, is_cam_static=False, return_F=False):
    """
    Computes the Sampson error or Euclidean distance for a given trajectory and camera extrinsics,
    considering only valid points as indicated by the mask. Uses fully vectorized operations for efficiency.

    Args:
        cam_extrinsics (np.ndarray): Array of shape (T, 4, 4) containing camera extrinsics.
        traj (np.ndarray): Array of shape (T, N, 2) containing tracked points.
        camera_K (np.ndarray): Array of shape (3, 3) containing camera intrinsic matrices.
        valid_pts_mask (np.ndarray): Array of shape (T, N) indicating whether each point is valid at each time.
        is_cam_static (bool): If True, assumes the camera is static and uses Euclidean distance instead of Sampson error.
                             If False, computes Sampson error pair-wisely for all T(T-1)/2 pairs of frames.
        return_F (bool): If True, returns the fundamental matrices as well.

    Returns:
        If is_cam_static:
            np.ndarray: Error array of shape (T-1, N) with invalid points set to NaN.
        If not is_cam_static:
            np.ndarray: Error array of shape (T*(T-1)//2, N) with invalid points set to NaN.
            np.ndarray: (Optional if return_F=True) Array of shape (T*(T-1)//2, 3, 3) containing fundamental matrices.
    """
    T = traj.shape[0]
    N = traj.shape[1]
    
    if is_cam_static:
        # Implementation for static camera - consecutive frames
        # Pre-allocate the result array
        sdist = np.full((T-1, N), np.nan)
        
        # Create validity masks for consecutive frames
        valid_pairs = np.logical_and(valid_pts_mask[:-1], valid_pts_mask[1:])
        
        # Calculate Euclidean distances in one vectorized operation
        diffs = traj[1:] - traj[:-1]  # Shape: (T-1, N, 2)
        euclidean_dists = np.linalg.norm(diffs, axis=2)  # Shape: (T-1, N)
        
        # Apply the validity mask
        sdist = np.where(valid_pairs, euclidean_dists, np.nan)
        
        return sdist
    else:
        # Generate all pairs (i,j) where i<j using vectorized operations
        # Create meshgrid of all possible i,j combinations
        i, j = np.triu_indices(T, k=1)
        
        # Extract camera extrinsics for all pairs
        cam1_extrinsics = cam_extrinsics[i]  # (num_pairs, 4, 4)
        cam2_extrinsics = cam_extrinsics[j]  # (num_pairs, 4, 4)
        
        # Compute fundamental matrices for all pairs at once
        F_matrices = compute_fundamental_matrices(cam1_extrinsics, cam2_extrinsics, camera_K, camera_K)  # (num_pairs, 3, 3)
        
        # Extract trajectory points for all pairs
        traj1 = traj[i]  # (num_pairs, N, 2)
        traj2 = traj[j]  # (num_pairs, N, 2)
        
        # Extract validity masks for all pairs
        valid_mask1 = valid_pts_mask[i]  # (num_pairs, N)
        valid_mask2 = valid_pts_mask[j]  # (num_pairs, N)
        valid_pairs = np.logical_and(valid_mask1, valid_mask2)  # (num_pairs, N)
        
        # Compute Sampson distances for all pairs at once
        sampson_dists = compute_sampson_distance_temporal(F_matrices, traj1, traj2)  # (num_pairs, N)
        
        # Apply validity mask
        sdist = np.where(valid_pairs, sampson_dists, np.nan)
        
        if return_F:
            return sdist, F_matrices
        else:
            return sdist
        
        
        
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
    
    # Add homogeneous coordinates (T, N, 3)
    pts1_h = np.concatenate([pts1, np.ones((T, N, 1))], axis=2)
    pts2_h = np.concatenate([pts2, np.ones((T, N, 1))], axis=2)
    
    # Compute F*x1 for each time step
    # Reshape F to (T, 3, 3) and pts1_h to (T, N, 3)
    # Use einsum for batch matrix multiplication
    Fx1 = np.einsum('tij,tnj->tni', F, pts1_h)  # (T, N, 3)
    
    # Compute F^T*x2 for each time step
    Ftx2 = np.einsum('tji,tnj->tni', F, pts2_h)  # (T, N, 3)
    
    # Compute (x2^T F x1) using batch dot product
    x2Fx1 = np.sum(pts2_h * Fx1, axis=2)  # (T, N)
    num = x2Fx1 ** 2  # (T, N)
    
    # Compute denominator
    denom = Fx1[..., 0] ** 2 + Fx1[..., 1] ** 2 + Ftx2[..., 0] ** 2 + Ftx2[..., 1] ** 2  # (T, N)
    
    # Handle numerical stability
    denom = np.maximum(denom, 1e-10)
    
    # Compute Sampson distances
    sampson_distances = num / denom  # (T, N)
    
    return sampson_distances


def skew_symmetric_batch(vectors):
    """
    Compute a batch of skew-symmetric matrices for a set of 3D vectors.
    
    Parameters:
        vectors: numpy array of shape (T, 3)
    
    Returns:
        skew: numpy array of shape (T, 3, 3) where each 3x3 matrix is the skew-symmetric
              matrix of the corresponding vector.
    """
    T = vectors.shape[0]
    skew = np.zeros((T, 3, 3))
    skew[:, 0, 1] = -vectors[:, 2]
    skew[:, 0, 2] =  vectors[:, 1]
    skew[:, 1, 0] =  vectors[:, 2]
    skew[:, 1, 2] = -vectors[:, 0]
    skew[:, 2, 0] = -vectors[:, 1]
    skew[:, 2, 1] =  vectors[:, 0]
    return skew


def compute_fundamental_matrices(video1_w2c, video2_w2c, video1_K, video2_K):
    """
    Compute the fundamental matrix for each corresponding frame pair in two videos in batch mode.
    
    Both video1_w2c and video2_w2c are expected to be numpy arrays of shape (T, 4, 4) 
    representing the world-to-camera transformation for T frames.
    
    The camera intrinsics for video 1 and video 2 are given by video1_K and video2_K,
    which are numpy arrays of shape (3, 3).
    
    Returns:
        F_all: numpy array of shape (T, 3, 3) containing the fundamental matrices for each frame.
    """
    # Ensure the two videos have the same number of frames.
    T1 = video1_w2c.shape[0]
    T2 = video2_w2c.shape[0]
    if T1 != T2:
        raise ValueError("The number of frames in video1_w2c and video2_w2c must be the same.")
    
    # Extract rotation (upper left 3x3) and translation (first three elements of last column) for each frame.
    R1 = video1_w2c[:, :3, :3]  # Shape: (T, 3, 3)
    t1 = video1_w2c[:, :3, 3]   # Shape: (T, 3)
    R2 = video2_w2c[:, :3, :3]
    t2 = video2_w2c[:, :3, 3]
    
    # Compute the relative rotation R_rel = R2 * R1^T for each frame.
    R1_T = np.transpose(R1, (0, 2, 1))  # Transpose each frame’s rotation matrix.
    R_rel = np.matmul(R2, R1_T)           # Shape: (T, 3, 3)
    
    # Compute the relative translation:
    # For each frame, t_rel = t2 - (R_rel @ t1)
    # Use einsum for batched matrix-vector multiplication.
    t1_transformed = np.einsum('tij,tj->ti', R_rel, t1)
    t_rel = t2 - t1_transformed           # Shape: (T, 3)
    
    # Generate the skew-symmetric matrices for the relative translations.
    t_skew = skew_symmetric_batch(t_rel)  # Shape: (T, 3, 3)
    
    # Compute the Essential matrix for each frame: E = [t_rel]_x * R_rel.
    E = np.matmul(t_skew, R_rel)          # Shape: (T, 3, 3)
    
    # Precompute the inverses of the camera intrinsic matrices.
    invK1 = np.linalg.inv(video1_K)
    invK2 = np.linalg.inv(video2_K)
    
    # Compute the Fundamental matrix for each frame:
    # F = inv(video2_K).T * E * inv(video1_K)
    # Note: Since invK2.T and invK1 are constant (shape 3x3), they will be broadcasted across the batch dimension.
    F_all = np.matmul(np.matmul(invK2.T, E), invK1)
    
    # Normalize the Fundamental matrix in each frame by its bottom-right element.
    # F_all = F_all / F_all[:, 2, 2][:, None, None]
    
    return F_all


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



def poisson_disk_sampling(pts_view1, pts_view2, valid_view1, valid_view2, threshold):
    """
    Perform Poisson Disk Sampling on points in two views with different time lengths.
    
    Args:
        pts_view1 (numpy.ndarray): Points in first view, shape (T1, N, 2)
        pts_view2 (numpy.ndarray): Points in second view, shape (T2, N, 2)
        valid_view1 (numpy.ndarray): Validity of points in first view, shape (T1, N)
        valid_view2 (numpy.ndarray): Validity of points in second view, shape (T2, N)
        threshold (float): Distance threshold for removing points
        
    Returns:
        numpy.ndarray: Indices of sampled points, shape (N',)
    """
    T1, N, _ = pts_view1.shape
    T2, N_check, _ = pts_view2.shape
    
    # Ensure both views have the same number of points
    assert N == N_check, "Both views must have the same number of points (N)"
    
    # Precompute the validity sum for each point across all timesteps in both views
    # The total validity is the sum of valid timesteps in view1 and view2
    validity_view1 = np.sum(valid_view1, axis=0)  # Shape: (N,)
    validity_view2 = np.sum(valid_view2, axis=0)  # Shape: (N,)
    total_validity = validity_view1 + validity_view2  # Shape: (N,)
    
    # Initialize distance matrices
    # We'll calculate average distances for each view separately, then combine
    avg_distances_view1 = np.zeros((N, N))
    valid_pairs_view1 = np.zeros((N, N))
    
    avg_distances_view2 = np.zeros((N, N))
    valid_pairs_view2 = np.zeros((N, N))
    
    # Calculate pairwise distances for view 1
    for t in range(T1):
        # Create a mask of valid points at this timestep
        valid_t = valid_view1[t]  # Shape: (N,)
        
        # Get valid indices
        valid_indices = np.where(valid_t)[0]
        
        if len(valid_indices) > 0:
            # Extract valid points at this timestep
            pts_valid = pts_view1[t, valid_indices]  # Shape: (n_valid, 2)
            
            # Compute pairwise distances
            diff = pts_valid[:, np.newaxis, :] - pts_valid[np.newaxis, :, :]  # Shape: (n_valid, n_valid, 2)
            dist = np.sqrt(np.sum(diff**2, axis=2))  # Shape: (n_valid, n_valid)
            
            # Update distance matrix for view 1
            for i, idx_i in enumerate(valid_indices):
                for j, idx_j in enumerate(valid_indices):
                    avg_distances_view1[idx_i, idx_j] += dist[i, j]
                    valid_pairs_view1[idx_i, idx_j] += 1
    
    # Calculate pairwise distances for view 2
    for t in range(T2):
        # Create a mask of valid points at this timestep
        valid_t = valid_view2[t]  # Shape: (N,)
        
        # Get valid indices
        valid_indices = np.where(valid_t)[0]
        
        if len(valid_indices) > 0:
            # Extract valid points at this timestep
            pts_valid = pts_view2[t, valid_indices]  # Shape: (n_valid, 2)
            
            # Compute pairwise distances
            diff = pts_valid[:, np.newaxis, :] - pts_valid[np.newaxis, :, :]  # Shape: (n_valid, n_valid, 2)
            dist = np.sqrt(np.sum(diff**2, axis=2))  # Shape: (n_valid, n_valid)
            
            # Update distance matrix for view 2
            for i, idx_i in enumerate(valid_indices):
                for j, idx_j in enumerate(valid_indices):
                    avg_distances_view2[idx_i, idx_j] += dist[i, j]
                    valid_pairs_view2[idx_i, idx_j] += 1
    
    # Compute final average distances for each view
    mask1 = valid_pairs_view1 > 0
    avg_distances_view1[mask1] /= valid_pairs_view1[mask1]
    avg_distances_view1[~mask1] = float('inf')
    
    mask2 = valid_pairs_view2 > 0
    avg_distances_view2[mask2] /= valid_pairs_view2[mask2]
    avg_distances_view2[~mask2] = float('inf')
    
    # Combine distances from both views
    # For each pair of points, if they're valid in both views, average the distances
    # If they're only valid in one view, use that distance
    # If they're not valid in either view, set to infinity
    combined_valid_pairs = valid_pairs_view1 + valid_pairs_view2
    combined_distances = avg_distances_view1 * valid_pairs_view1 + avg_distances_view2 * valid_pairs_view2
    
    # Avoid division by zero
    mask_combined = combined_valid_pairs > 0
    combined_distances[mask_combined] /= combined_valid_pairs[mask_combined]
    combined_distances[~mask_combined] = float('inf')
    
    # Set self-distances to infinity to avoid selecting the same point
    np.fill_diagonal(combined_distances, float('inf'))
    
    # Initialize array to track selected points
    selected = np.zeros(N, dtype=bool)
    available = np.ones(N, dtype=bool)
    
    # Sort indices by total validity count (descending)
    sorted_indices = np.argsort(-total_validity)
    
    while np.any(available):
        # Find the available point with the highest validity
        available_sorted = sorted_indices[available[sorted_indices]]
        
        if len(available_sorted) == 0:
            break
            
        # Select the first available point (highest validity)
        current_idx = available_sorted[0]
        selected[current_idx] = True
        available[current_idx] = False
        
        # Remove points that are too close to the selected point
        too_close = combined_distances[current_idx] < threshold
        available[too_close] = False
    
    return np.where(selected)[0]



def poisson_disk_sampling_torch(pts_view1, pts_view2, valid_view1, valid_view2, threshold, device="cuda"):
    """
    CUDA-accelerated implementation of Poisson Disk Sampling using PyTorch.
    
    Args:
        pts_view1 (numpy.ndarray): Points in first view, shape (T1, N, 2)
        pts_view2 (numpy.ndarray): Points in second view, shape (T2, N, 2)
        valid_view1 (numpy.ndarray): Validity of points in first view, shape (T1, N)
        valid_view2 (numpy.ndarray): Validity of points in second view, shape (T2, N)
        threshold (float): Distance threshold for removing points
        device (str): PyTorch device to use (default: "cuda")
        
    Returns:
        numpy.ndarray: Indices of sampled points, shape (N',)
    """
    # Check if CUDA is available and set device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = "cpu"
    
    device = torch.device(device)
    
    # Move inputs to torch tensors on specified device
    t_pts_view1 = torch.tensor(pts_view1, dtype=torch.float32, device=device)
    t_pts_view2 = torch.tensor(pts_view2, dtype=torch.float32, device=device)
    t_valid_view1 = torch.tensor(valid_view1, dtype=torch.bool, device=device)
    t_valid_view2 = torch.tensor(valid_view2, dtype=torch.bool, device=device)
    
    T1, N, _ = t_pts_view1.shape
    T2, N_check, _ = t_pts_view2.shape
    
    # Ensure both views have the same number of points
    assert N == N_check, "Both views must have the same number of points (N)"
    
    # Precompute validity sums
    validity_view1 = torch.sum(t_valid_view1.float(), dim=0)  # Shape: (N,)
    validity_view2 = torch.sum(t_valid_view2.float(), dim=0)  # Shape: (N,)
    total_validity = validity_view1 + validity_view2  # Shape: (N,)
    
    # Initialize distance and valid pair tensors
    avg_distances_view1 = torch.zeros((N, N), dtype=torch.float32, device=device)
    valid_pairs_view1 = torch.zeros((N, N), dtype=torch.float32, device=device)
    avg_distances_view2 = torch.zeros((N, N), dtype=torch.float32, device=device)
    valid_pairs_view2 = torch.zeros((N, N), dtype=torch.float32, device=device)
    
    # Process view 1
    for t in range(T1):
        valid_indices = torch.where(t_valid_view1[t])[0]
        if len(valid_indices) > 0:
            pts_valid = t_pts_view1[t, valid_indices]
            
            # Efficient pairwise distance calculation using PyTorch broadcasting
            diffs = pts_valid.unsqueeze(1) - pts_valid.unsqueeze(0)
            dists = torch.sqrt(torch.sum(diffs**2, dim=2))
            
            # Create indices grid using torch.meshgrid
            i_indices, j_indices = torch.meshgrid(valid_indices, valid_indices, indexing='ij')
            
            # Update distance matrices 
            avg_distances_view1[i_indices, j_indices] += dists
            valid_pairs_view1[i_indices, j_indices] += 1
    
    # Process view 2
    for t in range(T2):
        valid_indices = torch.where(t_valid_view2[t])[0]
        if len(valid_indices) > 0:
            pts_valid = t_pts_view2[t, valid_indices]
            
            # Efficient pairwise distance calculation using PyTorch broadcasting
            diffs = pts_valid.unsqueeze(1) - pts_valid.unsqueeze(0)
            dists = torch.sqrt(torch.sum(diffs**2, dim=2))
            
            # Create indices grid using torch.meshgrid
            i_indices, j_indices = torch.meshgrid(valid_indices, valid_indices, indexing='ij')
            
            # Update distance matrices
            avg_distances_view2[i_indices, j_indices] += dists
            valid_pairs_view2[i_indices, j_indices] += 1
    
    # Compute average distances
    mask1 = valid_pairs_view1 > 0
    avg_distances_view1[mask1] /= valid_pairs_view1[mask1]
    avg_distances_view1[~mask1] = float('inf')
    
    mask2 = valid_pairs_view2 > 0
    avg_distances_view2[mask2] /= valid_pairs_view2[mask2]
    avg_distances_view2[~mask2] = float('inf')
    
    # Combine distances from both views
    combined_valid_pairs = valid_pairs_view1 + valid_pairs_view2
    combined_distances = avg_distances_view1 * valid_pairs_view1 + avg_distances_view2 * valid_pairs_view2
    
    mask_combined = combined_valid_pairs > 0
    combined_distances[mask_combined] /= combined_valid_pairs[mask_combined]
    combined_distances[~mask_combined] = float('inf')
    
    # Set self-distances to infinity
    combined_distances.fill_diagonal_(float('inf'))
    
    # Initialize selection arrays
    selected = torch.zeros(N, dtype=torch.bool, device=device)
    available = torch.ones(N, dtype=torch.bool, device=device)
    
    # Sort indices by total validity count (descending) - bring to CPU for sorting
    sorted_indices = torch.argsort(-total_validity)
    
    # Point selection loop
    while torch.any(available):
        # Get available indices sorted by validity
        available_sorted = sorted_indices[available[sorted_indices]]
        
        if len(available_sorted) == 0:
            break
        
        # Select the highest validity point
        current_idx = available_sorted[0]
        selected[current_idx] = True
        available[current_idx] = False
        
        # Efficiently identify points that are too close
        too_close = combined_distances[current_idx] < threshold
        available[too_close] = False
    
    # Convert result back to numpy array
    result = torch.where(selected)[0].cpu().numpy()
    
    return result


def poisson_disk_sampling_torch_batch(
    pts_view1, pts_view2,
    valid_view1, valid_view2,
    threshold,
    batch_size: int = 1024,
    device: str = "cuda"
):
    """
    CUDA-accelerated Poisson Disk Sampling on two views, with chunked masking.

    Args:
        pts_view1 (np.ndarray): (T1, N, 2)
        pts_view2 (np.ndarray): (T2, N, 2)
        valid_view1 (np.ndarray): (T1, N) bool
        valid_view2 (np.ndarray): (T2, N) bool
        threshold (float): distance threshold
        batch_size (int): chunk size for mask updates
        device (str): "cuda" or "cpu"

    Returns:
        np.ndarray: indices of sampled points
    """
    # --- Setup device & tensors ---
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    device = torch.device(device)
    
    t1 = torch.tensor(pts_view1, dtype=torch.float32, device=device)
    t2 = torch.tensor(pts_view2, dtype=torch.float32, device=device)
    v1 = torch.tensor(valid_view1, dtype=torch.bool, device=device)
    v2 = torch.tensor(valid_view2, dtype=torch.bool, device=device)

    T1, N, _ = t1.shape
    T2, N2, _ = t2.shape
    assert N == N2, "Both views must have same N"

    # --- Compute average pairwise distances per view ---
    def compute_avg_distances(pts, valid_mask):
        # returns (N,N) with inf where never valid
        dist_sum = torch.zeros((N, N), device=device)
        cnt = torch.zeros((N, N), device=device)
        for t in range(pts.shape[0]):
            idx = torch.nonzero(valid_mask[t], as_tuple=False).view(-1)
            if idx.numel() == 0:
                continue
            p = pts[t, idx]  # (M,2)
            # broadcasted pairwise
            d = torch.norm(p[:, None, :] - p[None, :, :], dim=2)  # (M,M)
            ii, jj = torch.meshgrid(idx, idx, indexing='ij')
            dist_sum[ii, jj] += d
            cnt[ii, jj] += 1
        avg = dist_sum / torch.clamp(cnt, min=1)
        avg[cnt == 0] = float('inf')
        return avg, cnt

    avg1, cnt1 = compute_avg_distances(t1, v1)
    avg2, cnt2 = compute_avg_distances(t2, v2)

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

        # kill neighbors in chunks
        # we only need to check combined[idx, :]
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            chunk = slice(start, end)
            # load one chunk of distances
            dist_chunk = combined[idx, chunk]
            # mask those too close
            mask_kill = dist_chunk < threshold
            if mask_kill.any():
                available[chunk][mask_kill] = False

    return selected.nonzero(as_tuple=False).view(-1).cpu().numpy()
