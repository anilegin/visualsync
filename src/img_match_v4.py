from mast3r.model import AsymmetricMASt3R
from mast3r.fast_nn import fast_reciprocal_NNs

import mast3r.utils.path_to_dust3r
from mast3r.utils.coarse_to_fine import select_pairs_of_crops
from dust3r.inference import inference
from dust3r.utils.image import load_images
from dust3r.utils.geometry import geotrf

import os
import pickle
import imageio
import numpy as np
import copy
import cv2
from scipy.ndimage import label
import glob
import torch
import PIL.Image
import torchvision.transforms.functional as F
from matplotlib import pyplot as pl
import itertools
from tqdm import tqdm


from match_utils_v2 import post_process_matches, tensor_to_img, get_mask_bbox, filter_segmentation_ids, find_max_pixel_frame, optimal_segmentation_matches

# match across videos v4, not crop each person but also rely on global bg structure
from match_utils import process_image, get_HW_resolution, post_process_matches


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


def batch_dict_to_list_of_dicts(batch_dict):
    """
    Convert a nested dictionary with batch dimension to a list of dictionaries without batch dimension.
    
    Input structure:
    {
        'view1': {key1: tensor(B, *), key2: tensor(B, *)}, # or None
        'view2': {key1: tensor(B, *), key2: tensor(B, *)}, # or None
        'pred1': {key1: tensor(B, *), key2: tensor(B, *)}, # or None
        'pred2': {key1: tensor(B, *), key2: tensor(B, *)}, # or None
        'loss': tensor(B, *) # or None
    }
    
    Output structure:
    [
        {
            'view1': {key1: tensor(*), key2: tensor(*)}, # or None
            'view2': {key1: tensor(*), key2: tensor(*)}, # or None
            'pred1': {key1: tensor(*), key2: tensor(*)}, # or None
            'pred2': {key1: tensor(*), key2: tensor(*)}, # or None
            'loss': tensor(*) # or None
        },
        ...  # B elements in total
    ]
    
    Args:
        batch_dict (dict): Nested dictionary with batch dimension
        
    Returns:
        list: List of dictionaries without batch dimension
    """
    # Determine batch size
    batch_size = None
    for key, value in batch_dict.items():
        if value is not None:
            # If direct tensor with batch dimension (like 'loss')
            if not isinstance(value, dict):
                batch_size = value.shape[0]
                break
            # If nested dictionary
            for inner_key, inner_value in value.items():
                if inner_value is not None:
                    batch_size = inner_value.shape[0]
                    break
            if batch_size is not None:
                break
    
    if batch_size is None:
        return []  # All values were None
    
    # Initialize result list
    result = [{} for _ in range(batch_size)]
    
    # Process each key in the batch dictionary
    for key, value in batch_dict.items():
        if value is None:
            # If value is None, set None for all batch items
            for i in range(batch_size):
                result[i][key] = None
        elif isinstance(value, dict):
            # Handle nested dictionary case
            for i in range(batch_size):
                result[i][key] = {}
                for inner_key, inner_value in value.items():
                    if inner_value is None:
                        result[i][key][inner_key] = None
                    else:
                        # Extract the i-th slice to remove batch dimension
                        result[i][key][inner_key] = inner_value[i]
        else:
            # Handle direct tensor case (like 'loss')
            for i in range(batch_size):
                result[i][key] = value[i]
    
    return result



def filter_segmentation_masks(masks_list):
    """
    Filter segmentation classes from a video segmentation mask list
    with hard-coded thresholds for filtering.
    
    Args:
        masks_list: numpy array of segmentation masks with shape (T, H, W)
        
    Returns:
        masks_list: filtered segmentation masks with invalid IDs set to 0
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
    
    # Set thresholds (assuming dynamic camera)
    min_frame_ratio = 0.3
    min_pixel_count = 100
    
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
        filter_by_components = (frames_with_multiple > 0.3 * appear_frames)
        
        if apply_filter:
            # Filter based on conditions including the "covers too much" condition
            if avg_pixels < min_pixel_count or frame_ratio < min_frame_ratio or filter_by_components or covers_too_much:
                masks_list[masks_list == seg_id] = 0
    
    # Recalculate unique IDs after filtering
    unique_ids = np.unique(masks_list)
    unique_ids = unique_ids[unique_ids > 0]
    print("after filter, number of unique ids: ", len(unique_ids), unique_ids)
    
    # If no IDs remain after filtering, use the one with max pixel count
    if len(unique_ids) == 0 and len(original_unique_ids) > 0:
        # Find segmentation ID with maximum pixel count
        max_pixel_id = max(id_pixel_counts.items(), key=lambda x: x[1])[0]
        print(f"No segments passed filtering. Keeping ID {max_pixel_id} with maximum pixel count.")
        
        # Only keep the max pixel ID in the masks
        filtered_masks = np.zeros_like(masks_list)
        filtered_masks[masks_list == max_pixel_id] = max_pixel_id
        return filtered_masks
    
    return masks_list

if __name__ == '__main__':
    import argparse
    from imgcat import imgcat
    parser = argparse.ArgumentParser()
    # Option-1: provide dataset_root and video1_name, video2_name
    parser.add_argument("--dataset_root", type=str, default="/data11/shaowei3/datasets/egohumans/data2_preprocessed_corrected", help="input dataset root")
    parser.add_argument('--video1_name', type=str, default='volleyball_aria02_16_141')
    parser.add_argument('--video2_name', type=str, default='volleyball_aria04_32_119')
    
    # Option-2: provide video1_dir and video2_dir
    parser.add_argument('--video1_dir', type=str, default=None)
    parser.add_argument('--video2_dir', type=str, default=None)
    
    parser.add_argument("--save_root", type=str, default="results")
    
    parser.add_argument("--padding", type=int, default=10, help="extra pixels to include around the bounding box (if within image boundaries)")
    parser.add_argument("--max_image_size", type=int, default=None, help="max image size for the fine resolution")
    parser.add_argument("--viz_matches", type=int, default=10, help="visualize matches")

    parser.add_argument('--pixel_tol', default=5, type=int)
    
    parser.add_argument("--mask_ratio_threshold", type=float, default=0.8, help="mask ratio threshold for segmentation mask filtering")
    parser.add_argument("--mask_pixel_threshold", type=int, default=100, help="mask pixel threshold for segmentation class filtering")
    parser.add_argument("--min_matches", type=int, default=200, help="minimum number of matches to consider a pair")
    
    parser.add_argument('--interval', help='interval for cotracker', default=10, type=int)
    parser.add_argument('--mask_prefix', default="deva_improved", type=str, help="mask prefix")
    parser.add_argument("--blur_threshold", type=float, default=20.0, help="blurry frame threshold")
    parser.add_argument("--enable_blurry", action="store_true", help="enable blurry frame detection")
    
    parser.add_argument("--batch_size", type=int, default=50, help="batch size for inference")
    parser.add_argument("--vis", action="store_true", help="visualize correspondences")
    
    parser.add_argument("--filter_mask", action="store_true", help="filter masks")
    parser.add_argument("--ignore_mask", action="store_true", help="ignore masks and match full images")
    
    args = parser.parse_args()
    
    # disable coarse to fine , assert pxiel_tol > 0
    device = 'cuda'
    schedule = 'cosine'
    lr = 0.01
    niter = 300
    
    model_name = "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"
    # you can put the path to a local checkpoint in model_name if needed
    model = AsymmetricMASt3R.from_pretrained(model_name).to(device)

    # Compatibility fix:
    # Some MASt3R/DUST3R versions expose patch_size as (16, 16),
    # but match_utils.process_image expects an integer.
    patch_size = model.patch_embed.patch_size
    if isinstance(patch_size, tuple):
        patch_size = patch_size[0]

    fast_nn_params = dict(device=device, dist='dot', block_size=2**13)
  
    if args.video1_dir is not None and args.video2_dir is not None:
        args.video1_dir = args.video1_dir.rstrip('/')
        args.video2_dir = args.video2_dir.rstrip('/')
        
        mask_dir1 = os.path.join(args.video1_dir, args.mask_prefix, "Annotations")
        mask_dir2 = os.path.join(args.video2_dir,  args.mask_prefix, "Annotations")

        image_dir1 = os.path.join(args.video1_dir, "rgb_aligned")
        image_dir2 = os.path.join(args.video2_dir, "rgb_aligned") 
        video1_name = os.path.basename(args.video1_dir)
        video2_name = os.path.basename(args.video2_dir)
        
    else:
       
        mask_dir1 = os.path.join(args.dataset_root, args.video1_name, args.mask_prefix, "Annotations")
        mask_dir2 = os.path.join(args.dataset_root, args.video2_name, args.mask_prefix, "Annotations")
        
        image_dir1 = os.path.join(args.dataset_root, args.video1_name, "rgb_aligned")
        image_dir2 = os.path.join(args.dataset_root, args.video2_name, "rgb_aligned")
        video1_name = args.video1_name
        video2_name = args.video2_name
    
    save_dir = os.path.join(args.save_root, f"{video1_name}__{video2_name}")
    os.makedirs(save_dir, exist_ok=True)
    
    video1_list, video1_files = read_file_from_dir(image_dir1)
    masks1_list, mask1_files = read_file_from_dir(mask_dir1, read_mask=True)
    assert len(video1_list) == len(masks1_list)
    assert video1_list[0].shape[:2] == masks1_list[0].shape[:2] # H, W

    video2_list, video2_files = read_file_from_dir(image_dir2)
    masks2_list, mask2_files = read_file_from_dir(mask_dir2, read_mask=True)
    assert len(video2_list) == len(masks2_list)
    assert video2_list[0].shape[:2] == masks2_list[0].shape[:2] # H, W
    
    videos1 = np.array(video1_list)
    videos2 = np.array(video2_list)
    masks1 = np.array(masks1_list) # (T, H, W)
    masks2 = np.array(masks2_list) # (T, H, W)
    
    if args.filter_mask:
        masks1 = filter_segmentation_masks(masks1)
        masks2 = filter_segmentation_masks(masks2)

    if args.ignore_mask:
        print("ignore_mask=True: using full-image masks for MASt3R matching")
        masks1 = np.ones_like(masks1, dtype=np.uint8)
        masks2 = np.ones_like(masks2, dtype=np.uint8)
    
    if args.enable_blurry:
        is_blurry_list1 = blurry_frame_detect(videos1, threshold=args.blur_threshold)
        is_blurry_list2 = blurry_frame_detect(videos2, threshold=args.blur_threshold)
        valid_ratio1 = is_blurry_list1.sum() / len(is_blurry_list1)
        valid_ratio2 = is_blurry_list2.sum() / len(is_blurry_list2)
    
        if valid_ratio1 > 0.8: # too many blurry frames, disable blurry detection
            is_blurry_list1 = np.array([False] * len(is_blurry_list1), dtype=bool)
        if valid_ratio2 > 0.8: # too many blurry frames, disable blurry detection
            is_blurry_list2 = np.array([False] * len(is_blurry_list2), dtype=bool)
    else:
        is_blurry_list1 = np.array([False] * len(videos1), dtype=bool)
        is_blurry_list2 = np.array([False] * len(videos2), dtype=bool)
    
    sel_frames1 = sample_frames_with_max_valids(masks1, args.interval, is_blurry_list1)
    sel_frames2 = sample_frames_with_max_valids(masks2, args.interval, is_blurry_list2)

    video1_inputs = {}
    video1_full_dict = {}
    for idx, frame_idx in enumerate(sel_frames1):
        view1 = process_image(videos1[frame_idx], maxdim=max(model.patch_embed.img_size), patch_size=patch_size, load_mask=True, mask=masks1[frame_idx])
        W, H = view1['rgb'].size
        video1_full_dict[frame_idx] = view1
        video1_inputs[frame_idx] = {
            'img': view1['rgb_rescaled'].unsqueeze(0),
            # 'img_path': frame_path1,
            # 'mask_path': mask_path1,
            'true_shape': np.int32([view1['rgb_rescaled'].shape[1:]]),
            'to_orig': view1['to_orig'],
            'idx': idx,
            'instance': str(idx),
            'orig_shape': np.int32([H, W])}

    video2_inputs = {}
    video2_full_dict = {}
    for idx, frame_idx in enumerate(sel_frames2): 
        view2 = process_image(videos2[frame_idx], maxdim=max(model.patch_embed.img_size), patch_size=patch_size, load_mask=True, mask=masks2[frame_idx])
        video2_full_dict[frame_idx] = view2
        W, H = view2['rgb'].size
        video2_inputs[frame_idx] = {
            'img': view2['rgb_rescaled'].unsqueeze(0),
            # 'img_path': frame_path2,
            # 'mask_path': mask_path2,
            'true_shape': np.int32([view2['rgb_rescaled'].shape[1:]]),
            'to_orig': view2['to_orig'],
            'idx': len(sel_frames1)+idx,
            'instance': str(len(sel_frames1)+idx),
            'orig_shape': np.int32([H, W])}
       
    pair_frames_indices = list(itertools.product(sel_frames1, sel_frames2))
    pair_frames_indices.sort()
    
    pairs_input = []
    for (frame_idx1, frame_idx2) in pair_frames_indices:
        pairs_input.append((video1_inputs[frame_idx1], video2_inputs[frame_idx2]))
    
    corr_list = []
    for chunk in tqdm(range(0, len(pairs_input), args.batch_size)):
        pairs_chunk = pairs_input[chunk:chunk + args.batch_size]
        outputs = inference(pairs_chunk, model, device, batch_size=len(pairs_chunk), verbose=False)
        outputs = batch_dict_to_list_of_dicts(outputs) # convert to list of dicts
        for pair_idx in range(len(outputs)):
            output = outputs[pair_idx]
            pair_idx = pair_idx + chunk
            frame_idx1, frame_idx2 = pair_frames_indices[pair_idx]
            
            pred1, pred2 = output['pred1'], output['pred2']
            conf_list = [pred1['desc_conf'].squeeze(0).cpu().numpy(), pred2['desc_conf'].squeeze(0).cpu().numpy()]
            desc_list = [pred1['desc'].squeeze(0).detach(), pred2['desc'].squeeze(0).detach()]
            
            view1 = video1_full_dict[frame_idx1]
            view2 = video2_full_dict[frame_idx2]
            
            yM, xM = torch.where(view2['valid_rescaled'])
            P_view1, P_view2 = desc_list[0], desc_list[1]
            dmatches_im_view2, dmatches_im_view1 = fast_reciprocal_NNs(P_view2, P_view1, (xM, yM), pixel_tol=args.pixel_tol, **fast_nn_params)
            dmatches_confs = np.minimum(
                conf_list[1][dmatches_im_view2[:, 1], dmatches_im_view2[:, 0]],
                conf_list[0][dmatches_im_view1[:, 1], dmatches_im_view1[:, 0]]
            )
            dmatches_im_view1 = post_process_matches(dmatches_im_view1, view1)
            dmatches_im_view2 = post_process_matches(dmatches_im_view2, view2)
            
            if len(dmatches_im_view1) == 0 or len(dmatches_im_view2) == 0:
                continue
            
            mask1 = view1['valid'][dmatches_im_view1[:, 1].round().astype(int), dmatches_im_view1[:, 0].round().astype(int)]
            mask2 = view2['valid'][dmatches_im_view2[:, 1].round().astype(int), dmatches_im_view2[:, 0].round().astype(int)]
            mask = mask1 & mask2
            
            mask = np.array(mask, dtype=bool)
            dmatches_im_view1 = dmatches_im_view1[mask]
            dmatches_im_view2 = dmatches_im_view2[mask]
            dmatches_confs = dmatches_confs[mask]
            
            # correspondence dir
            img1_path = os.path.join(image_dir1, video1_files[frame_idx1])
            img2_path = os.path.join(image_dir2, video2_files[frame_idx2])
            mask1_path = os.path.join(mask_dir1, mask1_files[frame_idx1])
            mask2_path = os.path.join(mask_dir2, mask2_files[frame_idx2])
                        
            num_dmatches = dmatches_im_view2.shape[0]
            print(f'found {num_dmatches} dynamic matches')
            
            # store all matches (even 0)
            corr_results = {
                "img1_path": img1_path,
                "img2_path": img2_path,
                "frame_idx1": frame_idx1,
                "frame_idx2": frame_idx2,
                "mask1_path": mask1_path,
                "mask2_path": mask2_path,
                "dmatches_im_view1": dmatches_im_view1,
                "dmatches_im_view2": dmatches_im_view2,
                "dmatches_confs": dmatches_confs
            }
            
            corr_list.append(corr_results)
            
            if args.vis and num_dmatches > 2:
                fig = pl.figure(figsize=(16, 8))
                viz_imgs = [np.array(view1['rgb']), np.array(view2['rgb'])]
                H0, W0, H1, W1 = *viz_imgs[0].shape[:2], *viz_imgs[1].shape[:2]
                img0 = np.pad(viz_imgs[0], ((0, max(H1 - H0, 0)), (0, 0), (0, 0)), 'constant', constant_values=0)
                img1 = np.pad(viz_imgs[1], ((0, max(H0 - H1, 0)), (0, 0), (0, 0)), 'constant', constant_values=0)
                img = np.concatenate((img0, img1), axis=1)

                n_viz = min(args.viz_matches, num_dmatches)
                
                match_idx_to_viz = np.round(np.linspace(0, num_dmatches - 1, n_viz)).astype(int)
                viz_matches_im_view1 = dmatches_im_view1[match_idx_to_viz]
                viz_matches_im_view2 = dmatches_im_view2[match_idx_to_viz]
                ax = fig.add_subplot(111)
                ax.imshow(img)
                cmap = pl.get_cmap('jet')
                for i in range(n_viz):
                    (x0, y0), (x1, y1) = viz_matches_im_view1[i].T, viz_matches_im_view2[i].T
                    ax.plot([x0, x1 + W0], [y0, y1], '-+', color=cmap(i / (n_viz - 1)), scalex=False, scaley=False)
                ax.set_title("dynamic matches")
                ax.axis('off')
                vis_save_dir = os.path.join(save_dir, "vis_corr")
                os.makedirs(vis_save_dir, exist_ok=True)
                img1_name = os.path.basename(img1_path)
                img2_name = os.path.basename(img2_path)
                save_path = os.path.join(vis_save_dir, f"{img1_name}_{img2_name}.png")
                pl.savefig(save_path, bbox_inches="tight")
                # imgcat(open(save_path, "rb").read())
                pl.clf()

    save_path = os.path.join(save_dir, "matches.pkl")
    with open(save_path, "wb") as f:
        pickle.dump(corr_list, f)
        
    print("done!")
