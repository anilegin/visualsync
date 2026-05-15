import os
import glob
import cv2
import time
import numpy as np
import pickle
from itertools import combinations
from tqdm import tqdm
import argparse
import itertools
import subprocess
from collections import defaultdict


def group_folders_by_prefix(folders):
    groups = defaultdict(list)
    for folder in folders:
        name = os.path.basename(folder)

        # PRIN names: ID_0_fpv_000_200, ID_0_cam_tpv_000_200
        parts = name.split("_")
        if len(parts) >= 2 and parts[0] == "ID":
            prefix = f"{parts[0]}_{parts[1]}"
        else:
            prefix = parts[0]

        groups[prefix].append(folder)
    return groups


def is_pickle_valid(filepath):
    try:
        with open(filepath, 'rb') as f:
            result = pickle.load(f)
            if "offset_error_list" in result:
                if not (np.isnan(result["offset_error_list"]).all()==True):
                    return True
            return True
    except Exception as e:
        print(f"Error loading pickle file: {e}")
        return False


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument("--dataset_root", type=str, default="/data11/shaowei3/datasets/egohumans/data2_preprocessed_corrected", help="input dataset root")
    parser.add_argument("--result_root", type=str, default="/data11/shaowei3/datasets/egohumans/data2_preprocessed_corrected_results_v2_improved", help="result root")
    parser.add_argument('--offset_range', default=51, type=int)
    """
    blender: --offset_range 30
    panoptic: --offset_range 100 
    egohumans: --offset_range 51
    """
    
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=-1)
    
    # addtional arguments
    parser.add_argument('--moving_threshold', default=1, type=float, help="moving pixel filter threshold")
    parser.add_argument('--pixel_threshold', default=4, type=float, help="poisson disk sampling threshold")
    parser.add_argument("--max_batch_size", type=int, default=1024, help="max batch size for filtering correspondences")
    parser.add_argument("--max_N", type=int, default=30000, help="max number of correspondences")
    
    parser.add_argument("--skip_exist", action="store_true", help="skip existing results")
    
    parser.add_argument("--enable_poisson", action='store_true', help="enable poisson disk sampling")
    parser.add_argument("--use_weight", action='store_true', help="use weighted error")
    parser.add_argument("--not_use_v2", action='store_true', help="use v1 result")
    
    parser.add_argument("--use_vggt", action='store_true', help="use vggt for camera pose")
    parser.add_argument("--vggt_choice", default="", type=str, choices=["", "samplev1", "samplev2", "full"], help="vgg sample choice")
    parser.add_argument("--use_hloc", action='store_true', help="use hloc camera pose")
    parser.add_argument("--use_gt_intrinsic", action='store_true', help="use gt intrinsic") # default is false
    parser.add_argument("--use_gt_sample", action='store_true', help="use gt sample")
    
    parser.add_argument("--use_F_list", action='store_true', help="use F_list")
    parser.add_argument("--group_name", type=str, default=None, help="group name")
    parser.add_argument("--use_epipolar", action='store_true', help="use epipolar distance")
    parser.add_argument("--disable_gt", action='store_true', help="disable gt")  # used for in-the-wild video
    parser.add_argument("--use_cosine", action='store_true', help="use cosine distance")  # used for in-the-wild video
    parser.add_argument("--use_algebraic", action='store_true', help="use algebraic distance")
    
    args = parser.parse_args()
    
    code_dir = os.path.dirname(os.path.abspath(__file__))
    
    log_file = os.path.join(args.result_root, "sync_v6.log")
    
    video_dirs = glob.glob(os.path.join(args.dataset_root, "*/"))
    video_dirs = [video.rsplit('/', 1)[0] for video in video_dirs if os.path.isdir(video) and '*' not in video]
    
    groups = group_folders_by_prefix(video_dirs)
    
    pairs = []
    for prefix, folders in groups.items():
        if args.group_name is not None and prefix != args.group_name:
            continue
        print("prefix", prefix, "num_folders", len(folders))
        folders.sort()
        pairs_subgroup = list(combinations(folders, 2))
        pairs.extend(pairs_subgroup)
    
    print(f"total {len(pairs)} pairs")

    # PRIN / robust pipeline fix:
    # Only schedule pairs that already have filtered track correspondences.
    # Otherwise job_sync tries to run ID folders whose previous stages failed or were never run.
    valid_ready_pairs = []
    for video1, video2 in pairs:
        video1_name = os.path.basename(video1)
        video2_name = os.path.basename(video2)

        parts = video1_name.split("_")
        if len(parts) >= 2 and parts[0] == "ID":
            prefix = f"{parts[0]}_{parts[1]}"
        else:
            prefix = video1_name.split("_")[0]

        pair_dir = os.path.join(args.result_root, prefix, video1_name + "__" + video2_name)
        tracks_match_path = os.path.join(pair_dir, "tracks_match_v2.npz")

        if os.path.exists(tracks_match_path):
            valid_ready_pairs.append((video1, video2))
        else:
            print(f"[SKIP NO TRACK MATCH] {tracks_match_path}")

    pairs = valid_ready_pairs
    print(f"total {len(pairs)} ready pairs with tracks_match_v2.npz")
    
    if args.skip_exist:
        valid_pairs = []
        for video1, video2 in tqdm(pairs):
            video1_name = os.path.basename(video1)
            video2_name = os.path.basename(video2)
            
            prefix = video1_name.split("_")[0]
            result_dir = os.path.join(args.result_root, prefix, video1_name + "__" + video2_name)
            if not args.not_use_v2:
                if args.use_vggt:
                    if args.vggt_choice != "":
                        result_path = os.path.join(result_dir, "result_candidates_v3_v2_vggt_{}.pkl".format(args.vggt_choice))
                    else:
                        result_path = os.path.join(result_dir, "result_candidates_v3_v2_vggt.pkl")
                elif args.use_hloc:
                    result_path = os.path.join(result_dir, "result_candidates_v3_v2_hloc.pkl")
                elif args.use_gt_sample:
                    result_path = os.path.join(result_dir, "result_candidates_v3_v2_gt_sample.pkl")
                elif args.use_F_list:
                    result_path = os.path.join(result_dir, "result_candidates_v3_v2_F_list.pkl")
                else:
                    result_path = os.path.join(result_dir, "result_candidates_v3_v2.pkl")
                if args.use_epipolar:
                    result_path = result_path.replace(".pkl", "_epipolar.pkl")
                if args.use_cosine:
                    result_path = result_path.replace(".pkl", "_cosine.pkl")
                if args.use_algebraic:
                    result_path = result_path.replace(".pkl", "_algebraic.pkl")
                if args.use_gt_intrinsic:
                        result_path = result_path.replace(".pkl", "_gt_intrinsic.pkl")
            else:
                result_path = os.path.join(result_dir, "result_candidates_v3.pkl") 
            
            if os.path.exists(result_path) and is_pickle_valid(result_path):
                print(f"result {result_path} exists, skip")
                continue
            else:
                valid_pairs.append((video1, video2))
        
        print(f"total {len(valid_pairs)} unexist pairs")           
        pairs = valid_pairs
        
    if args.end == -1:
        args.end = len(pairs) -1
    
    pairs = pairs[args.start: args.end + 1]
    
    runtime = []
    for video1, video2 in tqdm(pairs):
        video1_name = os.path.basename(video1)
        video2_name = os.path.basename(video2)
        
        command = []
        command.append(f"cd {code_dir}")
        

        execute = f"CUDA_VISIBLE_DEVICES={args.gpu} python shaowei_sync_v6.py " \
            f"--dataset_root={args.dataset_root} --result_root={args.result_root} " \
            f"--video1_name={video1_name} --video2_name={video2_name}  --moving_threshold={args.moving_threshold} " \
            f"--pixel_threshold={args.pixel_threshold} --max_batch_size={args.max_batch_size} --offset_range={args.offset_range} " \
            f"--max_N={args.max_N} "
        
        if not args.not_use_v2:
            execute += "--use_v2 "
        if args.enable_poisson:
            execute += "--enable_poisson "
        if args.use_weight:
            execute += "--use_weight "
        if args.use_vggt:
            execute += f"--use_vggt --vggt_choice={args.vggt_choice} "
        if args.use_hloc:
            execute += "--use_hloc "
        if args.use_gt_intrinsic:
            execute += "--use_gt_intrinsic "
        if args.use_gt_sample:
            execute += "--use_gt_sample "
        if args.use_F_list:
            execute += "--use_F_list "
        if args.use_epipolar:
            execute += "--use_epipolar "
        if args.disable_gt:
            execute += "--disable_gt "
        if args.use_cosine:
            execute += "--use_cosine "
        if args.use_algebraic:
            execute += "--use_algebraic "
        command.append(execute)
        command = ";".join(command)
        # print(command)
        start_time = time.time()
        try:
            output = subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError as err:
            with open(os.path.join(log_file), 'a') as f:
                f.write(str(err))
                f.write("\n")
        end_time = time.time()
        elapsed_time = end_time - start_time
        runtime.append(elapsed_time)
        print(f"Processed {video1_name} and {video2_name} in {elapsed_time:.2f} seconds")
    if len(runtime) > 0:
        avg_runtime = sum(runtime) / len(runtime)
        print(f"Average runtime: {avg_runtime:.2f} seconds")
            
        
