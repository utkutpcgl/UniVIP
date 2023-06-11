""" get_max_iou and box_filter methods were copied from the 
ORL (Unsupervised Object-Level Representation Learning from Scene Images) paper's github repo:
https://github.com/Jiahao000/ORL/tree/2ad64f7389d20cb1d955792aabbe806a7097e6fb 

Minutes passed.
30:42 for train
49:48 for unlabelled dataset
"""

import numpy as np
import torch
from torch import Tensor
from tqdm import tqdm
import pickle
from icecream import ic
import time
from pathlib import Path
from multiprocessing import Pool

NUM_PROC = 80
DEBUG = True
TARGET_PKL = "/home/kuartis-dgx1/utku/UniVIP/COCO/unlabeled2017_selective_search_proposal_enumerated.pkl"
IOU_THRESH=0.5
if not DEBUG:
    ic.disable()

previous_timestamp = int(time.time())
def unixTimestamp():
    global previous_timestamp
    current_timestamp = int(time.time())
    minutes, seconds = divmod(current_timestamp-previous_timestamp, 60)
    previous_timestamp = current_timestamp
    return 'Local time elapsed %02d:%02d |> ' % (minutes, seconds)

ic.configureOutput(prefix=unixTimestamp)


def get_max_iou(filtered_boxes: Tensor, candidate_box: Tensor) -> Tensor:
    """
    filtered_boxes : multiple coordinate for predict bounding boxes (x, y, w, h)
    candidate_box :   the coordinate for ground truth bounding box (x, y, w, h)
    return :   the max iou score about filtered_boxes and candidate_box
    """
    # 1.get the coordinate of inters
    ixmin = torch.max(filtered_boxes[:, 0], candidate_box[0])
    ixmax = torch.min(filtered_boxes[:, 0] + filtered_boxes[:, 2], candidate_box[0] + candidate_box[2])
    iymin = torch.max(filtered_boxes[:, 1], candidate_box[1])
    iymax = torch.min(filtered_boxes[:, 1] + filtered_boxes[:, 3], candidate_box[1] + candidate_box[3])

    iw = torch.clamp(ixmax - ixmin, min=0.)
    ih = torch.clamp(iymax - iymin, min=0.)

    # 2. calculate the area of inters
    inters = iw * ih

    # 3. calculate the area of union
    uni = (filtered_boxes[:, 2] * filtered_boxes[:, 3] + candidate_box[2] * candidate_box[3] - inters)

    # 4. calculate the overlaps and find the max overlap between filtered_boxes and candidate_box
    iou = inters / uni
    return torch.max(iou)


def box_filter(candidate_boxes_tensor, max_iou_thr=None):
    """Since the order is important cannot apply async multiprocessing."""
    # NOTE I have to re-generate all candidate_boxes, called filtered_box_proposals.append(box) before max_iou_thr: 
    assert candidate_boxes_tensor.numel!=0 # number of elements
    filtered_candidate_boxes_tensor = torch.ones(0,4)
    for box in candidate_boxes_tensor:
        # Calculate width and height of the box
        iou_max = get_max_iou(filtered_candidate_boxes_tensor, box)
        if iou_max > max_iou_thr:
            continue
        filtered_candidate_boxes_tensor = torch.cat((filtered_candidate_boxes_tensor,box.unsqueeze(0)))
    return filtered_candidate_boxes_tensor # [[x,y,w,h], [x,y,w,h], [x,y,w,h] ...]

def load_pkl(pkl_file):
    with open(pkl_file, 'rb') as f:
        raw_data = pickle.load(f)
    return raw_data

def filter_proposals_single(boxes_per_image_tensor):
    # Empty lists will be appended for images without proposals.
    filtered_boxes_64_single = box_filter(candidate_boxes=boxes_per_image_tensor, max_iou_thr=IOU_THRESH)
    return filtered_boxes_64_single

def filter_proposals_with_names_single(args:list):
    img_path, boxes_per_image_tensor = args
    result = filter_proposals_single(boxes_per_image_tensor)
    return img_path, result

def dump_pkl(pkl_file, data):
    with open(pkl_file, 'wb') as file:
        pickle.dump(data, file)


def main(images_dicts):
    images_lists_with_names = list(images_dicts.items())
    with Pool(processes=NUM_PROC) as p:
        results = list(tqdm(p.imap(filter_proposals_with_names_single, images_lists_with_names), total=len(images_lists_with_names)))
    
    # Sort the results by the original index and extract the results
    filtered_data_64 = {"bbox": {name: bboxes for name, bboxes in results}}
    return filtered_data_64

if __name__ == "__main__":
    # Load data from pickle file
    ic("load_pkl")
    raw_data = load_pkl(TARGET_PKL)
    # Access the dictionary
    images_dicts = raw_data["bbox"]
    # Create an iterable that includes the original indexes
    ic("filter_proposals")
    filtered_data_64 = main(images_dicts=images_dicts)
    
    # Save filtered data into separate JSON files
    ic("dump_pkl")
    dump_pkl(f'{Path(TARGET_PKL).stem}_filtered_64.pkl', filtered_data_64)
