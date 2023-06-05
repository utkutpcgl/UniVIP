import json
import pickle
from icecream import ic
import time
from pathlib import Path

DEBUG = True
FILTERED_96_PKL = '/home/kuartis-dgx1/utku/UniVIP/COCO/image_info_unlabeled2017/annotations/image_info_unlabeled2017.json'
FILTERED_64_PKL = '/home/kuartis-dgx1/utku/UniVIP/COCO/image_info_unlabeled2017/annotations/image_info_unlabeled2017.json'

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

# Check if the boxes correspond to correct images (sorted)
def load_pkl(pkl_file):
    with open(pkl_file, 'rb') as f:
        raw_data = pickle.load(f)
    return raw_data
# Load the filtered box proposal pkl files, convert to faster readable form (tensors?)
FILTERED_96_PROPOSALS = load_pkl(pkl_file=FILTERED_96_PKL)
FILTERED_64_PROPOSALS = load_pkl(pkl_file=FILTERED_64_PKL)
# How do I construct the dataloader and set the batch (size)?

def tlhw_to_xyxy(t,l,h,w):
    return x1,y1,x2,y2

# 1. Get scene overlaps given the coordinates
def get_scene_overlap(flipped_bool_one, crop_coordinates_one, flipped_bool_two, crop_coordinates_two, original_image_shape):
    # TODO check this twice
    """crop_coordinates has (top, left, height, width)"""
    min_size = 96
    def get_overlap(coord_one, coord_two):
        x1_1, y1_1, x2_1, y2_1 = coord_one
        x1_2, y1_2, x2_2, y2_2 = coord_two
        x1 = max(x1_1, x1_2) # left
        y1 = max(y1_1, y1_2) # top
        x2 = min(x2_1, x2_2) # right
        y2 = min(y2_1, y2_2) # bottom

        return (x1, y1, x2, y2) if x1 < x2 and y1 < y2 else None

    crop_coordinates_one = tlhw_to_xyxy(crop_coordinates_one)
    crop_coordinates_two = tlhw_to_xyxy(crop_coordinates_two)

    if flipped_bool_one:
        pass
    if flipped_bool_two:
        pass

    (x1, y1, x2, y2) = get_overlap()

    if x2 - x1 < min_size or y2 - y1 < min_size:
        # TODO check this twice
        return None # since these cannot have proposals with 96 box proposals.

    return (x1, y1, x2, y2)


# 2. If they have at least K object regions in the overlapping region T return the scenes s1 and s2 (they are our targets)

def select_scenes(img, img_path, K, iters):
    """Returns scenes with at least K common targets in the overlapping regions."""
    return scene1,scene2

def check_K_instances(img_path, overlap_region):
    """ Tell if the scenes have at least K common instances in the overlapping regions."""
    


# 3. We return instances directly since we dont know which instances to match directly?

# Calculate each loss based on the method (Lscene -> BYOL, Ls-i, Li-i)