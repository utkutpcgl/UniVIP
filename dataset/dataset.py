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


# 1. Randomly crop two areas of the image. # Already done in byol's forward pass
# 2. If they have at least K object regions in the overlapping region T return the scenes s1 and s2 (they are our targets)

def select_scenes(img, img_path, K, iters):
    """Returns scenes with at least K common targets in the overlapping regions."""
    return scene1,scene2

def check_K_instances(img_path, overlap_region):
    """ Tell if the scenes have at least K common instances in the overlapping regions."""
    


# 3. We return instances directly since we dont know which instances to match directly?

# Calculate each loss based on the method (Lscene -> BYOL, Ls-i, Li-i)