import pickle
from icecream import ic
import time
from pathlib import Path
import torch
from torch import nn
from torchvision import transforms as T
import torchvision.transforms.functional as TF
import random

DEBUG = True
DEFAULT_FILTERED_PKL = '/home/kuartis-dgx1/utku/UniVIP/COCO/image_info_unlabeled2017/annotations/image_info_unlabeled2017.json'
FALLBACK_FILTERED_PKL = '/home/kuartis-dgx1/utku/UniVIP/COCO/image_info_unlabeled2017/annotations/image_info_unlabeled2017.json'
DEFAULT_FILTER_SIZE = 96
FALLBACK_FILTER_SIZE = 64

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
DEFAULT_FILTERED_PROPOSALS = load_pkl(pkl_file=DEFAULT_FILTERED_PKL)
FALLBACK_FILTERED_PROPOSALS = load_pkl(pkl_file=FALLBACK_FILTERED_PKL)
# How do I construct the dataloader and set the batch (size)?

def tlhw_to_xyxy(t, l, h, w):
    x1 = l
    y1 = t
    x2 = l + w
    y2 = t + h
    return x1, y1, x2, y2

# augmentation utils

class RandomApply(nn.Module):
    def __init__(self, fn, p):
        super().__init__()
        self.fn = fn
        self.p = p
    def forward(self, x):
        return x if random.random() > self.p else self.fn(x)

def transform_image(image, image_size, type_two = False):
        # The git author missed different augmentations: https://github.com/lucidrains/byol-pytorch/issues/31#issuecomment-707925576
        # Apply random resized crop
        rand_res_crop = T.RandomResizedCrop((image_size, image_size))
        top, left, height, width = rand_res_crop.get_params(image, rand_res_crop.scale, rand_res_crop.ratio)
        # if size is int (not list) smaller edge will be scaled to match this.
        # byol uses bicubic interpolation.
        image = TF.resized_crop(image, top, left, height, width, size=(image_size,  image_size), interpolation="bicubic")

        # Apply horizontal flip (TODO might not need), T.RandomHorizontalFlip()
        flipped_bool = random.random() < 0.5
        if flipped_bool:
            image = TF.hflip(image) # Used functional.

        # TODO needs pil image as input?
        # since the order has to change (with randperm i get_params) I must use ColorJitter below.
        col_jit = RandomApply(T.ColorJitter(0.4, 0.4, 0.2, 0.1), p = 0.8)
        image = col_jit(image)

        # Apply grayscale with probability 0.2
        gray_scale = T.RandomGrayscale(p=0.2)
        image = gray_scale(image)


        # Apply gaussian blur with probability 0.2
        gaussian_prob = 0.1 if type_two else 1 # 1 for type_one
        gaus_blur = RandomApply(T.GaussianBlur((23, 23)), p = gaussian_prob)
        image = gaus_blur(image)

        # TODO add solarization as UniVIP applied it (like BYOL)

        # TODO do I have to call to tensor here in any case?

        # They apply normalization (not explicit in the paper: https://github.com/lucidrains/byol-pytorch/issues/4#issue-641183816)
        # Normalize the image (image has to be a tensor at this point, transforms before this can work on PIL images.)
        norm = T.Normalize(mean=torch.tensor([0.485, 0.456, 0.406]),std=torch.tensor([0.229, 0.224, 0.225]))
        # TODO make sure image is dtype=torch.float32
        image = norm(image)
        return image, flipped_bool, (top, left, height, width)


# 1. Get scene overlaps given the coordinates
def get_scene_overlap(crop_coordinates_one, crop_coordinates_two, fallback=False):
    min_size = DEFAULT_FILTER_SIZE if not fallback else FALLBACK_FILTER_SIZE # filter smaller boxes if fallback.
    """crop_coordinates has (top, left, height, width)"""
    def get_overlap(coord_one, coord_two):
        x1_1, y1_1, x2_1, y2_1 = coord_one
        x1_2, y1_2, x2_2, y2_2 = coord_two
        x1 = max(x1_1, x1_2) # left
        y1 = max(y1_1, y1_2) # top
        x2 = min(x2_1, x2_2) # right
        y2 = min(y2_1, y2_2) # bottom

        return (x1, y1, x2, y2) if x1 < x2 and y1 < y2 else None

    coord_one = tlhw_to_xyxy(crop_coordinates_one)
    coord_two = tlhw_to_xyxy(crop_coordinates_two)

    (x1, y1, x2, y2) = get_overlap(coord_one, coord_two)

    return None if (x2 - x1 < min_size or y2 - y1 < min_size) else (x1, y1, x2, y2)

    
def check_box_in_region(box, overlap_region):
    """Check if box is inside the region fully."""
    x1, y1, x2, y2 = box
    r_x1, r_y1, r_x2, r_y2 = overlap_region
    return x1 >= r_x1 and y1 >= r_y1 and x2 <= r_x2 and y2 <= r_y2

def get_overlapping_boxes(img_path, overlap_region, fallback):
    """ Get the proposed boxes in the overlapping region."""
    all_proposals = DEFAULT_FILTERED_PROPOSALS if not fallback else FALLBACK_FILTERED_PROPOSALS
    proposal_boxes_for_image = all_proposals[img_path]
    overlapping_boxes = []
    for box in proposal_boxes_for_image:
        # TODO check if box is inside the overlap region, if yes increment count.
        if check_box_in_region(box, overlap_region=overlap_region):
            overlapping_boxes.append(box)
    return overlapping_boxes

# 2. If they have at least K object regions in the overlapping region T return the scenes s1 and s2 (they are our targets)
def select_scenes(img, img_path, image_size, K=4, iters=20):
    """Returns scenes with at least K common targets in the overlapping regions."""
    while True:
        # I need the information which regions of the images were cropped and if RandomHorizontalFlip was applied (the region will change accordingly.)
        fallback = iters <= 0 # Use smaller box filter after some iterations.
        scene_one, _, crop_coordinates_one  = transform_image(img, image_size, type_two=False)
        scene_two, _, crop_coordinates_two = transform_image(img, image_size, type_two=True)
        overlap_coord = get_scene_overlap(crop_coordinates_one, crop_coordinates_two, fallback)
        if overlap_coord is None: # Check there is a large enough overlap
            continue
        # now check K common instances.
        overlapping_boxes = get_overlapping_boxes(img_path, overlap_coord, fallback)
        if len(overlapping_boxes) == K:
            return scene_one, scene_two, overlapping_boxes
        iters -= 1

# Calculate each loss based on the method (Lscene -> BYOL, Ls-i, Li-i)