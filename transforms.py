import pickle
from icecream import ic
import time
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms as T
import torchvision.transforms.functional as TF
import random
from naive_box_generation import add_n_random_boxes

DEBUG = True
FILTERED_PKL = '/home/kuartis-dgx1/utku/UniVIP/data_ops/train2017_selective_search_proposal_enumerated_filtered_64_with_names_with_tensors_fixed_iou_trial_250.pkl'
FILTER_SIZE = 64
K_COMMON_INSTANCES = 4

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
FILTERED_PROPOSALS = load_pkl(pkl_file=FILTERED_PKL)
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

def crop_scene(image, image_size):
    # The git author missed different augmentations: https://github.com/lucidrains/byol-pytorch/issues/31#issuecomment-707925576
    # Apply random resized crop
    rand_res_crop = T.RandomResizedCrop((image_size, image_size))
    top, left, height, width = rand_res_crop.get_params(image, rand_res_crop.scale, rand_res_crop.ratio)
    # if size is int (not list) smaller edge will be scaled to match this.
    # byol uses bicubic interpolation.
    image = TF.resized_crop(image, top, left, height, width, size=(image_size,  image_size), interpolation="bicubic")
    return image, (top, left, height, width)

def common_augmentations(image, type_two = False):
    rand_hor_flip = T.RandomHorizontalFlip(p=0.5)
    image = rand_hor_flip(image)

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

    solarize_prob = 0.2 if type_two else 0 # assymetric augm
    assert not (image<=1).all(), "Images are in 0-1"
    solarize_threshold = 128
    solarize = T.RandomSolarize(threshold=solarize_threshold, p=solarize_prob)
    image = solarize(image)

    # TODO do I have to call to tensor here in any case?

    # They apply normalization (not explicit in the paper: https://github.com/lucidrains/byol-pytorch/issues/4#issue-641183816)
    # Normalize the image (image has to be a tensor at this point, transforms before this can work on PIL images.)
    norm = T.Normalize(mean=torch.tensor([0.485, 0.456, 0.406]),std=torch.tensor([0.229, 0.224, 0.225]))
    # TODO make sure image is dtype=torch.float32
    image = norm(image)
    return image


# 1. Get scene overlaps given the coordinates, TODO does not operate on tensors (should work on batches)
def get_scene_overlap(crop_coordinates_one, crop_coordinates_two):
    min_overlap_size = FILTER_SIZE*3/2 # filter scenes with too small overlap
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

    return None if (x2 - x1 < min_overlap_size or y2 - y1 < min_overlap_size) else (x1, y1, x2, y2)


def check_box_in_region(boxes, overlap_region):
    """Check if boxes are inside the region fully."""
    r_x1, r_y1, r_x2, r_y2 = overlap_region
    return (boxes[:, 0] >= r_x1) & (boxes[:, 1] >= r_y1) & (boxes[:, 2] <= r_x2) & (boxes[:, 3] <= r_y2)


def get_overlapping_boxes(img_path, overlap_region):
    """ Get the proposed boxes in the overlapping region."""
    all_proposals = FILTERED_PROPOSALS
    proposal_boxes_for_image = all_proposals[img_path]
    inside_region_mask = check_box_in_region(proposal_boxes_for_image, overlap_region=overlap_region)
    overlapping_boxes = proposal_boxes_for_image[inside_region_mask]
    return overlapping_boxes if len(overlapping_boxes)!=0 else torch.zeros(0, 4, dtype=torch.int64)


# 2. If they have at least K_common_instances object regions in the overlapping region T return the scenes s1 and s2 (they are our targets)
def select_scenes(img, img_path, image_size, K_common_instances=K_COMMON_INSTANCES, iters=20):
    # TODO test this with trial pkl.
    # NOTE we get only K_common_instances boxes and ablations show there is no improvement after 4!!
    """Returns scenes with at least K_common_instances common targets in the overlapping regions."""
    best_scenes={"overlapping_boxes":torch.zeros(0,4, dtype=torch.int64), "overlap_coord":None, "s1":None, "s2":None}
    while iters > 0:
        # I need the information which regions of the images were cropped and if RandomHorizontalFlip was applied (the region will change accordingly.)
        scene_one, crop_coordinates_one  = crop_scene(img, image_size)
        scene_two, crop_coordinates_two = crop_scene(img, image_size)
        overlap_coord = get_scene_overlap(crop_coordinates_one, crop_coordinates_two)
        if overlap_coord is None: # Check there is a large enough overlap
            continue
        # now check K_common_instances common instances.
        overlapping_boxes = get_overlapping_boxes(img_path, overlap_coord)
        if len(overlapping_boxes) >= K_common_instances:
            return scene_one, scene_two, overlapping_boxes[:K_common_instances] # Get only first K_common_instances boxes.
        elif len(overlapping_boxes) > len(best_scenes["overlapping_boxes"]): # Update the best boxes for the fallback case.
            best_scenes["overlapping_boxes"], best_scenes["overlap_coord"], best_scenes["s1"], best_scenes["s2"] = overlapping_boxes, overlap_coord, scene_one, scene_two
        iters -= 1
    else:
        # Add random boxes to the overlapping coordinates.
        missing_box_num = K_common_instances-len(best_scenes["overlapping_boxes"])
        best_scenes["overlapping_boxes"] = add_n_random_boxes(overlap_coord=best_scenes["overlap_coord"], overlapping_boxes=best_scenes["overlapping_boxes"], n_random_boxes=missing_box_num)
        return best_scenes["s1"], best_scenes["s2"], best_scenes["overlapping_boxes"][:K_common_instances] # Get only first K_common_instances boxes.

    
def get_concatenated_instances(img, overlapping_boxes, instance_dim):
    # Resize and feed instances in overlapping boxes to the online encoder
    # When there is a batch dimension the instance_dim should be 1
    instances = []
    for box in overlapping_boxes:
        x1, x2, y1, y2 = box
        instance = img[..., y1:y2, x1:x2] # crop instance from image tensor
        instance = F.interpolate(instance, size=(96, 96), mode="bicubic") # resize instance to 96x96
        instances.append(instance)
    return torch.stack(instances, dim=instance_dim) # vertical stack.
