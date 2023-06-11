import torch
from torch import Tensor

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

def generate_random_box(overlap_coord, overlapping_boxes, min_size=64, max_ratio=3, iou_threshold=0.5, max_trials=30):
    """overlap_coord is (x1,y1,x2,y2)
    Generate random x,y,w,h uniformly in the allowed ranges.
    """
    iou = 1
    trial_count = 0 # NOTE after some iterations neglect the iou rule to avoid being stuck.
    while iou>iou_threshold and trial_count<max_trials:
        # Generate w
        (o_x1,o_y1,o_x2,o_y2)=overlap_coord
        max_width, max_height = o_x2-o_x1, o_y2-o_y1
        # generate width uniformly between min_size and max_width
        width = torch.randint(min_size, max_width + 1, (1,))
        # generate hieght uniformly between max(min_size,width/max_ratio) and min(max_height,widht*max_ratio)
        min_height = max(min_size, width / max_ratio)
        max_height = min(max_height, width * max_ratio)
        height = torch.randint(int(min_height), int(max_height) + 1, (1,))
        # generate random x1 and y1 uniformly (top-left coordinates) such that o_x1<=x1 and x1+width<=o_x2 and o_y1<=y1 and y1+height<o_y
        x1 = torch.randint(o_x1, o_x2 - width + 1, (1,)).item()
        y1 = torch.randint(o_y1, o_y2 - height + 1, (1,)).item()
        # create the random box coordinates as a tensor.
        random_box = torch.tensor([x1, y1, width, height])
        if len(overlapping_boxes) != 0:
            iou = get_max_iou(filtered_boxes=overlapping_boxes, candidate_box=random_box)
        else:
            iou=0
        trial_count+=1
    return random_box

def add_n_random_boxes(overlap_coord,overlapping_boxes,n_random_boxes):
    for _ in range(n_random_boxes):
        # based on iou, min_size and ratio add new random boxes
        random_box = generate_random_box(overlap_coord=overlap_coord,overlapping_boxes=overlapping_boxes)
        # Append the random box to overlapping_boxes tensor
        overlapping_boxes = torch.cat((overlapping_boxes, random_box.unsqueeze(0)), dim=0)
    return overlapping_boxes