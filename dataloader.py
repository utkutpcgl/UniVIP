# Author: Utku Mert Topçuoğlu (took help from chatgpt https://chat.openai.com/share/caa323fa-babc-4b26-9faf-028cf10f740b.)
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import torchvision.io as io
import pickle
from icecream import ic
import time
from transforms import select_scenes, get_concatenated_instances, K_COMMON_INSTANCES

DEBUG = True
FILTERED_PKL = '/home/kuartis-dgx1/utku/UniVIP/COCO_proposals/trial/train2017_selective_search_proposal_enumerated_filtered_64_with_names_with_tensors_fixed_iou_trial_250.pkl'
DATASET_PATH = "/raid/utku/datasets/COCO_dataset/train2017/"
IMAGE_SIZE = 224

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

class CustomDataset(Dataset):
    def __init__(self, filtered_proposals:dict, image_size=IMAGE_SIZE):
        self.filtered_proposals = list(filtered_proposals["bbox"].items())
        self.total_sample_count=len(self.filtered_proposals)
        self.image_size=image_size
        scene_one, _, concatenated_instances = self.__getitem__(0)
        assert scene_one.dtype == torch.float32 
        assert concatenated_instances.shape[0]==K_COMMON_INSTANCES
    
    def __len__(self):
        # Return the total number of samples in the dataset
        return self.total_sample_count # or the number of image paths

    def __getitem__(self, idx):
        # Load your data here using the idx
        img_name, proposal_boxes = self.filtered_proposals[idx] 
        img_path = DATASET_PATH + img_name
        # Load the image using torchvision's read_image, reads int8 (cv2 also reads so)
        img = io.read_image(img_path)/255 # convert to 0-1 float32
        # NOTE feeding non-normalized float image values to ColorJitter clamps all values to max(x,1.0)!!!
        
        scene_one, scene_two, overlapping_boxes = select_scenes(img=img,proposal_boxes=proposal_boxes,image_size=self.image_size) # return scene_one, scene_two, overlapping_boxes
        concatenated_instances = get_concatenated_instances(img, overlapping_boxes)
        if scene_one.shape[0] == 1:
            scene_one, scene_two, concatenated_instances = scene_one.expand(3, -1, -1), scene_two.expand(3, -1, -1), concatenated_instances.expand(K_COMMON_INSTANCES, 3, -1, -1)
        # Return the data in the format you need, for example:
        return (scene_one, scene_two, concatenated_instances)


def init_dataset(batch_size, ddp=False):
    # Initialize your dataset
    dataset = CustomDataset(FILTERED_PROPOSALS)
    num_samples = len(dataset)
    sampler = None
    if ddp:
        # Initialize DistributedSampler
        sampler = DistributedSampler(dataset)
        dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler,shuffle=True)
    else:
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader, sampler, num_samples

