from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import torchvision.io as io
import pickle
from icecream import ic
import time

DEBUG = True
FILTERED_PKL = '/home/kuartis-dgx1/utku/UniVIP/COCO_proposals/trial/train2017_selective_search_proposal_enumerated_filtered_64_with_names_with_tensors_fixed_iou_trial_250.pkl'

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
    def __init__(self, filtered_proposals:dict, transform=None):
        self.filtered_proposals = list(filtered_proposals.items())
        self.total_sample_count=len(self.filtered_proposals)
        self.transform = transform
    
    def __len__(self):
        # Return the total number of samples in the dataset
        return self.total_sample_count # or the number of image paths

    def __getitem__(self, idx):
        # Load your data here using the idx
        img_path, proposal_boxes = self.filtered_proposals[idx] 
        # Load the image using torchvision's read_image
        img = io.read_image(img_path)
        
        # apply transforms if they are defined
        if self.transform:
            img = self.transform(img)
        
        # Return the data in the format you need, for example:
        return img, proposal_boxes


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

