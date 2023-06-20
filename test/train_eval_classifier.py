""" They use random resized crop for training 
https://github.com/Jiahao000/ORL/blob/2ad64f7389d20cb1d955792aabbe806a7097e6fb/configs/benchmarks/linear_classification/imagenet/r50_multihead_28ep.py#L24"""
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
import torchvision
from pathlib import Path
from torchvision import models
from ..train.uni_vip import UVIP
from typing import OrderedDict

EPOCHS = 28
NUM_CLASSES = 1000
NUM_WORKERS = 4
BATCH_SIZE = 256
IMAGE_SIZE = 224
LOG_DIR = Path("/home/kuartis-dgx1/utku/UniVIP/logs")
CHECKPOINT_PATH = LOG_DIR/"uni_vip_pretrained_model.pt"

# Define the ResNet model
def get_single_resnet():
    # If you're in a distributed environment, make sure all processes are synchronized before loading
    resnet = models.resnet50(weights=None)
    model = UVIP(resnet)

    state_dict = model.state_dict()
    new_state_dict = {}

    for name, param in state_dict.items():
        if name.startswith("online_encoder"):
            new_state_dict[name.replace("online_encoder.net.", "")] = param

    new_linear_layer = torch.nn.Linear(2048, 1000)
    new_state_dict["fc.weight"] = new_linear_layer.weight
    new_state_dict["fc.bias"] = new_linear_layer.bias

    new_state_dict = OrderedDict(new_state_dict)
    resnet = models.resnet50(pretrained=False)
    # Assign the updated state_dict to the model
    resnet.load_state_dict(new_state_dict)

    # Freeze previous layers
    for param in resnet.parameters():
        param.requires_grad = False

    # Enable training for the new linear layer
    for param in resnet.fc.parameters():
        param.requires_grad = True
    return resnet

def get_ddp_resnet(single_resnet, rank):
    single_resnet = single_resnet.to(rank)
    single_resnet = DDP(single_resnet, device_ids=[rank])
    single_resnet = torch.nn.SyncBatchNorm.convert_sync_batchnorm(single_resnet) # Not sure if UniVIP does this.


def get_transforms():
    # Copied from ORL (says UniVIP) https://github.com/Jiahao000/ORL/blob/2ad64f7389d20cb1d955792aabbe806a7097e6fb/configs/benchmarks/linear_classification/imagenet/r50_multihead_28ep.py#L24
    # They added this strange lightning noise which I neglected: https://github.com/Jiahao000/ORL/blob/2ad64f7389d20cb1d955792aabbe806a7097e6fb/openselfsup/datasets/pipelines/transforms.py#L44
    train_transforms =  transforms.Compose([
                        transforms.RandomResizedCrop(size=IMAGE_SIZE),
                        transforms.RandomHorizontalFlip(),
                        transforms.ColorJitter(brightness=0.4,contrast=0.4,saturation=0.4,hue=0.),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    val_transforms =    transforms.Compose([
                        transforms.Resize(size=256),
                        transforms.CenterCrop(size=IMAGE_SIZE),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    return train_transforms, val_transforms
    

def train(local_rank, single_resnet):
    # Define the model, loss function, and optimizer
    ddp_resnet = get_ddp_resnet(single_resnet=single_resnet)
    # Set up the distributed environment
    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(local_rank)
    
    # Set random seed for reproducibility
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    
    # Define the batch size and number of workers
    
    # Define the dataset and data loader
    train_transforms, val_transforms = get_transforms()
    
    train_dataset = datasets.ImageNet(root='imagenet/train', split='train', transform=train_transforms)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = DataLoader(train_dataset, BATCH_SIZE=BATCH_SIZE, sampler=train_sampler, NUM_WORKERS=NUM_WORKERS)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(ddp_resnet.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0001)
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20], gamma=0.1)
    
    # Training loop
    num_epochs = 28
    for epoch in range(num_epochs):
        train_sampler.set_epoch(epoch)
        ddp_resnet.train()
        
        for images, labels in train_loader:
            images = images.to(local_rank)
            labels = labels.to(local_rank)
            
            # Forward pass
            outputs = ddp_resnet(images)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Learning rate scheduling
        lr_scheduler.step()
        
        # Print training progress
        if local_rank == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}] completed.")
    
    # Evaluation on validation split
    ddp_resnet.eval()
    val_dataset = datasets.ImageNet(root='imagenet/val', split='val', transform=val_transforms)
    val_loader = DataLoader(val_dataset, BATCH_SIZE=BATCH_SIZE, shuffle=False, NUM_WORKERS=NUM_WORKERS)
    
    total_samples = 0
    correct_predictions = 0
    
    for images, labels in val_loader:
        images = images.to(local_rank)
        labels = labels.to(local_rank)
        
        with torch.no_grad():
            outputs = ddp_resnet(images)
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()
    
    accuracy = correct_predictions / total_samples * 100
    
    # Print the top-1 center-crop accuracy on the validation split
    if local_rank == 0:
        print(f"Top-1 Accuracy: {accuracy:.2f}%")

def main():
    # Set the number of GPUs and spawn multiple processes
    single_resnet = get_single_resnet()
    num_gpus = 8
    mp.spawn(train, nprocs=num_gpus, args=(num_gpus,single_resnet))
    
if __name__ == '__main__':
    main()
