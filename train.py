"""
Run with python -m torch.distributed.launch --nproc_per_node=NUM_GPUS_YOU_HAVE train_script.py

NOTE to load DDP weights: configure map_location properly
    map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
    ddp_model.load_state_dict(
        torch.load(CHECKPOINT_PATH, map_location=map_location)
"""
import os
from uni_vip import UVIP
import torch
from torchvision import models
from torch.optim.lr_scheduler import CosineAnnealingLR
from dataloader import init_dataset
from math import ceil

from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
import torch.distributed as dist

# DDP train settings.
USE_DDP = False
DEVICE = 0 # Device for single gpu training
WORLD_SIZE = 8 # Number of GPUs for multi gpu training

# was not pretrained by default for ORL also.https://github.com/Jiahao000/ORL/blob/2ad64f7389d20cb1d955792aabbe806a7097e6fb/configs/selfsup/orl/coco/stage3/r50_bs512_ep800.py#L7 
batch_size = 512
total_epochs = 800
# update momentum every iteration with cosine annealing.
base_learning_rate = 0.2 # same as ORL.
final_min_lr = 0 # Here it was said 0, no explicit in univip: https://github.com/Jiahao000/ORL/blob/2ad64f7389d20cb1d955792aabbe806a7097e6fb/configs/selfsup/orl/coco/stage3/r50_bs512_ep800.py#L144
CHECKPOINT_PATH = "uni_vip_pretrained_model.pt"



def update_lr(optimizer, lr):
    # Update the learning rate of the optimizer
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def ddp_setup(rank, world_size):
    # initialize the process group,  It ensures that every process will be able to coordinate through a master, using the same ip address and port. 
    # nccl backend is currently the fastest and highly recommended backend when using GPUs. This applies to both single-node and multi-node distributed training. https://pytorch.org/docs/master/generated/torch.nn.parallel.DistributedDataParallel.html 
    # TODO set ,timeout= to avoid any timeout during random box selection (might lose synchronization)
    dist.init_process_group(backend='nccl',rank=rank, world_size=world_size) # For multi GPU train.

def cleanup():
    dist.destroy_process_group()

def train(rank, world_size):
    # Create the model.
    if USE_DDP:
        ddp_setup(rank=rank, world_size=world_size)
    resnet = models.resnet50(pretrained=False)
    model = UVIP(resnet).to(rank)
    if USE_DDP:
        model = DDP(model, device_ids=[rank]) # will return ddp model output.

    # Optimizer and Scheduler
    optimizer = torch.optim.SGD(resnet.parameters(), weight_decay=0.0001, momentum=0.9, lr=base_learning_rate)
    # Warmup and schedulers
    warm_up_epochs = 4
    warm_up_lrs = [base_learning_rate/(warm_up_epochs-cur_epoch) for cur_epoch in range(warm_up_epochs)]
    # NOTE Do not apply cosine for the first warm_up_epochs, otherwise collides with warmup. First step considers base_lr not the current lr of the optimizer. Look here: https://pytorch.org/docs/stable/_modules/torch/optim/lr_scheduler.html#CosineAnnealingLR:~:text=elif%20self._step_count,optimizer.param_groups)%5D
    cosine_scheduler = CosineAnnealingLR(optimizer=optimizer, T_max=total_epochs-warm_up_epochs, eta_min=final_min_lr)

    # Training epochs.
    for cur_epoch in range(total_epochs):
        # Shuffle data at the beginning of each epoch
        if USE_DDP:
            sampler.set_epoch(epoch=cur_epoch) # to ensure that the distributed sampler shuffles the data differently at the beginning of each epoch
        # Warmup first epochs.
        if cur_epoch < warm_up_epochs:
            lr = warm_up_lrs[cur_epoch]
            update_lr(optimizer, lr=lr)

        # train for one epoch.
        for img_data in dataloader:
            iteration_count+=1
            loss = model(img_data)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # After each step teacher is updated based on BYOL paper.
            model.update_moving_average(tot_iter=total_iterations,cur_iter=iteration_count) # update moving average of target encoder
        # Cosine scheduler after some steps.
        if cur_epoch >= warm_up_epochs:
            cosine_scheduler.step(epoch=cur_epoch-warm_up_epochs)
        print("saving network")
        # save your improved network
        if rank == DEVICE:
            # Therefore, saving it in one process is sufficient.
            torch.save(model.state_dict(), CHECKPOINT_PATH)
            # TODO log here the epoch number and the model.
    
    if USE_DDP:
        cleanup() # For multi GPU train.

if __name__ == "__main__":
    # Create DataLoader with the custom dataset and the distributed sampler
    dataloader, sampler, num_samples = init_dataset(batch_size=batch_size, ddp=USE_DDP)
    total_iterations = ceil(num_samples / batch_size)*total_epochs 
    iteration_count=0

    if USE_DDP:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355' # set port for communication
        mp.spawn(train,args=(WORLD_SIZE,),nprocs=WORLD_SIZE, join=True)
    else:
        train(rank=DEVICE, world_size=1) # single gpu train

