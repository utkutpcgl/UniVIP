"""
Run with python -m torch.distributed.launch --nproc_per_node=NUM_GPUS_YOU_HAVE train_script.py
"""
from uni_vip import UVIP
import torch
from torchvision import models
from torch.optim.lr_scheduler import CosineAnnealingLR
from dataloader import init_dataset, TOTAL_ITERATIONS
import torch.distributed as dist
from math import ceil

dist.init_process_group(backend='nccl') # For multi GPU train.


# was not pretrained by default for ORL also.https://github.com/Jiahao000/ORL/blob/2ad64f7389d20cb1d955792aabbe806a7097e6fb/configs/selfsup/orl/coco/stage3/r50_bs512_ep800.py#L7 
batch_size = 512
total_epochs = 800
# update momentum every iteration with cosine annealing.
base_learning_rate = 0.2 # same as ORL.
final_min_lr = 0 # Here it was said 0, no explicit in univip: https://github.com/Jiahao000/ORL/blob/2ad64f7389d20cb1d955792aabbe806a7097e6fb/configs/selfsup/orl/coco/stage3/r50_bs512_ep800.py#L144

# Create the model.
resnet = models.resnet50(pretrained=False)
model = UVIP(resnet)
model = torch.nn.parallel.DistributedDataParallel(model)

# Optimizer and Scheduler
optimizer = torch.optim.SGD(resnet.parameters(), weight_decay=0.0001, momentum=0.9, lr=base_learning_rate)
# Warmup and schedulers
warm_up_epochs = 4
warm_up_lrs = [base_learning_rate/(warm_up_epochs-cur_epoch) for cur_epoch in range(warm_up_epochs)]
initial_lr = 0
warmup_factor = (base_learning_rate - initial_lr) / warm_up_epochs
# NOTE Do not apply cosine for the first warm_up_epochs, otherwise collides with warmup. First step considers base_lr not the current lr of the optimizer. Look here: https://pytorch.org/docs/stable/_modules/torch/optim/lr_scheduler.html#CosineAnnealingLR:~:text=elif%20self._step_count,optimizer.param_groups)%5D
cosine_scheduler = CosineAnnealingLR(optimizer=optimizer, T_max=total_epochs-warm_up_epochs, eta_min=final_min_lr)

# HELPER FUNCTIONS
def sample_unlabelled_images():
    return torch.randn(20, 3, 256, 256)

def update_lr(optimizer, lr):
    # Update the learning rate of the optimizer
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == "__main__":
    # Create DataLoader with the custom dataset and the distributed sampler
    dataloader, num_samples = init_dataset(batch_size=batch_size)
    total_iterations = ceil(num_samples / batch_size)*total_epochs 
    iteration_count=0
    for cur_epoch in range(total_epochs):
        # Warmup first epochs.
        if cur_epoch < warm_up_epochs:
            lr = warm_up_lrs[cur_epoch]
            update_lr(optimizer, lr=lr)

        # training
        for img_path, images in dataloader:
            iteration_count+=1
            images = sample_unlabelled_images()
            loss = model(images)
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
        torch.save(resnet.state_dict(), './improved-net.pt')