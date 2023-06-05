"""This code was originaly taken from 
https://github.com/lucidrains/byol-pytorch/blob/master/byol_pytorch/byol_pytorch.py 
and modified accordingly."""
import copy
import random
from functools import wraps

import torch
from torch import nn
import torch.nn.functional as F

from torchvision import transforms as T
import torchvision.transforms.functional as TF

# helper functions

def flatten(t):
    return t.reshape(t.shape[0], -1)

def singleton(cache_key):
    def inner_fn(fn):
        @wraps(fn)
        def wrapper(self, *args, **kwargs):
            instance = getattr(self, cache_key)
            if instance is not None:
                return instance

            instance = fn(self, *args, **kwargs)
            setattr(self, cache_key, instance)
            return instance
        return wrapper
    return inner_fn

def get_module_device(module):
    return next(module.parameters()).device

def set_requires_grad(model, val):
    for p in model.parameters():
        p.requires_grad = val

# loss fn

def loss_fn(x, y):
    x = F.normalize(x, dim=-1, p=2)
    y = F.normalize(y, dim=-1, p=2)
    return 2 - 2 * (x * y).sum(dim=-1)

# augmentation utils

class RandomApply(nn.Module):
    def __init__(self, fn, p):
        super().__init__()
        self.fn = fn
        self.p = p
    def forward(self, x):
        if random.random() > self.p:
            return x
        return self.fn(x)


# exponential moving average

class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

def update_moving_average(ema_updater, ma_model, current_model):
    for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
        old_weight, up_weight = ma_params.data, current_params.data
        ma_params.data = ema_updater.update_average(old_weight, up_weight)

# MLP class for projector and predictor

def MLP(dim, projection_size, hidden_size=4096):
    return nn.Sequential(
        nn.Linear(dim, hidden_size),
        nn.BatchNorm1d(hidden_size),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_size, projection_size)
    )

# a wrapper class for the base neural network
# will manage the interception of the hidden layer output
# and pipe it into the projecter and predictor nets

class NetWrapper(nn.Module):
    def __init__(self, net, projection_size, projection_hidden_size, layer = -2):
        super().__init__()
        self.net = net
        self.layer = layer

        self.projector = None
        self.projection_size = projection_size
        self.projection_hidden_size = projection_hidden_size

        self.hidden = {}
        self.hook_registered = False

    def _find_layer(self):
        if type(self.layer) == str:
            modules = dict([*self.net.named_modules()])
            return modules.get(self.layer, None)
        elif type(self.layer) == int:
            children = [*self.net.children()]
            return children[self.layer]
        return None

    def _hook(self, _, input, output):
        device = input[0].device
        self.hidden[device] = flatten(output)

    def _register_hook(self):
        layer = self._find_layer()
        assert layer is not None, f'hidden layer ({self.layer}) not found'
        handle = layer.register_forward_hook(self._hook)
        self.hook_registered = True

    @singleton('projector')
    def _get_projector(self, hidden):
        _, dim = hidden.shape
        projector = MLP(dim, self.projection_size, self.projection_hidden_size)
        return projector.to(hidden)

    def get_representation(self, x):
        if self.layer == -1:
            return self.net(x)

        if not self.hook_registered:
            self._register_hook()

        self.hidden.clear()
        _ = self.net(x)
        hidden = self.hidden[x.device]
        self.hidden.clear()

        assert hidden is not None, f'hidden layer {self.layer} never emitted an output'
        return hidden

    def forward(self, x, return_projection = True):
        representation = self.get_representation(x)

        if not return_projection:
            return representation

        projector = self._get_projector(representation)
        projection = projector(representation)
        return projection, representation

# main class

class BYOL(nn.Module):
    def __init__(
        self,
        net,
        image_size,
        hidden_layer = -2,
        projection_size = 256,
        projection_hidden_size = 4096,
        moving_average_decay = 0.99,
    ):
        super().__init__()
        self.net = net

        # default SimCLR augmentation
        self.image_size = image_size

        self.online_encoder = NetWrapper(net, projection_size, projection_hidden_size, layer=hidden_layer)

        self.target_encoder = None
        self.target_ema_updater = EMA(moving_average_decay)

        self.online_predictor = MLP(projection_size, projection_size, projection_hidden_size)

        # get device of network and make wrapper same device
        device = get_module_device(net)
        self.to(device)

        # send a mock image tensor to instantiate singleton parameters
        self.forward(torch.randn(2, 3, image_size, image_size, device=device))
    
    
    def transform_image(self, image):
        # TODO needs pil image as input?
        # Variables that have to be returned.
        flipped_bool = False
        # Apply color jitter with probability 0.3
        # TODO since the order has to change (with randperm i get_params) I must use ColorJitter below.
        if random.random() < 0.3:
            col_jit = T.ColorJitter(0.8, 0.8, 0.8, 0.2)
            image = col_jit(image)

        # Apply grayscale with probability 0.2
        if random.random() < 0.2:
            gray_scale = T.RandomGrayscale(p=0.2)
            image = gray_scale(image)

        # Apply horizontal flip
        if random.random() < 0.5:
            flipped_bool = True
            image = TF.hflip(image) # Used functional.

        # Apply gaussian blur with probability 0.2
        if random.random() < 0.2:
            gaus_blur = T.GaussianBlur((3, 3), (1.0, 2.0))
            image = gaus_blur(image)

        # Apply random resized crop
        rand_res_crop = T.RandomResizedCrop((self.image_size, self.image_size))
        crop_coordinates = top, left, height, width = rand_res_crop.get_params(image, rand_res_crop.scale, rand_res_crop.ratio)
        # if size is int (not list) smaller edge will be scaled to match this.
        image = TF.resized_crop(image, top, left, height, width, size=(self.image_size, self. image_size))

        # Normalize the image
        norm = T.Normalize(mean=torch.tensor([0.485, 0.456, 0.406]),std=torch.tensor([0.229, 0.224, 0.225]))
        # TODO make sure image is dtype=torch.float32
        image = norm(image)
        return image, flipped_bool, crop_coordinates

    @singleton('target_encoder')
    def _get_target_encoder(self):
        target_encoder = copy.deepcopy(self.online_encoder)
        set_requires_grad(target_encoder, False)
        return target_encoder

    def reset_moving_average(self):
        del self.target_encoder
        self.target_encoder = None

    def update_moving_average(self):
        assert self.target_encoder is not None, 'target encoder has not been created yet'
        update_moving_average(self.target_ema_updater, self.target_encoder, self.online_encoder)

    def forward(
        self,
        x,
        img_path,
        return_embedding = False,
        return_projection = True
    ):
        assert not (self.training and x.shape[0] == 1), 'you must have greater than 1 sample when training, due to the batchnorm in the projection layer'

        if return_embedding:
            return self.online_encoder(x, return_projection = return_projection)
        
        # Get the proposals for image1 and image2.
        
        # I need the information which regions of the images were cropped and if RandomHorizontalFlip was applied (the region will change accordingly.)
        scene_one, flipped_bool_one, crop_coordinates_one  = self.transform_image(x)
        scene_two, flipped_bool_two, crop_coordinates_two = self.transform_image(x)

        # Check if the scenes overlap, if not augment them again (Do not increment current_iteration).
        # check_scene_overlap()
        # ThenCheck if the scenes contain K objects, if not augment them again.
        # check_K_common_objects()

        online_proj_one, _ = self.online_encoder(scene_one)
        online_proj_two, _ = self.online_encoder(scene_two)

        online_pred_one = self.online_predictor(online_proj_one)
        online_pred_two = self.online_predictor(online_proj_two)

        with torch.no_grad():
            target_encoder = self._get_target_encoder()
            # TODO here you have to pass instances and scenes trough the encoder and define the loss accordingly.
            # Pass image one and two
            target_proj_one, _ = target_encoder(image_one)
            target_proj_two, _ = target_encoder(image_two)
            target_proj_one.detach_()
            target_proj_two.detach_()

        loss_one = loss_fn(online_pred_one, target_proj_two.detach())
        loss_two = loss_fn(online_pred_two, target_proj_one.detach())

        loss = loss_one + loss_two
        return loss.mean()
