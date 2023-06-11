"""This code was originaly taken from 
https://github.com/lucidrains/byol-pytorch/blob/master/byol_pytorch/byol_pytorch.py 
and modified accordingly."""
import copy
from functools import wraps
import torch
from torch import nn
import torch.nn.functional as F

from transforms import select_scenes, common_augmentations, get_concatenated_instances, K_COMMON_INSTANCES
from sink_knop_dist import SinkhornDistance

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
    """Cosine similarity is (proportional (/2)) to MSE when x-y are l2 normalized
    https://stats.stackexchange.com/questions/146221/is-cosine-similarity-identical-to-l2-normalized-euclidean-distance"""
    # L2 normalization (Divided L2 norm), hence, resulting l2_norm = 1 --> MSE = cosine_sim 
    x = F.normalize(x, dim=-1, p=2) # 
    y = F.normalize(y, dim=-1, p=2)
    return (2 - 2 * (x * y).sum(dim=-1)).mean()


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
        image_size = 224, # default byol
        hidden_layer = -2,
        projection_size = 256,
        projection_hidden_size = 4096,
        moving_average_decay = 0.99,
        K_common_instances=K_COMMON_INSTANCES, # Number of instances in the ovrelap.
    ):
        super().__init__()
        self.net = net
        self.image_size = image_size

        self.online_encoder = NetWrapper(net, projection_size, projection_hidden_size, layer=hidden_layer)
        self.online_predictor = MLP(projection_size, projection_size, projection_hidden_size)

        self.target_encoder = None
        self.target_ema_updater = EMA(moving_average_decay)

        # UniVIP specific layers
        self.K_common_instances = K_common_instances
        self.instances_to_scene_linear = nn.Linear(projection_size*K_common_instances, projection_size) # K concatenated
        self.sinkhorn_distance = SinkhornDistance()

        # get device of network and make wrapper same device
        device = get_module_device(net)
        self.to(device)

        # send a mock image tensor to instantiate singleton parameters
        self.forward(torch.randn(2, 3, image_size, image_size, device=device))
    

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

    def ii_loss_fn(self, online_pred_instance, target_proj_instance, online_pred_avg, target_proj_avg):
        """(target_proj_one+target_proj_two)/2, (online_pred_one+online_pred_two)/2
        return optimal_plan_matrix"""
        # instance to instance loss (optimal transport and sinkhorn-knopp)
        # Step 1: Compute dot_product_matrix
        O_matrix = online_pred_instance # (instance numbers K) x (number of features)
        T_matrix = target_proj_instance # (instance numbers K) x (number of features)
        dot_product_matrix = torch.mm(O_matrix.t(), T_matrix)
        # Step 2: Compute Norm matrices
        norm_matrix_O = torch.norm(O_matrix, dim=0, keepdim=True)
        norm_matrix_T = torch.norm(T_matrix, dim=0, keepdim=True)
        # Step 3: Compute C
        ot_cosine_similarity_matrix = (dot_product_matrix / (norm_matrix_O.t() * norm_matrix_T))
        cost_matrix = 1 - ot_cosine_similarity_matrix
        a_vector = torch.max(T_matrix.t*online_pred_avg,0)
        b_vector = torch.max(O_matrix.t*target_proj_avg,0)
        _, optimal_plan_matrix = self.sinkhorn_distance(a_vector,b_vector,cost_matrix)
        loss_ii = torch.sum(-torch.mul(optimal_plan_matrix*ot_cosine_similarity_matrix), dim=(-2,-1)) # Forces similar instance representations to be close to each other.
        return loss_ii

    def forward(
        self,
        img,
        img_path,
        return_embedding = False,
        return_projection = True
    ):
        assert not (self.training and img.shape[0] == 1), 'you must have greater than 1 sample when training, due to the batchnorm in the projection layer'

        if return_embedding:
            return self.online_encoder(img, return_projection = return_projection)
        
        # FEED THE SCENES
        # Get the scenes and box_proposals (overlapping_boxes) for image1 and image2.
        scene_one, scene_two, overlapping_boxes = select_scenes(img, img_path, self.image_size)
        scene_one = common_augmentations(scene_one,type_two=False)
        scene_two = common_augmentations(scene_two,type_two=True)
        with torch.no_grad():
            target_encoder = self._get_target_encoder()
            # Pass scene one and two
            target_proj_one, _ = target_encoder(scene_one)
            target_proj_two, _ = target_encoder(scene_two)
            target_proj_one.detach_()
            target_proj_two.detach_()
        online_proj_one, _ = self.online_encoder(scene_one)
        online_proj_two, _ = self.online_encoder(scene_two)
        online_pred_one = self.online_predictor(online_proj_one)
        online_pred_two = self.online_predictor(online_proj_two)


        # FEED THE INSTANCES
        # Get the crops from the image, resize them to 96, feed to online network, and concatenate them
        # Resize and feed instances in overlapping boxes to the online encoder
        instance_dim = 1 if img.ndim==4 else 0 # Choose the instance dimension 
        concatenated_instances = get_concatenated_instances(img, overlapping_boxes, instance_dim=instance_dim)
        concatenated_instances_squeezed = concatenated_instances.reshape(-1,*concatenated_instances.size()[-3:]) # Squeeze along batch dim
        concatenated_instances_squeezed = common_augmentations(concatenated_instances_squeezed,type_two=False)
        online_proj_instance, _ = self.online_encoder(concatenated_instances_squeezed)
        online_pred_instance = self.online_predictor(online_proj_instance)
        # restore the batch dimension if it was available
        online_pred_instance = online_pred_instance if instance_dim==0 else online_pred_instance.reshape(img.shape[0], self.K_common_instances, online_pred_instance.shape[-1])
        online_pred_concatenated_instance = torch.concat(online_pred_instance, dim=instance_dim)
        online_concatenated_final_instance_representations = self.instances_to_scene_linear(online_pred_concatenated_instance)
        with torch.no_grad():
            # Pass instances in the overlapping region
            target_proj_instance, _ = target_encoder(concatenated_instances_squeezed)
            target_proj_instance.detach_()
            target_proj_instance = target_proj_instance if instance_dim==0 else target_proj_instance.reshape(img.shape[0], self.K_common_instances, target_proj_instance.shape[-1])
        

        # CALCULATE LOSSES
        # Scene to scene loss
        loss_ss_one = loss_fn(online_pred_one, target_proj_two.detach())
        loss_ss_two = loss_fn(online_pred_two, target_proj_one.detach())
        loss_ss = loss_ss_one + loss_ss_two
        # Scene to instance loss
        loss_si_one = loss_fn(online_concatenated_final_instance_representations, target_proj_one.detach())
        loss_si_two = loss_fn(online_concatenated_final_instance_representations, target_proj_two.detach())
        loss_si = loss_si_one + loss_si_two
        # instance to instance loss (optimal transport and sinkhorn-knopp)
        online_pred_avg = (online_pred_one+online_pred_two)/2
        target_proj_avg = (target_proj_one+target_proj_two)/2
        # TODO check this loss twice.
        loss_ii = self.ii_loss_fn(online_pred_instance, target_proj_instance, online_pred_avg, target_proj_avg)

        return loss_ss + loss_si + loss_ii
