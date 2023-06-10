# Discussion about the difference between this and the original sinkhorn algorithms implementation. This one is applying something like momentum: https://chat.openai.com/share/710bf589-eb7f-4c62-87d9-8a587c7b1105
# I took it from https://github.com/Megvii-BaseDetection/OTA/blob/2c85b4d0f9031396854aae969330dde2ab5eacbd/playground/detection/coco/ota.x101.fpn.coco.800size.1x/fcos.py#L384
# and modified accordingly.
# OTA took it from https://github.com/gpeyre/SinkhornAutoDiff/blob/master/sinkhorn_pointcloud.py and modified without mentioning.
import torch

class SinkhornDistance(torch.nn.Module):
    r"""
        Args:
        eps (float): regularization coefficient
        max_iter (int): maximum number of Sinkhorn iterations
    """

    def __init__(self, eps=1e-1, max_iter=50): # Default values from the paper OTA (and UniVIP)
        super(SinkhornDistance, self).__init__()
        self.eps = eps
        self.max_iter = max_iter

    def forward(self, mu, nu, C):
        u = torch.ones_like(mu) # demander a
        v = torch.ones_like(nu) # supplier b

        # Sinkhorn iterations
        for _ in range(self.max_iter):
            # NOTE original algorithm first updates u then v. Hence modified.
            u = self.eps * (torch.log(mu + 1e-8) - torch.logsumexp(self.M(C, u, v), dim=-1)) + u
            # transpose((-2, -1)) to avoid batch dimension
            v = self.eps * (torch.log(nu + 1e-8) - torch.logsumexp(self.M(C, u, v).transpose(-2, -1), dim=-1)) + v

        U, V = u, v
        # Transport plan pi = diag(a)*K*diag(b)
        pi = torch.exp(self.M(C, U, V)).detach()
        # Sinkhorn distance
        # transpose((-2, -1)) to avoid batch dimension
        cost = torch.sum(pi * C, dim=(-2, -1))
        return cost, pi

    def M(self, C, u, v):
        '''
        "Modified cost for logarithmic updates"
        "$M_{ij} = (-c_{ij} + u_i + v_j) / epsilon$"
        '''
        # Original code updates (-C + u.unsqueeze(1) + v.unsqueeze(0)) / epsilon, but here we avoid batch dimension.
        return (-C + u.unsqueeze(-1) + v.unsqueeze(-2)) / self.eps
