# %%
import torch
import torch.utils.data
import torchvision
from torch import nn
from typing import Tuple, Optional
import torch.nn.functional as F
from tqdm import tqdm
from easydict import EasyDict
import matplotlib.pyplot as plt
from torch.amp import GradScaler, autocast
import os 

from cfg_utils.args import * 


class CFGDiffusion():
    def __init__(self, eps_model: nn.Module, n_steps: int, device: torch.device):
        super().__init__()
        self.eps_model = eps_model
        self.n_steps = n_steps
        
        self.lambda_min = -20
        self.lambda_max = 20



    ### UTILS
    def get_exp_ratio(self, l: torch.Tensor, l_prim: torch.Tensor):
        return torch.exp(l-l_prim)
    
    def get_lambda(self, t: torch.Tensor): 
        # TODO: Write function that returns lambda_t for a specific time t. Do not forget that in the paper, lambda is built using u in [0,1]
        u = t.float() / self.n_steps
        lambda_t = self.lambda_min + (self.lambda_max - self.lambda_min) * u
        return lambda_t.view(-1, 1, 1, 1)

        
    
    def alpha_lambda(self, lambda_t: torch.Tensor): 
        #TODO: Write function that returns Alpha(lambda_t) for a specific time t according to (1)
        var = 1.0 / (1.0 + torch.exp(-2.0 * lambda_t))
        return var.sqrt()
    
    def sigma_lambda(self, lambda_t: torch.Tensor): 
        alpha = self.alpha_lambda(lambda_t)
        var = 1.0 - alpha ** 2
        return var.sqrt()
    
    ## Forward sampling
    def q_sample(self, x: torch.Tensor, lambda_t: torch.Tensor, noise: torch.Tensor):
        #TODO: Write function that returns z_lambda of the forward process, for a specific: x, lambda l and N(0,1) noise  according to (1)
        alpha = self.alpha_lambda(lambda_t)
        sigma = self.sigma_lambda(lambda_t)
        return alpha * x + sigma * noise
               
    def sigma_q(self, lambda_t: torch.Tensor, lambda_t_prim: torch.Tensor):
        #TODO: Write function that returns variance of the forward process transition distribution q(•|z_l) according to (2)
        alpha_t = self.alpha_lambda(lambda_t)
        alpha_t_prim = self.alpha_lambda(lambda_t_prim)
        var = 1.0 - (alpha_t ** 2) / (alpha_t_prim ** 2)
        return var.sqrt()
    
    def sigma_q_x(self, lambda_t: torch.Tensor, lambda_t_prim: torch.Tensor):
        #TODO: Write function that returns variance of the forward process transition distribution q(•|z_l, x) according to (3)
        sigma_l = self.sigma_lambda(lambda_t)
        sigma_q_ = self.sigma_q(lambda_t, lambda_t_prim)
        var = (sigma_l ** 2) / (sigma_q_ ** 2)
        return var.sqrt()

    ### REVERSE SAMPLING
    def mu_p_theta(self, z_lambda_t: torch.Tensor, x: torch.Tensor, lambda_t: torch.Tensor, lambda_t_prim: torch.Tensor):
        #TODO: Write function that returns mean of the forward process transition distribution according to (4)
        eps_theta = self.eps_model(z_lambda_t, lambda_t, y)
        scale = self.sigma_q_x(lambda_t, lambda_t_prim) ** 2
        mu = z_lambda_t + scale * eps_theta
        return mu

    def var_p_theta(self, lambda_t: torch.Tensor, lambda_t_prim: torch.Tensor, v: float=0.3):
        #TODO: Write function that returns var of the forward process transition distribution according to (4)
        sigma_qx = self.sigma_q_x(lambda_t, lambda_t_prim) ** 2
        sigma_q_ = self.sigma_q(lambda_t, lambda_t_prim) ** 2
        var = v * sigma_qx + (1 - v) * sigma_q_
        return var
    
    def p_sample(self, z_lambda_t: torch.Tensor, lambda_t : torch.Tensor, lambda_t_prim: torch.Tensor,  x_t: torch.Tensor, set_seed=False):
        # TODO: Write a function that sample z_{lambda_t_prim} from p_theta(•|z_lambda_t) according to (4) 
        # Note that x_t correspond to x_theta(z_lambda_t)
        if set_seed:
            torch.manual_seed(42)
         x_t = z_lambda_t - self.sigma_lambda(lambda_t) * self.eps_model(z_lambda_t, lambda_t, y)
         x_t = x_t / self.alpha_lambda(lambda_t)

         mu = self.mu_p_theta(z_lambda_t, y, lambda_t, lambda_t_prim)
         var = self.var_p_theta(lambda_t, lambda_t_prim)
         noise = torch.randn_like(z_lambda_t)

         return mu + torch.sqrt(var) * noise

    ### LOSS
    def loss(self, x0: torch.Tensor, labels: torch.Tensor, noise: Optional[torch.Tensor] = None, set_seed=False):
        if set_seed:
            torch.manual_seed(42)
        batch_size = x0.shape[0]
        dim = list(range(1, x0.ndim))
        t = torch.randint(
            0, self.n_steps, (batch_size,), device=x0.device, dtype=torch.long
        )
        if noise is None:
            noise = torch.randn_like(x0)
        lambda_t = self.get_lambda(t)
        z_lambda = self.q_sample(x0, lambda_t, noise)

        eps_pred = self.eps_model(z_lambda, lambda_t, labels)

        return F.mse_loss(eps_pred, noise)



    
