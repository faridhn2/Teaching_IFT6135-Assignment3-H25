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
        t = t / self.n_steps
        lambda_min = torch.tensor(self.lambda_min).expand_as(t).to(args.device)
        lambda_max = torch.tensor(self.lambda_max).expand_as(t).to(args.device)
        
        b = torch.arctan(torch.exp(-lambda_max / 2))
        a = torch.arctan(torch.exp(-lambda_min / 2)) - b
        
        return - 2 * torch.log(torch.tan(a * t + b)).reshape(-1, 1, 1, 1)

        
    
    def alpha_lambda(self, lambda_t: torch.Tensor): 
        #TODO: Write function that returns Alpha(lambda_t) for a specific time t according to (1)
        var = 1.0 / (1.0 + torch.exp(-1.0 * lambda_t))
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
    
    def sigma_q_x(self, lambda_t: torch.Tensor, lambda_t_prim: torch.Tensor) -> torch.Tensor:
        """
        Compute the standard deviation of the forward process q(z_lambda | z_lambda_prim)
        based on the formula: sigma^2_{lambda|lambda'} = (1 - exp(lambda - lambda')) * sigma^2_lambda
        """
        # Ensure shapes are broadcastable
        lambda_diff = lambda_t - lambda_t_prim

        # Compute sigma^2_lambda = 1 / (1 + exp(lambda))
        sigma2_lambda = 1.0 / (1.0 + torch.exp(lambda_t))

        # Apply the forward process variance formula
        sigma2 = (1.0 - torch.exp(lambda_diff)) * sigma2_lambda

        # Return standard deviation
        return torch.sqrt(sigma2.clamp(min=1e-20)) 

    
    def mu_p_theta(self, z_lambda_t: torch.Tensor, x: torch.Tensor, lambda_t: torch.Tensor, lambda_t_prim: torch.Tensor):
        alpha_lambda = self.alpha_lambda(lambda_t)
        alpha_lambda_prim = self.alpha_lambda(lambda_t_prim)
        e_l_ratio = self.get_exp_ratio(lambda_t,lambda_t_prim)
        z_lambda_t = e_l_ratio * alpha_lambda_prim/alpha_lambda * z_lambda_t
        x_new = (1 - e_l_ratio) * alpha_lambda_prim * x
        mu = z_lambda_t + x_new
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
        mu_theta = self.mu_p_theta(z_lambda_t, x_t, lambda_t, lambda_t_prim)
        var_theta = self.var_p_theta(lambda_t, lambda_t_prim)
        noise = torch.randn(z_lambda_t.shape, device=z_lambda_t.device)
        return mu_theta + noise * var_theta.sqrt()

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

        eps_pred = self.eps_model(z_lambda, labels)

        return F.mse_loss(eps_pred, noise)



    
