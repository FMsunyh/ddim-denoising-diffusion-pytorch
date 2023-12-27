from typing import Tuple, Optional
from denoising_diffusion.utils import gather

import torch
import torch.nn.functional as F
import torch.utils.data
from torch import nn

import numpy as np

class DenoiseDiffusion:
    """
    ## Denoise Diffusion
    """

    def __init__(self, eps_model: nn.Module, n_steps: int, s_steps: int, device: torch.device, ddim_eta: float = 0.):
        """
        * `eps_model` is $\textcolor{lightgreen}{\epsilon_\theta}(x_t, t)$ model
        * `n_steps` is $t$
        * `device` is the device to place constants on
        """
        super().__init__()
        self.eps_model = eps_model

        # beta取值，线性等分向量
        self.beta = torch.linspace(0.00085, 0.012, n_steps).to(device)

        # alpha取值， 1-beta
        self.alpha = 1. - self.beta
        
        """
        b = torch.Tensor([1,2,3,4])
        a = torch.cumprod(b, dim=0)
        print(a) a=[1,2,6,24]
        """
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)
        
        #step默认值为T=1000(n_steps), S=50(s_steps)
        self.n_steps = n_steps
        
        c = self.n_steps // s_steps
        self.time_steps = np.asarray(list(range(0, self.n_steps, c))) + 1
        
        
        # sigma^2 = beta in DDPM
        # self.sigma2 = self.beta

        # for DDIM
        self.ddim_alpha = self.alpha_bar[self.time_steps].clone().to(torch.float32)
        # self.ddim_alpha_sqrt = torch.sqrt(self.ddim_alpha)
        # self.ddim_alpha_prev = torch.cat([self.alpha_bar[0:1], self.alpha_bar[self.time_steps[:-1]]])
        # self.ddim_sigma = (ddim_eta *
        #                        ((1 - self.ddim_alpha_prev) / (1 - self.ddim_alpha) *
        #                         (1 - self.ddim_alpha / self.ddim_alpha_prev)) ** .5)
        
        # self.ddim_sqrt_one_minus_alpha = (1. - self.ddim_alpha) ** .5
        
    # 下文中的q函数，就是训练过程需要用到的计算,
    def q_xt_x0(self, x0: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        获取 q(x_t|x_0)的分布，就是如果通过x_0直接得到x_t
        """

        # self.alpha_bar存放的值 是在init函数计算出来的，每个index下的数值，都是和前面的alpha相乘
        # gather函数，通过t作为index去获取alpha_bar中的元素， 类似数组alpha_bar[t]
        mean = gather(self.ddim_alpha, t) ** 0.5 * x0
        # $(1-\bar\alpha_t) \mathbf{I}$
        var = 1 - gather(self.ddim_alpha, t)
        #
        return mean, var

    #论文中的 【Algorithm 1 Training】
    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, eps: Optional[torch.Tensor] = None):
        """
        #### Sample from $q(x_t|x_0)$

        \begin{align}
        q(x_t|x_0) &= \mathcal{N} \Big(x_t; \sqrt{\bar\alpha_t} x_0, (1-\bar\alpha_t) \mathbf{I} \Big)
        \end{align}
        """

        # 【Algorithm 1 Training】中第4步的epsilon
        if eps is None:
            eps = torch.randn_like(x0)

        # 获取x_t,从 q(x_t|x_0)分布中得到，也就是添加了Noise的图片，简称Noise image,Noise image就是喂给Unet做训练的数据
        mean, var = self.q_xt_x0(x0, t)
        return mean + (var ** 0.5) * eps

    # # 下文中的q函数，就是推理过程需要用到的计算，也就是论文中的 【Algorithm 2 Sampling】
    # def p_sample(self, xt: torch.Tensor, t: torch.Tensor):
    #     """
    #     #### Sample from ${p_\theta}(x_{t-1}|x_t)$
    #     """

    #     # epsilon_theta(x_t, t)，这个就是Unet网络，输入为x_t,t
    #     eps_theta = self.eps_model(xt, t)
    #     # alpha_bar_t
    #     alpha_bar = gather(self.alpha_bar, t)
    #     # alpha_t
    #     alpha = gather(self.alpha, t)
        
    #     eps_coef = (1 - alpha) / (1 - alpha_bar) ** .5
       
    #     mean = 1 / (alpha ** 0.5) * (xt - eps_coef * eps_theta)
    #     # sigma^2
    #     var = gather(self.sigma2, t)

    #     # epsilon
    #     eps = torch.randn(xt.shape, device=xt.device)
    #     # Sample
    #     return mean + (var ** .5) * eps

    def loss(self, x0: torch.Tensor, noise: Optional[torch.Tensor] = None):
        """
        #### Simplified Loss
        """
        # batch size
        batch_size = x0.shape[0]
        
        # t是随机的
        
        t = torch.randint(0, len(self.time_steps), (batch_size,), device=x0.device, dtype=torch.long)

        # epsilon，就是N(0,1)出来的一个Noise,也就是GT label
        if noise is None:
            noise = torch.randn_like(x0)

        # 根据x_0,t获取x_t
        xt = self.q_sample(x0, t, eps=noise)
        
        # 根据x_t,t，预测eps_theta,也就是UNet预测出来的predict noise
        eps_theta = self.eps_model(xt, t)

        # MSE loss
        return F.mse_loss(noise, eps_theta)