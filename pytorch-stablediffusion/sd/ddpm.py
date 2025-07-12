import torch
import numpy as np

class DDPMSampler:

    def __init__(self, generator: torch.Generator, num_training_steps=1000, beta_start: float = 0.00085, beta_end: float=0.0120):
        #beta is a series of numbers that indicates the variance of the noise that we add with each of these steps
        # the start and end values were a choice made by the authors
        # will be using a linear scheduler, 1000 numbers between start and end

        self.betas = torch.linspace(beta_start ** 0.5, beta_end ** 0.5, num_training_steps, dtype=torch.float32) ** 2

        # alpha bar is the product of alpha going from 1 to T
        self.alphas = 1.0 - self.betas
        self.apha_cumprod = torch.cumprod(self.alphas, 0) 
        self.one = torch.tensor(1.0)

        self.generator = generator
        self.num_training_steps = num_training_steps
        self.timesteps = torch.from_numpy(np.arange(0, num_training_steps)[::-1].copy())
    
    def set_inference_timesteps(self, num_inference_steps=50):
        self.num_inference_steps = num_inference_steps
        # 999, 998, 997, ... 0 = 1000 steps
        # 999, 999-20, 999-40, ... 0 = 50 steps
        step_ratio = self.num_training_steps // num_inference_steps
        timesteps = (np.arange(0, num_inference_steps) * step_ratio).round()[::-1].copy().astype(np.int64)
        self.timesteps = torch.from_numpy(timesteps)