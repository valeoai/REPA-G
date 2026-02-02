import torch
import torch.nn as nn
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import interp1d
import math


class VanillaPotential:
    def __init__(self, lamda=1):
        super().__init__()
        self.lamda = lamda
  
    def compute(self, f, cond):
        f = nn.functional.normalize(f,dim=-1)
        return - self.lamda * (f*cond).sum(dim=-1)


class DiffusionModel(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=128, num_layers=3, depth=1):
        super().__init__()
        self.depth = depth
        assert  self.depth < num_layers, "Depth must be less than number of layers"
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.model = nn.ModuleList(layers)
        self.projection_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def interpolate(self, x, t):
        # Simple linear interpolation for demonstration
        z = torch.randn_like(x)
        return x * (1 - t) + t * z, z-x
    
    def forward(self, x, f, t):
        x_t, v = self.interpolate(x, t)
        x = torch.cat([x_t, t], dim=-1)

        for i, layer in enumerate(self.model):
            x = layer(x)
            if i == self.depth:
                f_tilde = self.projection_head(x)

        loss_flow_matching = nn.MSELoss()(x,v)
        loss_repa = -torch.mean((nn.functional.normalize(f_tilde,dim=-1)*f).sum(dim=-1))
        return x, loss_flow_matching, loss_repa
    
    @torch.no_grad()
    def sample(self, x, num_steps=10):
        for step in range(num_steps):
            x_init = x.clone()
            t = 1 - torch.ones(x.size(0), 1) * (step / num_steps)
            t = t.to(x.device)
            #Predict velocity
            v = torch.cat([x, t], dim=-1)
            for layer in self.model:
                v = layer(v)
            t_cur = 1 - step / num_steps
            eps = torch.randn_like(x)
            score = - (x_init + (1 - t_cur) * v) / t_cur
            d = v - score * t_cur
            x = x - d / num_steps + math.sqrt(2*t_cur/num_steps)*eps
        return x
    
    def sample_cond(self, x, cond, potential, num_steps=10):
        for step in range(num_steps):
            x_init = x.clone().detach().requires_grad_(True)
            t = 1 - torch.ones(x.size(0), 1) * (step / num_steps)
            t = t.to(x.device)
            #Predict velocity
            v = torch.cat([x_init, t], dim=-1)
            for i, layer in enumerate(self.model):
                v = layer(v)
                if i == self.depth:
                    f_tilde = self.projection_head(v)
            
            potential_value = potential.compute(f_tilde, cond)
            # Compute gradient independently for each batch element
            grad = torch.autograd.grad(
                outputs=potential_value,
                inputs=x_init,
                grad_outputs=torch.ones_like(potential_value),
                create_graph=False,
                retain_graph=False
            )[0]
            t_cur = 1 - step / num_steps
            eps = torch.randn_like(x)
            score = - (x_init + (1 - t_cur) * v) / t_cur
            d = v.detach() - (score.detach() - 2 * grad) * t_cur
            x = x - d / num_steps + math.sqrt(2*t_cur/num_steps)*eps
        return x