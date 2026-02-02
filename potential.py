import torch

class PotentialBase:
    def __init__(self, cond, lamda=1.0):
        self.cond = torch.nn.functional.normalize(cond, dim=-1)
        if not isinstance(lamda, torch.Tensor):
            self.lamda = torch.tensor([lamda], device=cond.device, dtype=torch.float32)
        else:
            self.lamda = lamda.to(cond.device, dtype=torch.float32)

    def compute(self, f):
        raise NotImplementedError


class MultiPotential:
    def __init__(self, potentials, secondary_potential_guidance_threshold):
        self.potentials = potentials
        self.secondary_potential_guidance_threshold = secondary_potential_guidance_threshold

    def compute(self, f):
        total_energy = 0
        for potential in self.potentials:
            total_energy += potential.compute(f)
        return total_energy
    
    def get_potential_at_t(self, t):
        if t >= self.secondary_potential_guidance_threshold:
            return self
        else:
            return self.potentials[0]


class RepaPotential(PotentialBase):
    def __init__(self, cond, lamda=1, mask=None):
        super().__init__(cond, lamda=lamda)
        if mask is None:
            mask = torch.ones(cond.shape[:2], device=cond.device)
        self.mask = mask
    
    def compute(self, f):
        f = torch.nn.functional.normalize(f,dim=-1)
        # Ensure lamda is broadcastable to the batch dimension of f
        # f shape: (B, N, D), self.cond shape: (B, N, D) or (1, N, D)
        # self.mask shape: (B, N) or (1, N)
        # self.lamda shape: (B,) or (1,)
        masked_similarity = (f * self.cond * self.mask[:, :, None]).sum(dim=-1)  # (B, N)
        energy = -self.lamda.view(-1, 1) * torch.sum(masked_similarity, dim=-1, keepdim=True) / torch.sum(self.mask, dim=-1, keepdim=True)
        return energy.squeeze(-1)
    
    
class MeanFeatAlignment(PotentialBase):
    def __init__(self, cond, lamda=1):
        super().__init__(cond, lamda=lamda)

    def compute(self, f):
        mean_f = torch.mean(f, dim=1, keepdim=True)  # (B, 1, D)
        mean_f = torch.nn.functional.normalize(mean_f,dim=-1)
        alignment = (mean_f * self.cond).sum(dim=-1)  # (B, N)
        return - self.lamda * alignment


class FreeEnergy(PotentialBase):
    def __init__(self, cond, lamda=1, T=1):
        super().__init__(cond, lamda=lamda)
        if not isinstance(T, torch.Tensor):
            self.T = torch.tensor([T], device=cond.device, dtype=torch.float32)
        else:
            self.T = T.to(cond.device, dtype=torch.float32)
        self.T = self.T.view(-1, 1) # Reshape for broadcasting: (B,) -> (B, 1)
    
    def compute(self, f):
        f = torch.nn.functional.normalize(f, dim=-1)
        logits = (f * self.cond).sum(dim=-1) / self.T  # (N,)
        logZ = torch.logsumexp(logits, dim=-1)
        free_energy = -self.T.squeeze(-1) * logZ
        return self.lamda * free_energy

def feature_dir_update(x, f, potential, retain_graph=False):
    energy = potential.compute(f)  # suppose shape: (N,)
    grad = torch.autograd.grad(
        outputs=energy,
        inputs=x,
        grad_outputs=torch.ones_like(energy),
        create_graph=False,
        retain_graph=retain_graph,
    )[0]
    return grad