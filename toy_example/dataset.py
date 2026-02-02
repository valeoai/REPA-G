import torch
from torch.utils.data import Dataset
import numpy as np

class ToyDataset(Dataset):
    def __init__(self, num_samples=1000, cell_size=1, cells_range=1):
        super().__init__()
        self.num_samples = num_samples
        self.cell_size = cell_size
        self.cells_range = cells_range
        self.data = self._generate_data()
        self.features = self.feature_from_coord(self.data)
    
    def _generate_data(self):
        X = np.random.uniform(-self.cells_range, self.cells_range, (self.num_samples*3,2))

        # Compute cell indices and mask (one cell out of two active)
        i = np.floor(X[:,0] / self.cell_size).astype(int)
        j = np.floor(X[:,1] / self.cell_size).astype(int)
        mask = ((i + j) % 2)   # 1 = active, 0 = empty

        X = X[mask==1][:self.num_samples]
        return X
    
    def feature_from_coord(self, X):

        r_max = np.sqrt(2)*(self.cell_size/2)  # max distance from center to corner

        i = np.floor(X[:,0] / self.cell_size).astype(int)
        j = np.floor(X[:,1] / self.cell_size).astype(int)

        # Coordinates relative to cell center (u,v) in range approx [-T/2, T/2]
        cell_centers_x = (i + 0.5) * self.cell_size
        cell_centers_y = (j + 0.5) * self.cell_size
        u = X[:,0] - cell_centers_x
        v = X[:,1] - cell_centers_y

        r = np.sqrt(u**2 + v**2)
        t = np.clip(r / r_max, 0.0, 1.0)*2-1  # 0=center, 1=corner

        #distance to edge of cell
        h = (self.cell_size/2 - np.abs(u))*2/self.cell_size
        w = (self.cell_size/2 - np.abs(v))*2/self.cell_size

        norm = np.linalg.norm(np.vstack([t,h-w]).T, axis=-1)

        angle = np.arccos(t/norm)*np.sign(h-w)

        # Feature map: active -> point on unit circle; empty -> origin
        phi = np.zeros_like(X)
        phi[:,0] = np.cos(angle)
        phi[:,1] = np.sin(angle)
        return phi
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.float32), torch.tensor(self.features[idx], dtype=torch.float32)

