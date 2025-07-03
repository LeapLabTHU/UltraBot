import torch
import numpy as np

def get_noised_coordinate(coordinate, a_min=0, a_max=20):
    coordinate_noise = torch.zeros(coordinate.shape, device=coordinate.device)
    for i in range(coordinate.shape[0]):
        for j in range(coordinate.shape[1]):
            # 10 percent chance +-1
            if np.random.rand() < 0.1:
                coordinate_noise[i, j] = np.random.choice([1., -1.])
            elif np.random.rand() < 0.05:
                coordinate_noise[i, j] = np.random.choice([2., -2.])
            elif np.random.rand() < 0.01:
                coordinate_noise[i, j] = np.random.choice(list(np.arange(-10,11))) * 1.0
    return torch.clip(coordinate + coordinate_noise, a_min, a_max)
