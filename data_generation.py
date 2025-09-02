import numpy as np
import torch
device = torch.device("cpu")
#device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

def generate_los_data(num_points=10000, grid_size=(100, 100, 99), bounds=((0.005, 0.995), (0.005, 0.995), (0.01, 0.99))):
    nx, ny, nz = grid_size
    x_bounds, y_bounds, z_bounds = bounds
    x = np.linspace(x_bounds[0], x_bounds[1], nx)
    y = np.linspace(y_bounds[0], y_bounds[1], ny)
    z = np.linspace(z_bounds[0], z_bounds[1], nz)
    dz = (z_bounds[1] - z_bounds[0]) / (nz - 1)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    r2 = (X - 0.5)**2 + (Y - 0.5)**2 + (Z - 0.5)**2
    T = -800 * r2 + 2000
    kappa = -2 * r2 + 5
    lambda_um = 1.0
    c1, c2 = 3.741832e8, 1.4388e4
    I_b_lambda = (c1 / (lambda_um**5)) / (np.exp(c2 / (lambda_um * T)) - 1)
    tau = kappa * dz
    I = np.zeros_like(Z)
    for k in range(nz-1, -1, -1):  # Reverse order for LOS
        if k == nz-1:
            I[:, :, k] = I_b_lambda[:, :, k] * (1 - np.exp(-tau[:, :, k]))
        else:
            I[:, :, k] = I[:, :, k+1] * np.exp(-tau[:, :, k]) + I_b_lambda[:, :, k] * (1 - np.exp(-tau[:, :, k]))
    # Sample points
    idx = np.random.choice(nx * ny * nz, num_points, replace=False)
    idx_x, idx_y, idx_z = np.unravel_index(idx, (nx, ny, nz))
    coords = np.column_stack([X[idx_x, idx_y, idx_z], Y[idx_x, idx_y, idx_z], Z[idx_x, idx_y, idx_z], 
                             np.zeros(num_points), np.zeros(num_points), np.full(num_points, lambda_um)])
    I_true = I[idx_x, idx_y, idx_z]
    
    return torch.tensor(coords, dtype=torch.float32), torch.tensor(I_true, dtype=torch.float32)