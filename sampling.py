import torch
device = torch.device("cpu")
import numpy as np
from scipy.stats import qmc
from physics import is_inside_nozzle, is_on_nozzle_boundary
from data import load_data
from loss import compute_loss

def generate_training_data(num_col_points=10000, num_bc_points=2000, batch_size=4096):
    try:
        coords, I_true, _, ri_mean, ri_std, ri_mean_original, ri_std_original = load_data()
        if coords is None:
            raise ValueError("Data loading failed, cannot proceed with training.")

        coords = coords.to(device)
        I_true = I_true.to(device)

        # Uniform sampling with slight bias toward central region
        bounds = np.array([[0.1, 0.9], [0.1, 0.9], [0.05, 0.95]])
        sampler = qmc.LatinHypercube(d=3)
        sample_col = sampler.random(n=num_col_points * 2)
        sample_col = qmc.scale(sample_col, bounds[:, 0], bounds[:, 1])
        
        # Bias toward central region (x=0.5±0.1, y=0.5±0.1)
        central_mask = ((sample_col[:, 0] > 0.4) & (sample_col[:, 0] < 0.6) &
                        (sample_col[:, 1] > 0.4) & (sample_col[:, 1] < 0.6))
        sample_col_central = sample_col[central_mask]
        sample_col_other = sample_col[~central_mask]
        
        num_central = int(num_col_points * 0.6)  # 60% of points in central region
        if len(sample_col_central) < num_central:
            sample_col_central = np.vstack([sample_col_central] * (num_central // len(sample_col_central) + 1))[:num_central]
        sample_col = np.vstack([sample_col_central[:num_central], sample_col_other[:(num_col_points - num_central)]])
        
        # Convert to tensors before passing to is_inside_nozzle
        x_tensor = torch.tensor(sample_col[:, 0], dtype=torch.float32, device=device)
        y_tensor = torch.tensor(sample_col[:, 1], dtype=torch.float32, device=device)
        z_tensor = torch.tensor(sample_col[:, 2], dtype=torch.float32, device=device)
        mask = is_inside_nozzle(x_tensor, y_tensor, z_tensor)
        
        # Try NumPy conversion, fall back to PyTorch if it fails
        try:
            sample_col_mask = mask.cpu().numpy()
        except RuntimeError as e:
            print(f"NumPy conversion error: {e}. Using PyTorch fallback.")
            sample_col_mask = mask.cpu()  # Keep as tensor for further operations
        
        sample_col = sample_col[sample_col_mask][:num_col_points]
        if len(sample_col) < num_col_points:
            sample_col = np.vstack([sample_col] * (num_col_points // len(sample_col) + 1))[:num_col_points]
        x_col = torch.tensor(sample_col, dtype=torch.float32).to(device)
        x_col = torch.cat([coords[:, :3], x_col], dim=0)

        # Boundary points
        bounds_bc = np.array([[0.1, 0.9], [0.05, 0.95]])
        sampler_2d = qmc.LatinHypercube(d=2)
        sample_bc = sampler_2d.random(n=num_bc_points)
        sample_bc = qmc.scale(sample_bc, bounds_bc[:, 0], bounds_bc[:, 1])
        x_bc = np.zeros((len(sample_bc), 3))
        x_bc[:, 1] = sample_bc[:, 0]
        x_bc[:, 2] = sample_bc[:, 1]
        for i in range(len(x_bc)):
            y, z = x_bc[i, 1], x_bc[i, 2]
            choice = np.random.choice(['side', 'top', 'bottom'], p=[0.6, 0.2, 0.2])
            if choice == 'side':
                angle = np.random.uniform(0, 2 * np.pi)
                x_bc[i, 0] = 0.5 + 0.4 * np.cos(angle)
                x_bc[i, 1] = 0.5 + 0.4 * np.sin(angle)
            elif choice == 'top':
                radius = np.random.uniform(0, 0.4)
                angle = np.random.uniform(0, 2 * np.pi)
                x_bc[i, 0] = 0.5 + radius * np.cos(angle)
                x_bc[i, 1] = 0.5 + radius * np.sin(angle)
                x_bc[i, 2] = 0.95
            else:
                radius = np.random.uniform(0, 0.4)
                angle = np.random.uniform(0, 2 * np.pi)
                x_bc[i, 0] = 0.5 + radius * np.cos(angle)
                x_bc[i, 1] = 0.5 + radius * np.sin(angle)
                x_bc[i, 2] = 0.05
        x_bc = torch.tensor(x_bc[:num_bc_points], dtype=torch.float32).to(device)

        col_dataset = torch.utils.data.TensorDataset(x_col.cpu())
        col_loader = torch.utils.data.DataLoader(col_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
        bc_dataset = torch.utils.data.TensorDataset(x_bc.cpu())
        bc_loader = torch.utils.data.DataLoader(bc_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)

        return col_loader, bc_loader, x_col, x_bc, coords, I_true, ri_mean, ri_std, ri_mean_original, ri_std_original

    except Exception as e:
        print(f"Error in generate_training_data: {e}")
        raise

def residual_adaptive_sampling(model, x_col, x_bc, weights, ri_mean, ri_std, num_new_points=8000, device=device):
    try:
        x_col, x_bc = x_col.to(device), x_bc.to(device)
        _, _, (L_te, L_bc) = compute_loss(model, x_col, x_bc, weights, ri_mean, ri_std, device)
        L_te_weight = torch.abs(L_te)
        L_bc_weight = torch.abs(L_bc)
        alpha_te, beta_te = 0.8, 0.2
        alpha_bc, beta_bc = 0.8, 0.2
        L_te_weight = alpha_te * torch.exp(L_te_weight) + beta_te
        L_bc_weight = alpha_bc * torch.exp(L_bc_weight) + beta_bc
        L_te_weight = torch.clamp(L_te_weight, min=1e-6)
        L_bc_weight = torch.clamp(L_bc_weight, min=1e-6)
        num_new_points_te = min(num_new_points, L_te_weight.shape[0])
        num_new_points_bc = min(int(num_new_points // 1.2), L_bc_weight.shape[0])
        probs_te = L_te_weight / torch.sum(L_te_weight)
        indices_te = torch.multinomial(probs_te.flatten(), num_new_points_te, replacement=True)
        new_points_te = x_col[indices_te]
        bounds = np.array([[0.1, 0.9], [0.1, 0.9], [0.05, 0.95]])
        sampler = qmc.LatinHypercube(d=3)
        sample_col = sampler.random(n=num_new_points)
        sample_col = qmc.scale(sample_col, bounds[:, 0], bounds[:, 1])
        x, y, z = sample_col[:, 0], sample_col[:, 1], sample_col[:, 2]
        x_tensor = torch.tensor(x, dtype=torch.float32, device=device)
        y_tensor = torch.tensor(y, dtype=torch.float32, device=device)
        z_tensor = torch.tensor(z, dtype=torch.float32, device=device)
        central_mask = (x_tensor > 0.4) & (x_tensor < 0.6) & (y_tensor > 0.4) & (y_tensor < 0.6)
        sample_col_central = sample_col[central_mask.cpu().numpy()]
        sample_col_other = sample_col[~central_mask.cpu().numpy()]
        num_central = int(num_new_points * 0.6)
        if len(sample_col_central) < num_central:
            sample_col_central = np.vstack([sample_col_central, sample_col_central[:num_central - len(sample_col_central)]])
        sample_col = np.vstack([sample_col_central[:num_central], sample_col_other[:(num_new_points - num_central)]])
        x_tensor = torch.tensor(sample_col[:, 0], dtype=torch.float32, device=device)
        y_tensor = torch.tensor(sample_col[:, 1], dtype=torch.float32, device=device)
        z_tensor = torch.tensor(sample_col[:, 2], dtype=torch.float32, device=device)
        
        # Try NumPy conversion, fall back to PyTorch if it fails
        mask = is_inside_nozzle(x_tensor, y_tensor, z_tensor)
        try:
            sample_col_mask = mask.cpu().numpy()
        except RuntimeError as e:
            print(f"NumPy conversion error: {e}. Using PyTorch fallback.")
            sample_col_mask = mask.cpu()  # Keep as tensor
        
        sample_col = sample_col[sample_col_mask][:num_new_points]
        if len(sample_col) < num_new_points:
            sample_col = np.vstack([sample_col, sample_col[:num_new_points - len(sample_col)]])
        x_col_new = torch.tensor(sample_col, dtype=torch.float32).to(device)
        x_col = torch.cat([x_col, x_col_new], dim=0)
        
        probs_bc = L_bc_weight / torch.sum(L_bc_weight)
        indices_bc = torch.multinomial(probs_bc.flatten(), num_new_points_bc, replacement=True)
        new_points_bc = x_bc[indices_bc]
        bounds_bc = np.array([[0.1, 0.9], [0.05, 0.95]])
        sampler_2d = qmc.LatinHypercube(d=2)
        sample_bc = sampler_2d.random(n=int(num_new_points // 1.2))
        sample_bc = qmc.scale(sample_bc, bounds_bc[:, 0], bounds_bc[:, 1])
        y_bc, z_bc = sample_bc[:, 0], sample_bc[:, 1]
        y_bc_tensor = torch.tensor(y_bc, dtype=torch.float32, device=device)
        z_bc_tensor = torch.tensor(z_bc, dtype=torch.float32, device=device)
        central_mask_bc = (y_bc_tensor > 0.4) & (y_bc_tensor < 0.6)
        sample_bc_central = sample_bc[central_mask_bc.cpu().numpy()]
        sample_bc_other = sample_bc[~central_mask_bc.cpu().numpy()]
        num_central_bc = int(len(sample_bc) * 0.6)
        if len(sample_bc_central) < num_central_bc:
            sample_bc_central = np.vstack([sample_bc_central, sample_bc_central[:num_central_bc - len(sample_bc_central)]])
        sample_bc = np.vstack([sample_bc_central[:num_central_bc], sample_bc_other[:(len(sample_bc) - num_central_bc)]])
        x_bc_new = np.zeros((len(sample_bc), 3))
        x_bc_new[:, 1] = sample_bc[:, 0]
        x_bc_new[:, 2] = sample_bc[:, 1]
        for i in range(len(x_bc_new)):
            y, z = x_bc_new[i, 1], x_bc_new[i, 2]
            choice = np.random.choice(['side', 'top', 'bottom'], p=[0.6, 0.2, 0.2])
            if choice == 'side':
                angle = np.random.uniform(0, 2 * np.pi)
                x_bc_new[i, 0] = 0.5 + 0.4 * np.cos(angle)
                x_bc_new[i, 1] = 0.5 + 0.4 * np.sin(angle)
            elif choice == 'top':
                radius = np.random.uniform(0, 0.4)
                angle = np.random.uniform(0, 2 * np.pi)
                x_bc_new[i, 0] = 0.5 + radius * np.cos(angle)
                x_bc_new[i, 1] = 0.5 + radius * np.sin(angle)
                x_bc_new[i, 2] = 0.95
            else:
                radius = np.random.uniform(0, 0.4)
                angle = np.random.uniform(0, 2 * np.pi)
                x_bc_new[i, 0] = 0.5 + radius * np.cos(angle)
                x_bc_new[i, 1] = 0.5 + radius * np.sin(angle)
                x_bc_new[i, 2] = 0.05
        x_bc_new = torch.tensor(x_bc_new[:int(num_new_points // 1.2)], dtype=torch.float32).to(device)
        x_bc = torch.cat([x_bc, x_bc_new], dim=0)
        return x_col, x_bc
    except Exception as e:
        print(f"Residual adaptive sampling failed: {str(e)}. Skipping this round...")
        return x_col, x_bc