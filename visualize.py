import matplotlib.pyplot as plt
import torch
import numpy as np
from scipy.interpolate import griddata
import os
device = torch.device("cpu")
#device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

# Ensure figures directory exists
os.makedirs('./figures', exist_ok=True)

def compute_mre(predictions, true_values):
    abs_error = torch.abs(predictions - true_values)
    rel_error = abs_error / (true_values + 1e-10)
    return torch.mean(rel_error).item()

def plot_history(history):
    plt.figure(figsize=(10, 6), dpi=150)
    plt.plot(history['total_loss'], label='Total Loss')
    if 'loss_te' in history:
        plt.plot(history['loss_te'], label='Transport Equation Loss', alpha=0.7)
    if 'loss_bc' in history:
        plt.plot(history['loss_bc'], label='Boundary Condition Loss', alpha=0.7)
    if 'loss_pen' in history:
        plt.plot(history['loss_pen'], label='Penalty Loss', alpha=0.7)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curves')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.yscale('log')  # Log scale for better visualization of loss convergence
    plt.savefig('./figures/loss_curve.png')
    plt.close()

def plot_raw_data(coords, I_true, ri_mean, ri_std):
    coords = coords.cpu().numpy()
    I_true = (I_true.cpu().numpy() * ri_std + ri_mean) # Unnormalize
    # try for i:z_end in z_target to iterate over z_target?
    z_target = 0.7

    mask = np.abs(coords[:, 2] - z_target) < 0.01
    data_z83 = coords[mask]
    ri_z83 = I_true[mask]
    print(f"Raw data z=0.83 slice size: {len(data_z83)}")
    
    x = data_z83[:, 0]
    y = data_z83[:, 1]
    xi = np.linspace(0.1, 0.9, 100)
    yi = np.linspace(0.1, 0.9, 100)
    xi, yi = np.meshgrid(xi, yi)
    zi = griddata((x, y), ri_z83, (xi, yi), method='linear', fill_value=0.0)
    
    plt.figure(figsize=(6, 5), dpi=150)
    plt.contourf(xi, yi, zi, levels=50, cmap='jet')
    plt.colorbar(label='Radiation Intensity')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f'Raw Data Intensity (z={z_target:.2f})')  
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig(f'./figures/raw_data_z{z_target:.2f}.png')  
    plt.close()

def plot_predictions(model, coords, I_true, ri_mean, ri_std, z_target):
    model.eval()
    x = torch.linspace(0.1, 0.9, 100, device=device)
    y = torch.linspace(0.1, 0.9, 100, device=device)
    X, Y = torch.meshgrid(x, y, indexing='xy')
    Z = torch.full_like(X, z_target, device=device)
    points = torch.stack([X.ravel(), Y.ravel(), Z.ravel()], dim=1)
    phi = torch.full_like(points[:, 0:1], 0.1745, device=device)
    theta = torch.zeros_like(points[:, 0:1], device=device)
    lambda_ = torch.full_like(points[:, 0:1], 1e-6, device=device)
    points_full = torch.cat([points, phi, theta, lambda_], dim=1)
    
    # Batch prediction
    predictions = []
    with torch.no_grad():
        for i in range(0, points_full.shape[0], 1000):
            batch = points_full[i:i+1000]
            pred_batch = model(batch).cpu().numpy()
            predictions.append(pred_batch)
        predictions = np.concatenate(predictions).reshape(100, 100) * ri_std + ri_mean

    # Get true values at z_target
    coords_device = coords.to(device)
    I_true_device = I_true.to(device)
    true_idx = torch.isclose(coords_device[:, 2], torch.tensor(z_target, device=device), atol=1e-2)
    true_values = (I_true_device[true_idx] * ri_std + ri_mean).cpu().numpy()
    true_coords = coords_device[true_idx, :2].cpu().numpy()

    # Interpolate true values
    true_grid = griddata(true_coords, true_values, 
                        (X.cpu().numpy(), Y.cpu().numpy()), 
                        method='linear', fill_value=0.0)

    # Visualization
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    # Predicted
    im1 = ax1.contourf(X.cpu().numpy(), Y.cpu().numpy(), predictions, levels=50, cmap='jet')
    ax1.set_title(f'Predicted (z={z_target:.2f})')
    fig.colorbar(im1, ax=ax1, label='Intensity')
    
    # True
    im2 = ax2.contourf(X.cpu().numpy(), Y.cpu().numpy(), true_grid, levels=50, cmap='jet')
    ax2.set_title(f'True (z={z_target:.2f})')
    fig.colorbar(im2, ax=ax2, label='Intensity')
    
    # Difference
    difference = predictions - true_grid
    im3 = ax3.contourf(X.cpu().numpy(), Y.cpu().numpy(), difference, 
                      levels=50, cmap='jet')
    max_diff = np.max(np.abs(difference))
    im3.set_clim(-max_diff, max_diff)
    ax3.set_title(f'Difference (z={z_target:.2f})')
    fig.colorbar(im3, ax=ax3, label='Difference')

    plt.tight_layout()
    plt.savefig(f'./figures/predictions_z{z_target:.2f}.png')
    plt.close()