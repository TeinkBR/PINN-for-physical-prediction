import torch
from physics import direction_vector, kappa_e, I_b, epsilon_lambda_w, KAPPA_SCALE
import numpy as np

device = torch.device("cpu")

def compute_loss(model, x_col, x_bc, x_data, I_true_data, weights, ri_mean, ri_std, device):
    # Existing code for PDE residual and boundary loss
    batch_size = x_col.shape[0]
    phi = torch.rand(1, device=device) * 2 * np.pi
    theta = torch.rand(1, device=device) * np.pi
    lambda_ = torch.full((batch_size, 1), 1e-6, device=device)
    phi = phi.repeat(batch_size, 1)
    theta = theta.repeat(batch_size, 1)
    x_col_full = torch.cat([x_col, phi, theta, lambda_], dim=1)
    
    batch_size_bc = x_bc.shape[0]
    phi_bc = phi[:batch_size_bc]
    theta_bc = theta[:batch_size_bc]
    lambda_bc = torch.full((batch_size_bc, 1), 1e-6, device=device)
    x_bc_full = torch.cat([x_bc, phi_bc, theta_bc, lambda_bc], dim=1)
    
    x_col_full = x_col_full.requires_grad_(True)
    x_bc_full = x_bc_full.requires_grad_(True)
    
    I_col = model(x_col_full)
    I_bc = model(x_bc_full)
    
    # PDE residual computation
    x, y, z = x_col_full[:, 0:1], x_col_full[:, 1:2], x_col_full[:, 2:3]
    phi, theta, lambda_ = x_col_full[:, 3:4], x_col_full[:, 4:5], x_col_full[:, 5:6]
    grad_I = torch.autograd.grad(I_col, x_col_full, grad_outputs=torch.ones_like(I_col), create_graph=True)[0]
    dI_dx, dI_dy, dI_dz = grad_I[:, 0:1], grad_I[:, 1:2], grad_I[:, 2:3]
    s = direction_vector(phi, theta).to(device)
    s_dot_grad_I = s[:, 0:1] * dI_dx + s[:, 1:2] * dI_dy + s[:, 2:3] * dI_dz
    kappa_e_val = kappa_e(x, y, z) / KAPPA_SCALE
    I_b_val = I_b(x, y, z, lambda_)
    I_b_norm = (I_b_val - ri_mean) / ri_std
    I_col_norm = I_col
    L_te = s_dot_grad_I + kappa_e_val * (I_col_norm - I_b_norm)
    
    # Boundary condition loss
    x_bc, y_bc, z_bc = x_bc_full[:, 0:1], x_bc_full[:, 1:2], x_bc_full[:, 2:3]
    epsilon_bc = epsilon_lambda_w(x_bc, y_bc, z_bc, phi_bc, theta_bc, lambda_bc)
    I_b_w_bc = torch.zeros_like(I_bc)
    L_bc = I_bc - epsilon_bc * I_b_w_bc
    
    # Penalty loss
    L_pen = 0.0
    param_count = 0
    for name, param in model.named_parameters():
        if 'weight' in name:
            L_pen += torch.sum(param ** 2)
            param_count += param.numel()
    if param_count > 0:
        L_pen = L_pen / param_count
    L_pen = torch.clamp(L_pen, max=1e6)
    
    # New: Data loss
    I_pred_data = model(x_data)
    loss_data = torch.mean((I_pred_data - I_true_data) ** 2)
    
    # Combine losses
    w_te, w_bc, w_pen, w_data = weights
    loss_te = torch.mean(L_te ** 2)
    loss_bc = torch.mean(L_bc ** 2)
    loss_pen = L_pen
    loss = w_te * loss_te + w_bc * loss_bc + w_pen * loss_pen + w_data * loss_data
    
    return loss, (loss_te, loss_bc, loss_pen, loss_data), (L_te, L_bc)

def update_weights(losses, weights, alpha=0.1, min_weight=0.5, max_weight=20.0):
    loss_te, loss_bc, _ = losses
    w_te, w_bc, w_pen = weights
    loss_te, loss_bc = loss_te.detach(), loss_bc.detach()
    total_loss = loss_te + loss_bc
    if total_loss > 0:
        w_te_new = w_te * (1 - alpha) + alpha * (loss_te / total_loss) * (loss_te / 3e-2 + 1)
        w_bc_new = w_bc * (1 - alpha) + alpha * (loss_bc / total_loss) * (loss_bc / 5e-7 + 1)
        w_te = min(max(w_te_new, min_weight), max_weight)
        w_bc = min(max(w_bc_new, min_weight), max_weight)
    w_pen = min(w_pen * 0.9, 1e-8)
    return (w_te, w_bc, w_pen)