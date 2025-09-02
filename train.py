import torch
device = torch.device("cpu")
import torch.optim as optim
from loss import compute_loss, update_weights
from sampling import generate_training_data, residual_adaptive_sampling
import numpy as np
# Try to import the compiled C++ extension; if missing, build it in-place using PyTorch's cpp_extension
try:
    import pinn_loss  # C++ module
except Exception:
    # Delay heavy imports until needed
    from torch.utils.cpp_extension import load
    import os
    here = os.path.dirname(__file__)
    src = os.path.join(here, 'pinn_loss.cpp')
    build_dir = os.path.join(here, 'build_ext')
    os.makedirs(build_dir, exist_ok=True)
    pinn_loss = load(name='pinn_loss', sources=[src], build_directory=build_dir, verbose=True)

def compute_mre(model, coords, I_true, ri_mean, ri_std, device, boundary_only=False):
    model.eval()
    phi = torch.rand(coords.shape[0], 1, device=device) * 2 * np.pi
    theta = torch.rand(coords.shape[0], 1, device=device) * np.pi
    lam = torch.full((coords.shape[0], 1), 1e-6, device=device)
    inputs = torch.cat([coords, phi, theta, lam], dim=1).requires_grad_(False)
    with torch.no_grad():
        predictions = model(inputs).squeeze()
    predictions = predictions * ri_std + ri_mean
    I_true = I_true * ri_std + ri_mean
    if boundary_only:
        from physics import is_on_nozzle_boundary
        x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
        mask = is_on_nozzle_boundary(x, y, z)
        predictions = predictions[mask]
        I_true = I_true[mask]
    abs_error = torch.abs(predictions - I_true)
    rel_error = abs_error / (I_true + 1e-10)
    mre = torch.mean(rel_error).item() if len(rel_error) > 0 else float('nan')
    return mre



def train_model(model, num_epochs=500, batch_size=4096, accumulation_steps=4, T_wall=0.0, patience=200):
    # Set device (MPS if available, else CPU)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    # Load data
    col_loader, bc_loader, x_col, x_bc, coords, I_true, ri_mean, ri_std, ri_mean_original, ri_std_original = generate_training_data(batch_size=batch_size)
    
    # Move model to device
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20, verbose=True)
    best_loss = float('inf')
    epochs_no_improve = 0

    weights = [1.0, 5.0, 1e-7]
    history = {
        'total_loss': [], 'loss_te': [], 'loss_bc': [], 'loss_pen': [],
        'bc_accuracy': [], 'accuracy': [], 'w_te': [], 'w_bc': [], 'w_pen': [], 'lr': [], 'mre': [], 'mre_bc': []
    }

    print("Starting training with AdamW optimizer...")
    for epoch in range(num_epochs):
        model.train()
        total_loss_epoch, loss_te_epoch, loss_bc_epoch, loss_pen_epoch = 0.0, 0.0, 0.0, 0.0
        bc_accuracy_epoch, accuracy_epoch, num_batches = 0.0, 0.0, 0

        col_iter = iter(col_loader)
        bc_iter = iter(bc_loader)
        optimizer.zero_grad()
        
        for _ in range(accumulation_steps):
            try:
                x_col_batch = next(col_iter)[0].to(device)
                try:
                    x_bc_batch = next(bc_iter)[0].to(device)
                except StopIteration:
                    bc_iter = iter(bc_loader)
                    x_bc_batch = next(bc_iter)[0].to(device)
            except StopIteration:
                break

            # Save model state
            model_state_path = "./model_state.pt"
            torch.save(model.state_dict(), model_state_path)

            # Call C++ loss function
            num_threads = 1 if device.type == "mps" else 4
            device_type = "mps" if device.type == "mps" else "cpu"
            loss, loss_te, loss_bc, loss_pen = pinn_loss.compute_loss(
                x_col_batch, x_bc_batch, weights, ri_mean, ri_std,
                model_state_path, num_threads, device_type
            )
            
            # Scale loss for gradient accumulation
            loss = loss / accumulation_steps
            loss.backward()
            
            if (_ + 1) % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
                optimizer.step()
                optimizer.zero_grad()

            # Accumulate loss metrics
            total_loss_epoch += loss.item() * accumulation_steps
            loss_te_epoch += loss_te.item()
            loss_bc_epoch += loss_bc.item()
            loss_pen_epoch += loss_pen.item()

            # Compute boundary accuracy (using Python compute_loss for residuals)
            _, _, (L_te, L_bc) = compute_loss(model, x_col_batch, x_bc_batch, weights, ri_mean, ri_std, device)
            bc_accuracy = torch.mean(torch.abs(L_bc))
            bc_accuracy_epoch += bc_accuracy.item()
            correct = torch.abs(L_bc) < 1e-3
            accuracy = torch.mean(correct.float()) * 100
            accuracy_epoch += accuracy.item()
            num_batches += 1

        # Update scheduler
        scheduler.step(total_loss_epoch / num_batches)
        
        # Average metrics
        avg_metrics = lambda x: x / num_batches if num_batches > 0 else 0
        history['total_loss'].append(avg_metrics(total_loss_epoch))
        history['loss_te'].append(avg_metrics(loss_te_epoch))
        history['loss_bc'].append(avg_metrics(loss_bc_epoch))
        history['loss_pen'].append(avg_metrics(loss_pen_epoch))
        history['bc_accuracy'].append(avg_metrics(bc_accuracy_epoch))
        history['accuracy'].append(avg_metrics(accuracy_epoch))
        history['w_te'].append(weights[0])
        history['w_bc'].append(weights[1])
        history['w_pen'].append(weights[2])
        history['lr'].append(optimizer.param_groups[0]['lr'])

        # Update weights
        weights = update_weights((loss_te, loss_bc, loss_pen), weights)

        # Compute MRE every 10 epochs
        if epoch % 10 == 0:
            mre = compute_mre(model, coords, I_true, ri_mean, ri_std, device)
            mre_bc = compute_mre(model, coords, I_true, ri_mean, ri_std, device, boundary_only=True)
            history['mre'].append(mre)
            history['mre_bc'].append(mre_bc)
            print(f"Epoch {epoch}, MRE: {mre:.4f}, MRE_BC: {mre_bc:.4f}")

        # Residual adaptive sampling every 5 epochs
        if epoch % 5 == 0 and epoch > 0:
            x_col, x_bc = residual_adaptive_sampling(model, x_col, x_bc, weights, ri_mean, ri_std, num_new_points=8000, device=device)
            col_dataset = torch.utils.data.TensorDataset(x_col.cpu())
            col_loader = torch.utils.data.DataLoader(col_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
            bc_dataset = torch.utils.data.TensorDataset(x_bc.cpu())
            bc_loader = torch.utils.data.DataLoader(bc_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
            x_col, x_bc = x_col.to(device), x_bc.to(device)

        # Early stopping
        if total_loss_epoch / num_batches < best_loss:
            best_loss = total_loss_epoch / num_batches
            epochs_no_improve = 0
            torch.save(model.state_dict(), './models/pinn_model_best.pt')
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered at epoch {epoch}")
                break

        if epoch % 4 == 0:
            print(f'Epoch {epoch}, Total Loss: {history["total_loss"][-1]:.6f}, '
                  f'L_te: {history["loss_te"][-1]:.6f}, L_bc: {history["loss_bc"][-1]:.6f}, '
                  f'L_pen: {history["loss_pen"][-1]:.6f}, Boundary Accuracy: {history["bc_accuracy"][-1]:.6f}, '
                  f'Learning Rate: {history["lr"][-1]:.6f}')

    torch.save(model.state_dict(), './models/pinn_model_final.pt')
    print("Model saved to ./models/pinn_model_final.pt")

    return model, history, coords, I_true, ri_mean, ri_std, ri_mean_original, ri_std_original