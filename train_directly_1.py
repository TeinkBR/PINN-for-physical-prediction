# 导入必要的库
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import qmc
import torch.optim as optim
import pandas as pd
import os
from mpl_toolkits.mplot3d import Axes3D

# 设置 matplotlib 字体支持中文或回退到英文
try:
    plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC']
except:
    plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False

# 设置随机种子以确保结果可重复
torch.manual_seed(123)

# 检查是否有 GPU 可用
#device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

device = torch.device("cpu")

print(f"正在使用的设备: {device}")

# 创建保存目录
os.makedirs("./figures", exist_ok=True)
os.makedirs("./models", exist_ok=True)

# 定义燃烧室几何模型
def is_inside_combustion_chamber(x, y, z):
    radius = np.sqrt((x - 0.5)**2 + (y - 0.5)**2)
    return (radius <= 0.4) & (z >= 0.05) & (z <= 0.95)

# 定义燃烧室边界条件
def is_on_combustion_chamber_boundary(x, y, z):
    radius = np.sqrt((x - 0.5)**2 + (y - 0.5)**2)
    return ((np.abs(radius - 0.4) < 1e-3) | (np.abs(z - 0.05) < 1e-3) | (np.abs(z - 0.95) < 1e-3)) & (z >= 0.05) & (z <= 0.95)

# 定义 ResPINN 模型，使用 SiLU 激活函数
class ResPINN(nn.Module):
    def __init__(self, num_hidden_layers=12, num_neurons=128):
        super(ResPINN, self).__init__()
        self.num_neurons = num_neurons
        self.input_layer = nn.Linear(6, num_neurons)
        self.hidden_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(num_neurons, num_neurons),
                nn.SiLU()
            ) for _ in range(num_hidden_layers)
        ])
        self.output_layer = nn.Linear(num_neurons, 1)

    def forward(self, x):
        x = self.input_layer(x)
        for layer in self.hidden_layers:
            residual = x
            x = layer(x)
            x = x + residual
        return self.output_layer(x)

# 设置物理参数及方向向量
def kappa_a(x, y, z):
    return -2 * ((x - 0.5)**2 + (y - 0.5)**2 + (z - 0.5)**2) + 5

def kappa_e(x, y, z):
    return kappa_a(x, y, z)

def temperature(x, y, z):
    return -800 * ((x - 0.5)**2 + (y - 0.5)**2 + (z - 0.5)**2) + 2000

def I_b(x, y, z):
    sigma = 5.67e-8
    T = temperature(x, y, z)
    return (sigma * T**4) / np.pi

I_b_w = 0

def epsilon_lambda_w(x, y, z, phi, theta, lambda_):
    base_epsilon = 0.8
    variation = 0.1 * torch.sin(lambda_)
    return base_epsilon + variation

def direction_vector(phi, theta):
    return torch.tensor([
        torch.sin(theta) * torch.cos(phi),
        torch.sin(theta) * torch.sin(phi),
        torch.cos(theta)
    ], dtype=torch.float32)

s = direction_vector(torch.tensor(0.0), torch.tensor(0.0)).to(device)

# 数据归一化参数
KAPPA_SCALE = 5.0

# 加载数据集并处理数据
def load_data(file_path='radiation_data_xyzri_phi10.csv', processed_file='processed_data.pt'):
    if os.path.exists(processed_file):
        print(f"Loading preprocessed data from: {processed_file}")
        try:
            data_dict = torch.load(processed_file, weights_only=False)
            return (data_dict['coords'], data_dict['I_true'], data_dict['I_true_normalized'],
                    data_dict['ri_mean'], data_dict['ri_std'], data_dict['ri_mean_original'], data_dict['ri_std_original'])
        except Exception as e:
            print(f"Failed to load preprocessed file: {str(e)}. Regenerating...")
            os.remove(processed_file)

    try:
        print("Attempting to read CSV with comma delimiter...")
        data = pd.read_csv(file_path)
    except pd.errors.ParserError:
        print("Comma delimiter failed. Trying semicolon...")
        data = pd.read_csv(file_path, sep=';')
    except Exception as e:
        print(f"Failed to read CSV: {str(e)}")
        return None, None, None, None, None, None, None

    print("CSV columns:", data.columns.tolist())
    data.columns = data.columns.str.strip()

    expected_columns = ['x', 'y', 'z', 'ri']
    if set(expected_columns).issubset(data.columns):
        print("Column names match, data loaded successfully!")
    else:
        print("Column names do not match. Assuming first row is data, assigning column names...")
        try:
            data = pd.read_csv(file_path, header=None, names=expected_columns)
            print("Reloaded with column names:", data.columns.tolist())
        except Exception as e:
            print(f"Reload failed: {str(e)}")
            return None, None, None, None, None, None, None

    print("\nFirst few rows of data:\n", data.head())
    print("\nData info:")
    print(data.info())
    print("\nData statistics:")
    print(data.describe())

    ri_mean_original = data['ri'].mean()
    ri_std_original = data['ri'].std()

    ri_mean = data['ri'].mean()
    ri_std = data['ri'].std()
    outlier_threshold = 3
    outliers = (data['ri'] < (ri_mean - outlier_threshold * ri_std)) | (data['ri'] > (ri_mean + outlier_threshold * ri_std))
    print(f"\nDetected {outliers.sum()} outliers. Replacing with mean...")
    data.loc[outliers, 'ri'] = ri_mean

    ri_mean = data['ri'].mean()
    ri_std = data['ri'].std()
    data['ri'] = (data['ri'] - ri_mean) / ri_std

    print("\nNormalized data statistics:")
    print(data.describe())

    coords = torch.tensor(data[['x', 'y', 'z']].values, dtype=torch.float32)
    I_true = torch.tensor(data['ri'].values, dtype=torch.float32)
    I_true_normalized = I_true

    print("\nData loaded successfully!")
    print("\nCoordinate data (first five rows):\n", data[['x', 'y', 'z']].head())
    print("\nRadiation intensity (first five rows):\n", data['ri'].head())

    data_dict = {
        'coords': coords,
        'I_true': I_true,
        'I_true_normalized': I_true_normalized,
        'ri_mean': ri_mean,
        'ri_std': ri_std,
        'ri_mean_original': ri_mean_original,
        'ri_std_original': ri_std_original
    }
    torch.save(data_dict, processed_file)
    print(f"Preprocessed data saved to: {processed_file}")

    return coords, I_true, I_true_normalized, ri_mean, ri_std, ri_mean_original, ri_std_original

# 定义 PINN 损失函数
def compute_loss(model, x_col, x_bc, weights, ri_mean, ri_std):
    phi = torch.zeros_like(x_col[:, 0:1], device=device)
    theta = torch.zeros_like(x_col[:, 0:1], device=device)
    lambda_ = torch.zeros_like(x_col[:, 0:1], device=device)
    x_col_full = torch.cat([x_col, phi, theta, lambda_], dim=1)

    phi_bc = torch.zeros_like(x_bc[:, 0:1], device=device)
    theta_bc = torch.zeros_like(x_bc[:, 0:1], device=device)
    lambda_bc = torch.zeros_like(x_bc[:, 0:1], device=device)
    x_bc_full = torch.cat([x_bc, phi_bc, theta_bc, lambda_bc], dim=1)

    x_col = x_col.requires_grad_(True)
    x_bc = x_bc.requires_grad_(True)
    x_col_full = x_col_full.requires_grad_(True)
    x_bc_full = x_bc_full.requires_grad_(True)

    I_col = model(x_col_full)
    I_bc = model(x_bc_full)

    I_col = torch.clamp(I_col, min=-2.5, max=2.5)
    I_bc = torch.clamp(I_bc, min=-2.5, max=2.5)

    x, y, z = x_col[:, 0:1], x_col[:, 1:2], x_col[:, 2:3]
    grad_I = torch.autograd.grad(I_col, x_col_full, grad_outputs=torch.ones_like(I_col, device=device), create_graph=True)[0]
    dI_dx = grad_I[:, 0:1]
    dI_dy = grad_I[:, 1:2]
    dI_dz = grad_I[:, 2:3]

    s_dot_grad_I = s[0] * dI_dx + s[1] * dI_dy + s[2] * dI_dz

    kappa_e_val = kappa_e(x, y, z) / KAPPA_SCALE
    I_b_val = I_b(x, y, z) / (ri_mean + ri_std)
    I_col_norm = I_col
    L_te = s_dot_grad_I - kappa_e_val * I_b_val + kappa_e_val * I_col_norm

    epsilon_bc = epsilon_lambda_w(x_bc[:, 0:1], x_bc[:, 1:2], x_bc[:, 2:3], phi_bc, theta_bc, lambda_bc)
    I_b_w_bc = torch.tensor(I_b_w, dtype=torch.float32, device=device).expand_as(I_bc)
    L_bc = I_bc - epsilon_bc * I_b_w_bc

    L_pen = 0.0
    param_count = 0
    for name, param in model.named_parameters():
        if 'weight' in name:
            L_pen += torch.sum(param ** 2)
            param_count += param.numel()
    if param_count > 0:
        L_pen = L_pen / param_count
    L_pen = torch.clamp(L_pen, max=1e6)

    w_te, w_bc, w_pen = weights
    loss_te = torch.mean(L_te ** 2)
    loss_bc = torch.mean(L_bc ** 2)
    loss_pen = L_pen
    loss = w_te * loss_te + w_bc * loss_bc + w_pen * loss_pen

    return loss, (loss_te, loss_bc, loss_pen), (L_te, L_bc)

# 计算梯度范数用于 GradNorm
def compute_gradient_norm(model, loss):
    grads = torch.autograd.grad(loss, model.parameters(), create_graph=False, allow_unused=True)
    grads = [g for g in grads if g is not None]
    if len(grads) == 0:
        return torch.tensor(0.0, device=device)
    grad_norm = torch.sqrt(sum([g.norm()**2 for g in grads]))
    return grad_norm

# 自适应权重调整
def update_weights(losses, weights, alpha=0.2, min_weight=0.5, max_weight=20.0):
    loss_te, loss_bc, _ = losses
    w_te, w_bc, w_pen = weights
    loss_te = loss_te.detach()
    loss_bc = loss_bc.detach()
    total_loss = loss_te + loss_bc
    if total_loss > 0:
        w_te_new = w_te * (1 - alpha) + alpha * (loss_te / total_loss) * (loss_te / 3e-2 + 1)
        w_bc_new = w_bc * (1 - alpha) + alpha * (loss_bc / total_loss) * (loss_bc / 5e-7 + 1)
        w_te = min(max(w_te_new, min_weight), max_weight)
        w_bc = min(max(w_bc_new, min_weight), max_weight)
    w_pen = min(w_pen * 0.9, 1e-8)
    return (w_te, w_bc, w_pen)

# 残差自适应采样，使用扰动方法
def residual_adaptive_sampling(model, x_col, x_bc, weights, ri_mean, ri_std, num_new_points=8000):
    try:
        x_col, x_bc = x_col.to(device), x_bc.to(device)

        _, _, (L_te, L_bc) = compute_loss(model, x_col, x_bc, weights, ri_mean, ri_std)

        # Perturbation-based sampling for x_col
        abs_L_te = torch.abs(L_te)
        K = 5  # Number of perturbations per point
        M = min(x_col.shape[0], num_new_points // K)
        if M == 0:
            M = 1
            K = num_new_points
        else:
            K = max(1, num_new_points // M)
        while M * K < num_new_points and M < x_col.shape[0]:
            M += 1
        _, indices = torch.topk(abs_L_te.flatten(), M, largest=True)
        selected_points = x_col[indices]

        sigma = 0.01  # Standard deviation for perturbations
        perturbations = torch.normal(0, sigma, size=(M, K, 3), device=device)
        new_points = selected_points[:, None, :] + perturbations  # Shape (M, K, 3)
        new_points = new_points.view(-1, 3)  # Shape (M*K, 3)

        # Clip to domain bounds
        bounds = torch.tensor([[0.1, 0.9], [0.1, 0.9], [0.05, 0.95]], device=device)
        new_points = torch.max(new_points, bounds[:, 0])
        new_points = torch.min(new_points, bounds[:, 1])

        # Check inside combustion chamber
        new_points_np = new_points.cpu().numpy()
        mask = is_inside_combustion_chamber(new_points_np[:, 0], new_points_np[:, 1], new_points_np[:, 2])
        mask = torch.from_numpy(mask).to(device)
        valid_new_points = new_points[mask]

        if valid_new_points.shape[0] > num_new_points:
            x_col_new = valid_new_points[:num_new_points]
        else:
            x_col_new = valid_new_points
            print(f"Only {x_col_new.shape[0]} valid new points generated for x_col, less than {num_new_points}")
        x_col = torch.cat([x_col, x_col_new], dim=0)

        # Existing boundary sampling
        L_bc_weight = torch.abs(L_bc)
        alpha_bc, beta_bc = 0.8, 0.2
        L_bc_weight = alpha_bc * torch.exp(L_bc_weight) + beta_bc
        L_bc_weight = torch.clamp(L_bc_weight, min=1e-6)
        num_new_points_bc = min(int(num_new_points // 1.2), L_bc_weight.shape[0])
        probs_bc = L_bc_weight / torch.sum(L_bc_weight)
        indices_bc = torch.multinomial(probs_bc.flatten(), num_new_points_bc, replacement=True)
        new_points_bc = x_bc[indices_bc]

        bounds_bc = np.array([[0.1, 0.9], [0.05, 0.95]])
        sampler_2d = qmc.LatinHypercube(d=2)
        sample_bc = sampler_2d.random(n=int(num_new_points // 1.2))
        sample_bc = qmc.scale(sample_bc, bounds_bc[:, 0], bounds_bc[:, 1])
        y_bc = sample_bc[:, 0]
        z_bc = sample_bc[:, 1]
        x_bc_new = np.zeros((len(sample_bc), 3))
        x_bc_new[:, 1] = y_bc
        x_bc_new[:, 2] = z_bc
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
        x_bc_new = torch.tensor(x_bc_new[:int(num_new_points // 1.2)], dtype=torch.float32, device=device)
        x_bc = torch.cat([x_bc, x_bc_new], dim=0)

        return x_col, x_bc
    except Exception as e:
        print(f"Residual adaptive sampling failed: {str(e)}, skipping this round...")
        return x_col, x_bc

# 生成训练数据
def generate_training_data(num_col_points=5000, num_bc_points=2000, batch_size=8000):
    coords, I_true, I_true_normalized, ri_mean, ri_std, ri_mean_original, ri_std_original = load_data()
    if coords is None:
        raise ValueError("Data loading failed, cannot proceed with training.")

    bounds = np.array([
        [0.1, 0.9],
        [0.1, 0.9],
        [0.05, 0.95],
    ])
    sampler = qmc.LatinHypercube(d=3)
    sample_col = sampler.random(n=num_col_points * 2)
    sample_col = qmc.scale(sample_col, bounds[:, 0], bounds[:, 1])
    x, y, z = sample_col[:, 0], sample_col[:, 1], sample_col[:, 2]
    mask = is_inside_combustion_chamber(x, y, z)
    sample_col = sample_col[mask][:num_col_points]
    if len(sample_col) < num_col_points:
        sample_col = np.vstack([sample_col, sample_col[:num_col_points - len(sample_col)]])
    x_col = torch.tensor(sample_col, dtype=torch.float32, device=device)

    bounds_bc = np.array([
        [0.1, 0.9],
        [0.05, 0.95],
    ])
    sampler_2d = qmc.LatinHypercube(d=2)
    sample_bc = sampler_2d.random(n=num_bc_points * 2)
    sample_bc = qmc.scale(sample_bc, bounds_bc[:, 0], bounds_bc[:, 1])
    y_bc = sample_bc[:, 0]
    z_bc = sample_bc[:, 1]
    x_bc = np.zeros((len(sample_bc), 3))
    x_bc[:, 1] = y_bc
    x_bc[:, 2] = z_bc

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

    x_bc = torch.tensor(x_bc[:num_bc_points], dtype=torch.float32, device=device)

    col_dataset = torch.utils.data.TensorDataset(x_col)
    col_loader = torch.utils.data.DataLoader(col_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    bc_dataset = torch.utils.data.TensorDataset(x_bc)
    bc_loader = torch.utils.data.DataLoader(bc_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)

    return col_loader, bc_loader, x_col, x_bc, coords, I_true, ri_mean, ri_std, ri_mean_original, ri_std_original

# 训练模型，采用两步优化（AdamW + L-BFGS）和 GradNorm
def train(model, num_epochs=200, batch_size=8000, num_adamw_epochs=150):
    col_loader, bc_loader, x_col, x_bc, full_coords, full_I, ri_mean, ri_std, ri_mean_original, ri_std_original = generate_training_data(batch_size=batch_size)

    model = model.to(device)
    optimizer_adamw = optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer_adamw, T_0=20, T_mult=2, eta_min=1e-5)

    weights = (1.0, 1.0, 1e-7)

    history = {
        'total_loss': [],
        'loss_te': [],
        'loss_bc': [],
        'loss_pen': [],
        'bc_accuracy': [],
        'accuracy': [],
        'w_te': [],
        'w_bc': [],
        'w_pen': [],
        'lr': []
    }

    print("Starting training with AdamW optimizer...")
    for epoch in range(num_epochs):
        if epoch < num_adamw_epochs:
            optimizer = optimizer_adamw
            use_lbfgs = False
        else:
            if 'optimizer_lbfgs' not in locals():
                optimizer_lbfgs = optim.LBFGS(model.parameters(), lr=0.1, max_iter=20)
            optimizer = optimizer_lbfgs
            use_lbfgs = True

        model.train()
        total_loss_epoch = 0.0
        loss_te_epoch = 0.0
        loss_bc_epoch = 0.0
        loss_pen_epoch = 0.0
        bc_accuracy_epoch = 0.0
        accuracy_epoch = 0.0
        num_batches = 0

        col_iter = iter(col_loader)
        bc_iter = iter(bc_loader)
        while True:
            try:
                x_col_batch = next(col_iter)[0]
                try:
                    x_bc_batch = next(bc_iter)[0]
                except StopIteration:
                    bc_iter = iter(bc_loader)
                    x_bc_batch = next(bc_iter)[0]
            except StopIteration:
                break

            x_col_batch, x_bc_batch = x_col_batch.to(device), x_bc_batch.to(device)

            if use_lbfgs:
                def closure():
                    optimizer.zero_grad()
                    loss, _, _ = compute_loss(model, x_col_batch, x_bc_batch, weights, ri_mean, ri_std)
                    loss.backward()
                    return loss
                optimizer.step(closure)
                loss, losses, residuals = compute_loss(model, x_col_batch, x_bc_batch, weights, ri_mean, ri_std)
            else:
                optimizer.zero_grad()
                loss, losses, residuals = compute_loss(model, x_col_batch, x_bc_batch, weights, ri_mean, ri_std)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.05)
                optimizer.step()

            loss_te, loss_bc, loss_pen = losses
            total_loss_epoch
# 可视化预测结果
def evaluate_model(model, full_coords, full_I, ri_mean, ri_std, ri_mean_original, ri_std_original, batch_size=10000):
    """
    评估模型并可视化预测结果
    参数：
    - model: 训练后的模型
    - full_coords, full_I: 测试数据
    - ri_mean, ri_std, ri_mean_original, ri_std_original: 标准化参数
    - batch_size: 评估批次大小
    """
    test_data = full_coords.to(device)
    true_I = full_I.numpy()
    num_samples = len(test_data)
    pred_I = []

    model.eval()
    with torch.no_grad():
        for i in range(0, num_samples, batch_size):
            batch_data = test_data[i:i + batch_size]
            phi = torch.zeros_like(batch_data[:, 0:1], device=device)
            theta = torch.zeros_like(batch_data[:, 0:1], device=device)
            lambda_ = torch.zeros_like(batch_data[:, 0:1], device=device)
            batch_data_full = torch.cat([batch_data, phi, theta, lambda_], dim=1)
            batch_pred = model(batch_data_full).cpu().numpy().flatten()
            pred_I.append(batch_pred)

    pred_I = np.concatenate(pred_I, axis=0)

    pred_I = pred_I * ri_std_original + ri_mean_original
    true_I = true_I * ri_std_original + ri_mean_original

    print(f"预测值范围: min={pred_I.min():.6f}, max={pred_I.max():.6f}")
    print(f"真实值范围: min={true_I.min():.6f}, max={true_I.max():.6f}")

    abs_error = np.abs(pred_I - true_I)
    rel_error = abs_error / (np.abs(true_I) + 1e-6)
    print(f"平均绝对误差: {np.mean(abs_error):.6f}")
    print(f"平均相对误差: {np.mean(rel_error):.6f}")
    print(f"最大绝对误差: {np.max(abs_error):.6f}")
    print(f"最小绝对误差: {np.min(abs_error):.6f}")
    print(f"绝对误差中位数: {np.median(abs_error):.6f}")

    nx, ny, nz = 100, 100, 99
    z_values = np.linspace(0, 98, 99, dtype=int)
    errors = []
    z_slices = []

    for z_slice in z_values:
        I_pred_slice = pred_I.reshape(nx, ny, nz)[:, :, z_slice]
        I_true_slice = true_I.reshape(nx, ny, nz)[:, :, z_slice]
        slice_error = np.mean(np.abs(I_pred_slice - I_true_slice))
        errors.append(slice_error)
        z_slices.append(z_slice)
        z_value = 0.05 + z_slice * (0.95 - 0.05) / (nz - 1)
        print(f"z_slice={z_slice}, z={z_value:.2f}, 平均绝对误差: {slice_error:.6f}")

    sorted_indices = np.argsort(errors)
    best_z_slice_1 = z_slices[sorted_indices[0]]
    best_z_slice_2 = z_slices[sorted_indices[1]]
    z_value_1 = 0.05 + best_z_slice_1 * (0.95 - 0.05) / (nz - 1)
    z_value_2 = 0.05 + best_z_slice_2 * (0.95 - 0.05) / (nz - 1)

    I_pred_slice_1 = pred_I.reshape(nx, ny, nz)[:, :, best_z_slice_1]
    I_true_slice_1 = true_I.reshape(nx, ny, nz)[:, :, best_z_slice_1]

    x = np.linspace(0.1, 0.9, nx)
    y = np.linspace(0.1, 0.9, ny)
    X, Y = np.meshgrid(x, y, indexing='ij')

    plt.figure(figsize=(18, 5), dpi=150)
    plt.subplot(1, 3, 1)
    plt.contourf(X, Y, I_pred_slice_1, levels=50, cmap='jet', alpha=0.8)
    plt.colorbar(label='预测辐射强度', pad=0.02)
    plt.xlabel('x 坐标', fontsize=12)
    plt.ylabel('y 坐标', fontsize=12)
    plt.title(f'预测辐射强度 (z={z_value_1:.2f})', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.5)

    plt.subplot(1, 3, 2)
    plt.contourf(X, Y, I_true_slice_1, levels=50, cmap='jet', alpha=0.8)
    plt.colorbar(label='真实辐射强度', pad=0.02)
    plt.xlabel('x 坐标', fontsize=12)
    plt.ylabel('y 坐标', fontsize=12)
    plt.title(f'真实辐射强度 (z={z_value_1:.2f})', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.5)

    diff = I_pred_slice_1 - I_true_slice_1
    plt.subplot(1, 3, 3)
    plt.contourf(X, Y, diff, levels=50, cmap='RdBu', alpha=0.8)
    plt.colorbar(label='预测-真实差值', pad=0.02)
    plt.xlabel('x 坐标', fontsize=12)
    plt.ylabel('y 坐标', fontsize=12)
    plt.title(f'差值 (z={z_value_1:.2f})', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig("./figures/prediction_comparison_1.png")
    plt.show()
    print(f"预测结果对比图 (z={z_value_1:.2f}) 已保存至 ./figures/prediction_comparison_1.png")

    I_pred_slice_2 = pred_I.reshape(nx, ny, nz)[:, :, best_z_slice_2]
    I_true_slice_2 = true_I.reshape(nx, ny, nz)[:, :, best_z_slice_2]

    plt.figure(figsize=(18, 5), dpi=150)
    plt.subplot(1, 3, 1)
    plt.contourf(X, Y, I_pred_slice_2, levels=50, cmap='jet', alpha=0.8)
    plt.colorbar(label='预测辐射强度', pad=0.02)
    plt.xlabel('x 坐标', fontsize=12)
    plt.ylabel('y 坐标', fontsize=12)
    plt.title(f'预测辐射强度 (z={z_value_2:.2f})', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.5)

    plt.subplot(1, 3, 2)
    plt.contourf(X, Y, I_true_slice_2, levels=50, cmap='jet', alpha=0.8)
    plt.colorbar(label='真实辐射强度', pad=0.02)
    plt.xlabel('x 坐标', fontsize=12)
    plt.ylabel('y 坐标', fontsize=12)
    plt.title(f'真实辐射强度 (z={z_value_2:.2f})', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.5)

    diff = I_pred_slice_2 - I_true_slice_2
    plt.subplot(1, 3, 3)
    plt.contourf(X, Y, diff, levels=50, cmap='RdBu', alpha=0.8)
    plt.colorbar(label='预测-真实差值', pad=0.02)
    plt.xlabel('x 坐标', fontsize=12)
    plt.ylabel('y 坐标', fontsize=12)
    plt.title(f'差值 (z={z_value_2:.2f})', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig("./figures/prediction_comparison_2.png")
    plt.show()
    print(f"预测结果对比图 (z={z_value_2:.2f}) 已保存至 ./figures/prediction_comparison_2.png")

    fig = plt.figure(figsize=(10, 8), dpi=150)
    ax = fig.add_subplot(111, projection='3d')
    mask = is_inside_combustion_chamber(full_coords[:, 0].numpy(), full_coords[:, 1].numpy(), full_coords[:, 2].numpy())
    scatter = ax.scatter(full_coords[mask, 0].numpy(), full_coords[mask, 1].numpy(), full_coords[mask, 2].numpy(),
                         c=pred_I[mask], cmap='jet', s=10, alpha=0.6)
    ax.set_xlabel('x 坐标', fontsize=12)
    ax.set_ylabel('y 坐标', fontsize=12)
    ax.set_zlabel('z 坐标', fontsize=12)
    ax.set_title('燃烧室内预测辐射强度分布 (三维视图)', fontsize=14)
    plt.colorbar(scatter, ax=ax, label='辐射强度', pad=0.1)
    plt.savefig("./figures/prediction_3d.png")
    plt.show()
    print("三维预测结果图已保存至 ./figures/prediction_3d.png")

# 可视化训练历史
def plot_history(history):
    """
    可视化训练历史
    参数：
    - history: 训练历史记录
    """
    epochs = range(len(history['total_loss']))

    plt.figure(figsize=(15, 10), dpi=150)
    plt.subplot(2, 2, 1)
    plt.semilogy(epochs, history['total_loss'], label='总损失', color='blue')
    plt.semilogy(epochs, history['loss_te'], label='传输方程损失', color='green')
    plt.semilogy(epochs, history['loss_bc'], label='边界条件损失', color='red')
    plt.semilogy(epochs, history['loss_pen'], label='正则化损失', color='purple')
    plt.xlabel('轮次', fontsize=12)
    plt.ylabel('损失值 (对数尺度)', fontsize=12)
    plt.title('损失随轮次变化', fontsize=14)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)

    plt.subplot(2, 2, 2)
    plt.semilogy(epochs, history['bc_accuracy'], label='边界准确度 (平均误差)', color='orange')
    plt.xlabel('轮次', fontsize=12)
    plt.ylabel('绝对误差 (对数尺度)', fontsize=12)
    plt.title('边界准确度随轮次变化', fontsize=14)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)

    plt.subplot(2, 2, 3)
    plt.plot(epochs, history['w_te'], label='w_te', color='green')
    plt.plot(epochs, history['w_bc'], label='w_bc', color='red')
    plt.plot(epochs, history['w_pen'], label='w_pen', color='purple')
    plt.xlabel('轮次', fontsize=12)
    plt.ylabel('权重值', fontsize=12)
    plt.title('权重随轮次变化', fontsize=14)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)

    plt.subplot(2, 2, 4)
    plt.plot(epochs, history['accuracy'], label='准确度 (%)', color='blue')
    plt.xlabel('轮次', fontsize=12)
    plt.ylabel('准确度 (%)', fontsize=12)
    plt.title('准确度随轮次变化', fontsize=14)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig("./figures/training_metrics.png")
    plt.show()
    print("训练指标图已保存至 ./figures/training_metrics.png")

# 主程序
if __name__ == "__main__":
    num_col_points = 7000
    num_bc_points = 5000

    model = ResPINN(num_hidden_layers=12, num_neurons=128)
    model, history, full_coords, full_I, ri_mean, ri_std, ri_mean_original, ri_std_original = train(model, num_epochs=200)
    plot_history(history)
    evaluate_model(model, full_coords, full_I, ri_mean, ri_std, ri_mean_original, ri_std_original)
