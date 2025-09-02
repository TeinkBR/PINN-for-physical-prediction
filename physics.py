import numpy as np
import torch
device = torch.device("cpu")
#device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")


def kappa_a(x, y, z):
    return -2 * ((x - 0.5)**2 + (y - 0.5)**2 + (z - 0.5)**2) + 5

def kappa_e(x, y, z):
    return kappa_a(x, y, z)

def temperature(x, y, z):
    return -800 * ((x - 0.5)**2 + (y - 0.5)**2 + (z - 0.5)**2) + 2000


def I_b(x, y, z, lambda_val):
    c1 = 3.741832e8  # W·μm⁴/m²
    c2 = 1.4388e4    # μm·K
    T = temperature(x, y, z)
    lambda_um = lambda_val * 1e6  # Convert m to μm
    exponent = c2 / (lambda_um * T)
    I_b_lambda = (c1 / (lambda_um**5)) / (torch.exp(exponent) - 1)
    return I_b_lambda

I_b_w = 0

def epsilon_lambda_w(x, y, z, phi, theta, lambda_):
    base_epsilon = 0.8
    variation = 0.1 * torch.sin(lambda_)
    return base_epsilon + variation

def direction_vector(phi, theta):
    return torch.stack([
        torch.sin(theta.squeeze()) * torch.cos(phi.squeeze()),
        torch.sin(theta.squeeze()) * torch.sin(phi.squeeze()),
        torch.cos(theta.squeeze())
    ], dim=1).to(phi.device)


#s = direction_vector(torch.tensor(0.0), torch.tensor(0.0))

KAPPA_SCALE = 5.0

def is_inside_nozzle(x, y, z):
    radius = torch.sqrt((x - 0.5)**2 + (y - 0.5)**2)
    return (radius <= 0.4) & (z >= 0.05) & (z <= 0.95)

def is_on_nozzle_boundary(x, y, z):
    radius = torch.sqrt((x - 0.5)**2 + (y - 0.5)**2)
    return ((torch.abs(radius - 0.4) < 1e-3) | (torch.abs(z - 0.05) < 1e-3) | (torch.abs(z - 0.95) < 1e-3)) & (z >= 0.05) & (z <= 0.95)

# PLACEHOLDER FOR NOZZLE GEOMETRY - TO BE IMPLEMENTED
# [Previous placeholder code remains unchanged]




# PLACEHOLDER FOR NOZZLE GEOMETRY - TO BE IMPLEMENTED
# def is_inside_nozzle(x, y, z):
#     """
#     Implement actual nozzle geometry here
#     Typical converging-diverging nozzle profile:
#         Converging section: z ∈ [z_min, z_throat]
#         Throat: narrowest point at z = z_throat
#         Diverging section: z ∈ [z_throat, z_max]
#     """
#     # Parameters to be defined based on actual nozzle dimensions
#     z_throat = 0.5  # Example throat position
#     r_inlet = 0.4   # Example inlet radius
#     r_throat = 0.1  # Example throat radius
#     r_exit = 0.3    # Example exit radius
#     
#     # Calculate radius based on z-position
#     if z < z_throat:  # Converging section
#         current_radius = r_inlet - (r_inlet - r_throat) * (z - z_min)/(z_throat - z_min)
#     else:  # Diverging section
#         current_radius = r_throat + (r_exit - r_throat) * (z - z_throat)/(z_max - z_throat)
#     
#     radius = np.sqrt((x - 0.5)**2 + (y - 0.5)**2)
#     return radius <= current_radius
