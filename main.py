#!/usr/bin/env python3
import os
import sys
import subprocess
import shutil
import numpy as np  # Import before torch
import torch

# Function to create and activate virtual environment
def setup_virtual_env(venv_name='pinn_env'):
    """Create a virtual environment if it doesn't exist."""
    if not os.path.exists(venv_name):
        print(f"Creating virtual environment {venv_name}...")
        subprocess.check_call([sys.executable, '-m', 'venv', venv_name])
    
    # Corrected platform detection
    if sys.platform == 'win32':
        # Windows paths
        bin_dir = 'Scripts'
        activate_script = os.path.join(venv_name, bin_dir, 'activate.bat')
        python_exe = 'python.exe'
        pip_exe = 'pip.exe'
    else:
        # macOS and Linux paths
        bin_dir = 'bin'
        activate_script = os.path.join(venv_name, bin_dir, 'activate')
        python_exe = 'python'
        pip_exe = 'pip'
    
    python_path = os.path.join(venv_name, bin_dir, python_exe)
    pip_path = os.path.join(venv_name, bin_dir, pip_exe)
    
    # Verify paths exist
    if not os.path.exists(python_path):
        raise FileNotFoundError(f"Python not found at: {python_path}")
    if not os.path.exists(pip_path):
        raise FileNotFoundError(f"Pip not found at: {pip_path}")
    
    return activate_script, python_path, pip_path

# Function to install required packages
def install_packages(pip_path):
    """Install necessary Python packages into the virtual environment."""
    print("Installing required packages...")
    packages = [
        'numpy==1.26.4',  # Must be installed BEFORE torch
        'torch==2.2.0',
        'scipy==1.13.0',
        'matplotlib==3.9.0',
        'pandas==2.2.2'
    ]
    try:
        subprocess.check_call([pip_path, 'install', '--upgrade', 'pip'])
        subprocess.check_call([pip_path, 'install', 'numpy==1.26.4'])  # Install numpy first
        subprocess.check_call([pip_path, 'install'] + packages[1:])  # Then other packages
        print("Packages installed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Failed to install packages: {e}")
        raise

# Function to run the main simulation and plotting with debugging
def run_main_script():
    """Execute the simulation and generate plots with error handling."""
    print("Starting run_main_script")
    try:
        from model import ResPINN
        from train import train_model
        from visualize import plot_history, plot_predictions, plot_raw_data
        from sampling import generate_training_data

        print("Initializing model")
        pinn_model = ResPINN(num_hidden_layers=6, num_neurons=64)
        print("Model initialized")

        print("Starting training")
        # Call train_model with only the required parameters
        pinn_model, history = train_model(
            pinn_model, 
            num_epochs=50, 
            batch_size=500
        )
        print("Training completed")

        print("Plotting history")
        plot_history(history)
        
        print("Plotting predictions")
        # We'll need to get the data for plotting - perhaps from sampling or data modules
        try:
            from data import load_data
            coords, I_true, _, ri_mean, ri_std, _, _ = load_data()
            plot_predictions(pinn_model, coords, I_true, ri_mean, ri_std, z_target=0.7)
        except Exception as e:
            print(f"Couldn't plot predictions: {e}")
        
        print("Plotting raw data")
        try:
            from data import load_data
            coords, I_true, _, ri_mean, ri_std, _, _ = load_data()
            plot_raw_data(coords, I_true, ri_mean, ri_std)
        except Exception as e:
            print(f"Couldn't plot raw data: {e}")
            
    except Exception as e:
        print(f"Error in run_main_script: {e}")
        raise

# Main function to orchestrate the setup and execution
def main():
    """Set up the virtual environment, install packages, and run the simulation."""
    venv_name = 'pinn_env'
    try:
        activate_script, python_path, pip_path = setup_virtual_env(venv_name)
        
        # Install packages
        install_packages(pip_path)
        
        # Run the script within the virtual environment
        script_path = os.path.abspath(__file__)
        subprocess.check_call([python_path, script_path, '--in-venv'])
    except Exception as e:
        print(f"Error in main: {e}")
        print("Troubleshooting steps:")
        print("1. Delete the virtual environment: rm -rf pinn_env")
        print("2. Check your Python installation: python --version")
        print("3. Try creating a venv manually: python -m venv test_env")
        raise

if __name__ == '__main__':
    if '--in-venv' in sys.argv:
        run_main_script()
    else:
        main()