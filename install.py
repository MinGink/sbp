
#install
import os
import subprocess

# Install PyTorch and its related packages
torch_install_command = "pip3 install torch torchvision torchaudio"
subprocess.call(torch_install_command, shell=True)

# Define the required packages and versions
packages = [
    "numpy",
    "pfrl",
    "setuptools==65.5.0",
    "gym==0.21",
    "procgen",
    "packaging",
    "pandas",
    "matplotlib",
    "wandb",
    "gymnasium",
    "tensorboard",
]

# Install the required packages
for package in packages:
    install_command = f"pip3 install {package}"
    subprocess.call(install_command, shell=True)

print("Installation completed.")
