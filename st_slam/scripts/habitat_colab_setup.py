#!/usr/bin/env python3
"""Starter script for the Habitat-Sim st_slam 5.0 integration.
Usage in Colab:
  !python habitat_colab_setup.py

This sets up the environment and prints instructions for the experiments.
"""

SETUP_COMMANDS = """
# ==========================================================
# st_slam 5.0 + Habitat-Sim Colab Environment Setup
# ==========================================================

# 1. Install Miniconda
!wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
!bash Miniconda3-latest-Linux-x86_64.sh -b -f -p /usr/local
!rm Miniconda3-latest-Linux-x86_64.sh

import sys
sys.path.insert(0, "/usr/local/lib/python3.10/site-packages")

# 2. Install habitat-sim and habitat-lab
!conda install -y -c conda-forge -c aihabitat habitat-sim=0.3.3 headless bullet python=3.9 cmake
!git clone --branch stable https://github.com/facebookresearch/habitat-lab.git /content/habitat-lab
!pip install -e /content/habitat-lab/habitat-lab

# 3. Download Gibson Tiny (~1.5GB)
!python -m habitat_sim.utils.datasets_download --uids habitat_test_scenes --data-path /content/data

# 4. Build st_slam C++ bridge
!apt-get install -y -qq libgoogle-glog-dev libgflags-dev libatlas-base-dev libsuitesparse-dev
!git clone https://github.com/ceres-solver/ceres-solver.git /content/ceres-solver
!mkdir -p /content/ceres-solver/build && cd /content/ceres-solver/build
!cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF -DBUILD_EXAMPLES=OFF
!make -j4 install
!rm -rf /content/ceres-solver

# 5. Clone st_slam and build
!git clone <YOUR_ST_SLAM_REPO> /content/st_slam
!cd /content/st_slam && pip install pybind11
!mkdir -p /content/st_slam/build && cd /content/st_slam/build
!cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_PYTHON_BINDINGS=ON -Dpybind11_DIR=$(python3 -c "import pybind11; print(pybind11.get_cmake_dir())")
!make st_slam_tracker -j4
!cp python/st_slam_tracker*.so /usr/local/lib/python3.9/site-packages/

print("\\n=== Setup complete! Run experiments below ===")
"""

EXPERIMENT_COMMANDS = """
# === Experiment 1: ATE drift vs path length ===
import habitat
import st_slam_tracker
import numpy as np

env = habitat.Env(config=habitat.get_config("benchmark/nav/pointnav/pointnav_gibson.yaml"))
tracker = st_slam_tracker.Tracker()

obs = env.reset()
poses_global, poses_smooth = [], []
while not env.episode_over:
    tracker.step(obs["rgb"], obs["depth"])
    poses_global.append(tracker.get_global_pose())
    poses_smooth.append(tracker.get_smooth_pose())
    action = {"action": "move_forward", "action_args": {"forward_amount": 0.25}}
    obs = env.step(action)

# Compute ATE
gt = env.get_metrics()["position"]
print(f"ATE global: {np.linalg.norm(np.array(poses_global[-1][:3]) - gt):.3f}m")
print(f"ATE smooth: {np.linalg.norm(np.array(poses_smooth[-1][:3]) - gt):.3f}m")

# === Experiment 2: Jerk ablation ===
print(f"Loops detected: {tracker.get_num_loops()}")
print(f"Keyframes: {tracker.get_num_kfs()}")
print(f"MapPoints: {tracker.get_num_mps()}")
"""

if __name__ == "__main__":
    print("=" * 70)
    print("  st_slam 5.0 + Habitat-Sim Colab Setup")
    print("=" * 70)
    print("To run, paste into a Colab cell and execute:")
    print(SETUP_COMMANDS)
    print()
    print("After setup, run Experiment 1 & 2:")
    print(EXPERIMENT_COMMANDS)
