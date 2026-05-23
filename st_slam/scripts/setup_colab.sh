#!/bin/bash
# ST-SLAM 4.0 Colab Setup Script
# Run this in Google Colab:
#   !git clone git@github.com:buaa-zy-2239/st_slam.git
#   !cd st_slam && bash scripts/setup_colab.sh

set -e

echo "=========================================="
echo " ST-SLAM 4.0 Colab Environment Setup"
echo "=========================================="

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJ_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJ_DIR"

# 1. Install system dependencies
echo "[1/6] Installing system dependencies..."
sudo apt-get update -qq
sudo apt-get install -y -qq \
    cmake g++ build-essential \
    libeigen3-dev libopencv-dev \
    libceres-dev libboost-all-dev \
    libsuitesparse-dev \
    2>&1 | tail -3

# 2. Install DBoW3
echo "[2/6] Installing DBoW3..."
if [ ! -f /usr/local/lib/libDBoW3.so ]; then
    git clone --depth=1 https://github.com/rmsalinas/DBow3.git /tmp/DBoW3
    cd /tmp/DBoW3
    mkdir -p build && cd build
    cmake .. -DCMAKE_BUILD_TYPE=Release
    make -j$(nproc)
    sudo make install
    sudo ldconfig
    cd "$PROJ_DIR"
    echo "DBoW3 installed."
else
    echo "DBoW3 already installed."
fi

# 3. Install Python deps for SuperPoint
echo "[3/6] Installing Python dependencies..."
pip install torch numpy opencv-python-headless \
    -q 2>&1 | tail -2

# 4. Download SuperPoint pretrained weights
echo "[4/6] Setting up SuperPoint model..."
mkdir -p "$PROJ_DIR/thirdparty/superpoint"
SP_DIR="$PROJ_DIR/thirdparty/superpoint"
if [ ! -f "$SP_DIR/superpoint_v1.pth" ]; then
    wget -q https://github.com/KinglittleQ/SuperPoint_SLAM/raw/master/Thirdparty/SuperPoint/pretrained/superpoint_v1.pth \
        -O "$SP_DIR/superpoint_v1.pth" 2>/dev/null || \
    echo "WARNING: Could not download pretrained weights. Using demo mode."
fi

# 5. Build ST-SLAM
echo "[5/6] Building ST-SLAM..."
cd "$PROJ_DIR"
rm -rf build
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_PREFIX_PATH=/usr/local
make -j$(nproc) 2>&1 | tail -5

# 6. Download TUM dataset if available
echo "[6/6] Checking TUM sequence..."
if [ ! -d "$PROJ_DIR/../datasets/tum/rgbd_dataset_freiburg1_xyz" ]; then
    echo "No local TUM dataset found."
    echo "To download:"
    echo "  cd $PROJ_DIR/../datasets/tum"
    echo "  wget https://vision.in.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_desk.tgz"
    echo "  tar xzf rgbd_dataset_freiburg1_desk.tgz"
fi

echo ""
echo "=========================================="
echo " ST-SLAM 4.0 Build Complete!"
echo "=========================================="
echo ""
echo "Run evaluation:"
echo "  cd $PROJ_DIR/build"
echo "  LD_LIBRARY_PATH=. ./st_slam_main /path/to/dataset 200"
echo ""
echo "Run SuperPoint preprocessing:"
echo "  python3 $PROJ_DIR/scripts/superpoint_infer.py \\"
echo "    /path/to/dataset/rgb /path/to/output/sp"
echo ""
