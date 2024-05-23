#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Function to update package lists and install dependencies
install_dependencies() {
    echo "Updating package lists and installing dependencies..."
    sudo apt-get update
    sudo apt-get install -y clang wget
    sudo rm -rf /var/lib/apt/lists/*
}

# Function to install Bazel
install_bazel() {
    echo "Installing Bazel 3.7.2..."
    sudo apt-get update
    sudo apt-get install -y wget
    wget -qO /tmp/bazel-installer.sh https://github.com/bazelbuild/bazel/releases/download/3.7.2/bazel-3.7.2-installer-linux-x86_64.sh
    chmod +x /tmp/bazel-installer.sh
    sudo /tmp/bazel-installer.sh
    rm /tmp/bazel-installer.sh
    sudo apt-get clean
    sudo rm -rf /var/lib/apt/lists/*
}

# Function to confirm Bazel installation
confirm_bazel_installation() {
    echo "Confirming Bazel installation..."
    bazel --version
}

# Function to print Clang version
print_clang_version() {
    echo "Printing Clang version..."
    clang --version
}

# Function to download LightGBM shared library
download_lightgbm() {
    echo "Downloading LightGBM shared library..."
    wget -O /usr/local/lib/lib_lightgbm.so https://github.com/microsoft/LightGBM/releases/download/v4.2.0/lib_lightgbm.so
}

# Function to download Catboost shared library
download_catboost() {
    echo "Downloading Catboost shared library..."
    wget -O /usr/local/lib/libcatboostmodel.so https://github.com/catboost/catboost/releases/download/v1.2.3/libcatboostmodel.so
}

# Function to download and extract libtorch
download_libtorch() {
    echo "Downloading and extracting libtorch..."
    wget -O /tmp/libtorch.zip https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.2.0%2Bcpu.zip
    sudo unzip /tmp/libtorch.zip -d /usr/local
    rm /tmp/libtorch.zip
}

# Main script execution
install_dependencies
install_bazel
confirm_bazel_installation
print_clang_version
download_lightgbm
download_catboost
download_libtorch

# Set environment variables
export LD_LIBRARY_PATH=/usr/local/lib
export LIGHTGBM_LIB_DIR=/usr/local/lib
export LIBTORCH=/usr/local/libtorch

echo "Script execution completed."
