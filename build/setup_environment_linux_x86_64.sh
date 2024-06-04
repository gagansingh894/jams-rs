#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Function to update package lists and install dependencies
install_dependencies() {
    echo "Updating package lists and installing dependencies..."
    apt-get update
    apt-get install -y clang wget
    rm -rf /var/lib/apt/lists/*
}

# Function to install Bazel
install_bazel() {
    echo "Installing Bazel 3.7.2..."
    apt-get update
    apt-get install -y wget
    wget -qO /tmp/bazel-installer.sh https://github.com/bazelbuild/bazel/releases/download/3.7.2/bazel-3.7.2-installer-linux-x86_64.sh
    chmod +x /tmp/bazel-installer.sh
    /tmp/bazel-installer.sh
    rm /tmp/bazel-installer.sh
    apt-get clean
    rm -rf /var/lib/apt/lists/*
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
    unzip /tmp/libtorch.zip -d /usr/local/lib/
    rm /tmp/libtorch.zip
}

# Function to download and download TensorFlow library
download_tensorflow() {
    # Variables
    FILENAME="libtensorflow-cpu-linux-x86_64-2.15.0.tar.gz"
    DOWNLOAD_URL="https://storage.googleapis.com/tensorflow/libtensorflow/${FILENAME}"
    INSTALL_DIR="/usr/local/lib/libtensorflow"

    echo "Downloading TensorFlow from $DOWNLOAD_URL..."
    wget -q --no-check-certificate "$DOWNLOAD_URL"

    echo "Extracting $FILENAME to $INSTALL_DIR..."
    mkdir -p "$INSTALL_DIR"
    tar -C "$INSTALL_DIR" -xzf "$FILENAME"
    rm "$FILENAME"

    echo "Running ldconfig to update library cache..."
    ldconfig "$INSTALL_DIR"

    echo "TensorFlow installation completed."
}

# Function to install Node.js and npm
install_nodejs() {
    echo "Adding NodeSource repository and installing Node.js..."
    curl -fsSL https://deb.nodesource.com/setup_20.x | bash -
    apt-get install -y nodejs
    rm -rf /var/lib/apt/lists/*
}

# Main script execution
install_dependencies
install_nodejs
install_bazel
confirm_bazel_installation
print_clang_version
download_lightgbm
download_catboost
download_libtorch
download_tensorflow

# add environment variables
# user might need to run these manually or add them to shell profile(.bashrc, .zshrc)
export COMMON_LIBS_PATH=/usr/local/lib
export LIGHTGBM_LIB_DIR=$COMMON_LIBS_PATH
export LIBTORCH=$COMMON_LIBS_PATH/libtorch
export LIBTORCH_INCLUDE=$COMMON_LIBS_PATH/libtorch
export LIBTORCH_LIB=$COMMON_LIBS_PATH/libtorch
export LD_LIBRARY_PATH=$COMMON_LIBS_PATH:$COMMON_LIBS_PATH/libtorch/lib
export LIBRARY_PATH=$LIBRARY_PATH:$COMMON_LIBS_PATH/libtensorflow
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$COMMON_LIBS_PATH/libtensorflow/lib

echo "Script execution completed."