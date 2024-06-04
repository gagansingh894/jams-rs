#!/bin/bash
#todo: in progress

# Function to install necessary dependencies
install_dependencies() {
    echo "Installing necessary dependencies..."
    apt-get update && \
    apt-get install -y \
        clang \
        curl \
        build-essential \
        unzip \
        wget \
        && \
    rm -rf /var/lib/apt/lists/*
}

# Function to install Node.js and npm
install_nodejs() {
    echo "Adding NodeSource repository and installing Node.js..."
    curl -fsSL https://deb.nodesource.com/setup_20.x | bash -
    apt-get install -y nodejs
    rm -rf /var/lib/apt/lists/*
}

# Function to install Bazel using Bazelisk
install_bazel() {
    local BAZELISK_VERSION=v1.20.0
    local BAZELISK_URL="https://github.com/bazelbuild/bazelisk/releases/download/${BAZELISK_VERSION}/bazelisk-linux-arm64"
    local BAZEL_VERSION=3.7.2

    echo "Installing Bazel ${BAZEL_VERSION} using Bazelisk..."
    curl -L $BAZELISK_URL -o /usr/local/bin/bazelisk && \
    chmod +x /usr/local/bin/bazelisk && \
    ln -s /usr/local/bin/bazelisk /usr/local/bin/bazel
}

# Function to confirm Bazel installation
confirm_bazel_installation() {
    echo "Confirming Bazel installation..."
    bazelisk version
    bazel version
}

# Function to print Clang version
print_clang_version() {
    echo "Printing Clang version..."
    clang --version
}

# Function to download the LightGBM shared library (.so file) from GitHub releases
download_lightgbm() {
    echo "Downloading LightGBM shared library..."
    wget -O /usr/local/lib/lib_lightgbm.so https://github.com/microsoft/LightGBM/releases/download/v4.2.0/lib_lightgbm.so
}

# Function to download the Catboost shared library (.so file) from GitHub releases
download_catboost() {
    echo "Downloading Catboost shared library..."
    wget -O /usr/local/lib/libcatboostmodel.so https://github.com/catboost/catboost/releases/download/v1.2.3/libcatboostmodel.so
}

# Function to download and extract libtorch
download_libtorch() {
    echo "Downloading and extracting libtorch..."
    wget -O /tmp/libtorch.zip https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.2.0%2Bcpu.zip && \
    unzip /tmp/libtorch.zip -d /usr/local/lib/ && \
    rm /tmp/libtorch.zip
}

# Function to set environment variables
set_environment_variables() {
    echo "Setting environment variables..."
    export LD_LIBRARY_PATH=/usr/local/lib
    export LIGHTGBM_LIB_DIR=/usr/local/lib
    export LIBTORCH=/usr/local/lib/libtorch
    export PATH=/usr/bin/node:$PATH
    export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
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
set_environment_variables

echo "Script execution completed."
