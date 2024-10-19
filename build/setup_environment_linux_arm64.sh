#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Function to update package lists and install dependencies
install_dependencies() {
    echo "Updating package lists and installing dependencies..."
    apt-get update
    apt-get install -y clang wget python3-pip python3-venv cmake
    rm -rf /var/lib/apt/lists/*
}

# Function to install Bazel
install_bazel() {
    echo "Installing Bazel 6.1.0..."
    apt-get update
    apt-get install -y wget
    wget -qO bin/bazel-6.1.0-linux-arm64 https://github.com/bazelbuild/bazel/releases/download/6.1.0/bazel-6.1.0-linux-arm64
    chmod +x bin/bazel-6.1.0-linux-arm64
    ln -s /bin/bazel-6.1.0-linux-arm64 /usr/local/bin/bazel
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

# Function to build LightGBM from source for ARM Linux (specific version)
build_lightgbm_from_source() {
    LIGHTGBM_VERSION="v4.2.0"

    echo "Cloning LightGBM repository (version $LIGHTGBM_VERSION)..."
    git clone --recursive https://github.com/microsoft/LightGBM.git /tmp/LightGBM
    cd /tmp/LightGBM

    echo "Checking out version $LIGHTGBM_VERSION..."
    git checkout tags/$LIGHTGBM_VERSION

    echo "Building LightGBM from source on ARM Linux..."
    mkdir build
    cd build
    cmake ..
    make -j 2

    echo "Moving LightGBM shared library to /usr/local/lib..."
    cp ../lib_lightgbm.so /usr/local/lib/lib_lightgbm.so

    echo "LightGBM build (version $LIGHTGBM_VERSION) completed."
}

# Function to download Catboost shared library
download_catboost() {
    echo "Downloading Catboost shared library..."
    wget -O /usr/local/lib/libcatboostmodel.so https://github.com/catboost/catboost/releases/download/v1.2.3/libcatboostr-linux-aarch64-v1.2.3.so
}

# Function to download and extract libtorch
download_libtorch() {
    echo "Downloading and extracting libtorch..."
    wget -O /tmp/libtorch.zip https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.2.0%2Bcpu.zip
    unzip /tmp/libtorch.zip -d /usr/local/lib/
    rm /tmp/libtorch.zip
}

# Function to build TensorFlow from source for ARM Linux (specific version)
build_tensorflow_from_source() {
    TENSORFLOW_VERSION="v2.15.0"  # Change this to the version you need

    echo "Cloning TensorFlow repository (version $TENSORFLOW_VERSION)..."
    git clone https://github.com/tensorflow/tensorflow.git /tmp/tensorflow
    cd /tmp/tensorflow

    echo "Checking out version $TENSORFLOW_VERSION..."
    git checkout tags/$TENSORFLOW_VERSION

    echo "Configuring TensorFlow build..."
    # Run TensorFlow's configure script
    yes "" | ./configure  # This will prompt you to specify various options, like CUDA (answer 'n' for no if not using GPU)

    echo "Building TensorFlow from source..."
    # Bazel build TensorFlow
    bazel build --config opt //tensorflow/tools/lib_package:libtensorflow


    tar -C /usr/local/lib -xzf libtensorflow.tar.gz
    echo "TensorFlow build (version $TENSORFLOW_VERSION) completed."
}



# Function to install protoc
install_protoc() {
    echo "Installing protoc..."
    apt-get update
    apt-get install -y protobuf-compiler
    rm -rf /var/lib/apt/lists/*
}

# Main script execution
install_dependencies
install_bazel
confirm_bazel_installation
print_clang_version
download_catboost
download_libtorch
build_lightgbm_from_source
build_tensorflow_from_source
install_protoc

# add environment variables
# user might need to run these manually or add them to shell profile(.bashrc, .zshrc)
export PROTOC=/usr/bin/protoc
export COMMON_LIBS_PATH=/usr/local/lib
export LIGHTGBM_LIB_DIR=$COMMON_LIBS_PATH
export LIBTORCH=$COMMON_LIBS_PATH/libtorch
export LIBTORCH_INCLUDE=$COMMON_LIBS_PATH/libtorch
export LIBTORCH_LIB=$COMMON_LIBS_PATH/libtorch
export LD_LIBRARY_PATH=$COMMON_LIBS_PATH:$COMMON_LIBS_PATH/libtorch/lib
export LIBRARY_PATH=$LIBRARY_PATH:$COMMON_LIBS_PATH/libtensorflow
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$COMMON_LIBS_PATH/libtensorflow/lib

echo "Script execution completed."