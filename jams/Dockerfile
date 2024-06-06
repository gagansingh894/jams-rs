FROM --platform=linux/amd64 ubuntu:22.04
LABEL authors="gagandeepsingh"

# Install necessary dependencies
RUN apt-get update && \
    apt-get install -y \
        clang \
        curl \
        build-essential \
        pkg-config \
        libssl-dev \
        unzip \
        wget \
        && \
    rm -rf /var/lib/apt/lists/*

# copy build scriot to install all the build dependencies
COPY setup_environment_linux_x86_64.sh .

# make file executable
RUN chmod +x setup_environment_linux_x86_64.sh

# install build dependencies
RUN ./setup_environment_linux_x86_64.sh

# set environment variable
ENV COMMON_LIBS_PATH=/usr/local/lib
ENV LIGHTGBM_LIB_DIR=$COMMON_LIBS_PATH
ENV LIBTORCH=$COMMON_LIBS_PATH/libtorch
ENV LIBTORCH_INCLUDE=$COMMON_LIBS_PATH/libtorch
ENV LIBTORCH_LIB=$COMMON_LIBS_PATH/libtorch
ENV LD_LIBRARY_PATH=$COMMON_LIBS_PATH:$COMMON_LIBS_PATH/libtorch/lib
ENV LIBRARY_PATH=$LIBRARY_PATH:$COMMON_LIBS_PATH/libtensorflow
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$COMMON_LIBS_PATH/libtensorflow/lib

# Install Rust using rustup
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y

# Set environment variables for Rust
ENV PATH="/root/.cargo/bin:${PATH}"

# Verify installation by printing the version of rustc
RUN rustc --version

# install jams-serve
RUN cargo install jams-serve

# entry point
ENTRYPOINT ["jams-serve", "start"]

# the user would need to expose the port, mount volumne and provide the model-dir flag in order for the server to start