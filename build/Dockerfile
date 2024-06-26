FROM --platform=linux/amd64 rust:1.78 as builder
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

# Clone the repo
RUN git clone https://github.com/gagansingh894/jams-rs.git
WORKDIR /jams-rs

# Install build dependencies
RUN build/setup_environment_linux_x86_64.sh

# Set environment variable
ENV COMMON_LIBS_PATH=/usr/local/lib
ENV LIGHTGBM_LIB_DIR=$COMMON_LIBS_PATH
ENV LIBTORCH=$COMMON_LIBS_PATH/libtorch
ENV LIBTORCH_INCLUDE=$COMMON_LIBS_PATH/libtorch
ENV LIBTORCH_LIB=$COMMON_LIBS_PATH/libtorch
ENV LD_LIBRARY_PATH=$COMMON_LIBS_PATH:$COMMON_LIBS_PATH/libtorch/lib
ENV LIBRARY_PATH=$LIBRARY_PATH:$COMMON_LIBS_PATH/libtensorflow
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$COMMON_LIBS_PATH/libtensorflow/lib

# Build the binaries in release mode
RUN cargo build --release

FROM --platform=linux/amd64 ubuntu:22.04 AS runtime
# Install OpenSSL - it is dynamically linked by some of our dependencies
# Install ca-certificates - it is needed to verify TLS certificates
# when establishing HTTPS connections
RUN apt-get update -y \
    && apt-get install -y --no-install-recommends openssl ca-certificates libgomp1 \
# Clean up
    && apt-get autoremove -y \
    && apt-get clean -y \
    && rm -rf /var/lib/apt/lists/*

# Copy the shared libraries
COPY --from=builder /usr/local/lib/lib_lightgbm.so usr/local/lib
COPY --from=builder /usr/local/lib/libcatboostmodel.so usr/local/lib
COPY --from=builder /usr/local/lib/libtensorflow usr/local/lib/libtensorflow
COPY --from=builder /usr/local/lib/libtorch usr/local/lib/libtorch

# Set environment variable - required for compilation
ENV COMMON_LIBS_PATH=/usr/local/lib
ENV LIGHTGBM_LIB_DIR=$COMMON_LIBS_PATH
ENV LIBTORCH=$COMMON_LIBS_PATH/libtorch
ENV LIBTORCH_INCLUDE=$COMMON_LIBS_PATH/libtorch
ENV LIBTORCH_LIB=$COMMON_LIBS_PATH/libtorch
ENV LD_LIBRARY_PATH=$COMMON_LIBS_PATH:$COMMON_LIBS_PATH/libtorch/lib
ENV LIBRARY_PATH=$LIBRARY_PATH:$COMMON_LIBS_PATH/libtensorflow
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$COMMON_LIBS_PATH/libtensorflow/lib

# Copy the binary
COPY --from=builder /jams-rs/target/release/jams jams

# Expose the port if the application listens on a specific port
# 3000 for HTTP, 4000 for gRPC
EXPOSE 3000
EXPOSE 4000

# Entrypoint
# The user would need to expose the port
ENTRYPOINT ["./jams"]
CMD ["start", "http"]