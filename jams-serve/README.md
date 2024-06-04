# JAMS-SERVE

This crate is part of wider project called **J.A.M.S - Just Another Model Server**. Please refer [here](https://github.com/gagansingh894/jams-rs).

`jams-serve` is a http and gRPC API for jams-core. The API is highly configurable allowing the user to select which components to use when setting up the model server.
It provides a CLI for server configuration and starting the server. Please refer to examples for different types of setup.

**Currently only http is supported. This crate is in early stages and is highly unstable**

## Setup
This project relies on couple of shared libraries. In order to easily setup please follow the steps below

1. Use the bash script [here](https://github.com/gagansingh894/jams-rs/blob/main/build) based on your system architecture

2. Run the following commands or add them to shell profile
```
# add environment variables
export COMMON_LIBS_PATH=/usr/local/lib
export LIGHTGBM_LIB_DIR=$COMMON_LIBS_PATH
export LIBTORCH=$COMMON_LIBS_PATH/libtorch
export LIBTORCH_INCLUDE=$COMMON_LIBS_PATH/libtorch
export LIBTORCH_LIB=$COMMON_LIBS_PATH/libtorch
export LD_LIBRARY_PATH=$COMMON_LIBS_PATH:$COMMON_LIBS_PATH/libtorch/lib
export LIBRARY_PATH=$LIBRARY_PATH:$COMMON_LIBS_PATH/libtensorflow
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$COMMON_LIBS_PATH/libtensorflow/lib
```

3. Ensure that cargo and rust compiler are installed. Follow instructions [here](https://www.rust-lang.org/tools/install) if not installed
4. Run the following command to install **jams-cli**
```
cargo install jams-serve
```

## Docker
TODO: A more easier approach would be to have a docker image.
