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
## Usage
The CLI expects a model directory containing the models. This can be either passed to
the CLI via the command

```
jams-serve start --model-dir path/to/model_dir
```

Alternatively, you can set the **MODEL_STORE_DIR** env variable pointing to the model directory
and run `jams-serve start`
```
export MODEL_STORE_DIR=path/to/model_dir
```

By default the server runs on port `3000`. You can override using the `--port` flag.

See below an example of local model dir. Notice the model naming convention
**<model_framework>-model_name**

```
.
└── local_model_store
    ├── catboost-my_awesome_binary_model
    ├── catboost-my_awesome_multiclass_model
    ├── catboost-my_awesome_regressor_model
    ├── catboost-titanic_model
    ├── lightgbm-my_awesome_binary_model_2.txt
    ├── lightgbm-my_awesome_reg_model.txt
    ├── lightgbm-my_awesome_xen_binary_model.txt
    ├── lightgbm-my_awesome_xen_prob_model.txt
    ├── pytorch-my_awesome_californiahousing_model.pt
    ├── tensorflow-my_awesome_autompg_model
    │   ├── assets
    │   ├── fingerprint.pb
    │   ├── keras_metadata.pb
    │   ├── saved_model.pb
    │   └── variables
    │       ├── variables.data-00000-of-00001
    │       └── variables.index
    ├── tensorflow-my_awesome_penguin_model
    │   ├── assets
    │   ├── fingerprint.pb
    │   ├── keras_metadata.pb
    │   ├── saved_model.pb
    │   └── variables
    │       ├── variables.data-00000-of-00001
    │       └── variables.index
    ├── tensorflow-my_awesome_sequential_model
    │   ├── assets
    │   ├── fingerprint.pb
    │   ├── keras_metadata.pb
    │   ├── saved_model.pb
    │   └── variables
    │       ├── variables.data-00000-of-00001
    │       └── variables.index
    └── torch-my_awesome_penguin_model.pt

```


## Docker
Please follow the following commands

1. git clone https://github.com/gagansingh894/jams-rs.git
2. `cd jams-serve`
3. `docker build -t <your-tag> .`
4. `docker run --rm -p 3000:3000 -v <host_directory>:<container_directory> -e MODEL_STORE_DIR=your-model-dir <image_name>
   `

