```
    ___           ________           _____ ______            ________
   |\  \         |\   __  \         |\   _ \  _   \         |\   ____\
   \ \  \        \ \  \|\  \        \ \  \\\__\ \  \        \ \  \___|_
 __ \ \  \        \ \   __  \        \ \  \\|__| \  \        \ \_____  \
|\  \\_\  \  ___   \ \  \ \  \  ___   \ \  \    \ \  \  ___   \|____|\  \
\ \________\|\__\   \ \__\ \__\|\__\   \ \__\    \ \__\|\__\    ____\_\  \
 \|________|\|__|    \|__|\|__|\|__|    \|__|     \|__|\|__|   |\_________\
                                                               \|_________|

J.A.M.S - Just Another Model Server
```

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0) 
[![Build](https://github.com/gagansingh894/jams-rs/actions/workflows/build.yml/badge.svg?branch=main)](https://github.com/gagansingh894/jams-rs/actions/workflows/build.yml)
![Codecov](https://img.shields.io/codecov/c/github/gagansingh894/jams-rs)



**J.A.M.S** acronym for Just Another Model Server aims to provide a fast, comprehensive and modular serving solution for tree based and deep learning models written in Rust 🦀

It is primarily targeted for software and data professionals for deploying their models in production

## Features
- Modular Design 📦
- (🚧) Configurable 🛠️
- Supports PyTorch and Tensorflow Models via FFI Bindings 🤖
- Support Tree Models - Catboost, LightGBM, (🚧) XGBoost via FFI Bindings 🌳
- (🚧) Support multiple backends for model stores - local file system, AWS S3, Azure Blob 🗳️
- (🚧) Support Redis and DynamoDB for in memory feature stores 🗂️
- HTTP & gRPC API 🚀
- CLI 💻  

The project is divided into the following crates

- jams-core ![](https://img.shields.io/crates/v/jams-core)
- jams-serve ![](https://img.shields.io/crates/v/jams-serve)
- jams ![](https://img.shields.io/crates/v/jams)

(🚧)`jams-core`
is the core library
which provides thin abstraction around common machine learning and deep learning models as well as databases like redis,
dynamodb which can be used as real time feature stores.
You can think of each component as a LEGO block which can be used to build a system depending on the requirements

(🚧)`jams-serve` is a http and gRPC API library for jams-core.
The API is highly configurable, allowing the user to select which components to use when setting up the model server.
Please refer to examples for different types of setup.

(🚧)`jams` is an easy-to-use CLI allowing user to make predictions by specifying model and an input string.

⚠️ **DISCLAIMER: jams is currently unstable and may not run properly on your machines. I have
tested the above on Apple Silicon and Linux x86_64 machines. Future releases will fix this**

## Setup
**Ensure that Cargo and Rust compiler are installed. Follow instructions [here](https://www.rust-lang.org/tools/install) if not installed**

This project relies on a couple of shared libraries. To easily set up, please follow the steps below


### Mac
1. Install [Homebrew](https://brew.sh/) if not already installed
2. Run the following command to install bazel, lightgbm, pytorch and tensorflow
```
brew install lightgbm pytorch tensorflow
```
3. Download catboost library(.dylib) directly from Github
```
wget -q https://github.com/catboost/catboost/releases/download/v1.2.5/libcatboostmodel-darwin-universal2-1.2.5.dylib -O /usr/local/lib/libcatboostmodel.dylib
```
4. Copy lightgbm to usr/local/lib
```
cp /opt/homebrew/Cellar/lightgbm/4.3.0/lib/lib_lightgbm.dylib /usr/local/lib
```
5. Add the following environment variables
```
export LIBTORCH=/opt/homebrew/Cellar/pytorch/2.2.0_4
export LIGHTGBM_LIB_PATH=/opt/homebrew/Cellar/lightgbm/4.3.0/lib/
export DYLD_LIBRARY_PATH=/usr/local/lib:$DYLD_LIBRARY_PATH
```

**Remember to check version numbers in the path as homebrew downloads the latest stable version.**

**Use brew info to get the exact path which you can use to set the environment variables**

6. Run the following command to install **jams**
```
cargo install jams
```

### Linux
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

3. Run the following command to install **J.A.M.S**
```
cargo install jams
```
---

## API Endpoints
Once **J.A.M.S** is up and running, these endpoints will help you interact with the server. 

Please refer to [OpenAPI Spec](https://github.com/gagansingh894/jams-rs/blob/main/openapi.yml) for details.

`/healthcheck`: Endpoint for health checks

`/api/predict`: Endpoint for making predictions

`/api/models`: Endpoint for managing models

Alternatively, you can also refer to the [proto definition](https://github.com/gagansingh894/jams-rs/blob/main/jams-serve/proto/api/v1/jams.proto). It provides the following **RPCs**

- `HealthCheck`
- `Predict`
- `GetModels`
- `AddModel`
- `UpdateModel`
- `DeleteModel`

---

## Usage
The CLI provides the following commands

```
- jams start
- jams predict
```

### start
Use this command to start either the HTTP/gRPC model server on `0.0.0.0:3000`/`0.0.0.0:4000` with separate rayon threadpool for computing predictions

The server expects a model directory containing the models. This can be either passed using the
```--model-dir``` flag

To start HTTP server
```
jams start http --model-dir path/to/model_dir
```

To start gRPC server
```
jams start grpc --model-dir path/to/model_dir
```

Alternatively, you can set the **MODEL_STORE_DIR** env variable pointing to the model directory
and run `jams start http` or `jams start grpc`
```
export MODEL_STORE_DIR=path/to/model_dir
```

**_If no path is provided for a model directory, the server will start but with a warning. 
The models can still be added via API endpoints._**

By default, the server runs on port `3000` and `2` workers in the rayon threadpool.You can override using
the `--port` and `--num-workers` flags respectively. The log level can also be changed to
`DEBUG` level using `--use-debug-level=true`.

Below is an example of local model dir.
Notice the model naming convention
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

### predict
Use this command for making predictions via CLI for making predictions for the following models

- Tensorflow
- Torch
- Catboost
- LightGBM

Refer below for some examples of the **predict** command.

There are multiple python scripts in [examples folder](https://github.com/gagansingh894/jams-rs/tree/main/jams-cli/examples) which would allow you to generate different models and their
corresponding sample json input. Below are some examples

#### Tensorflow
1. Run tensorflow_penguin_multiclass_classification_model.py
2. This will create two files- a model file and input json file
3. Run the following command and pass in the path for model file and input file
```
jams predict tensorflow --model-path=tensorflow_penguin_functional --input-path=tensorflow_input.json

```

#### Torch
1. Run torch_penguin_multiclass_classification_model.py
2. This will create two files- a model file and input json file
3. Run the following command and pass in the path for model file and input file
```
jams predict torch --model-path=torch_penguin.pt --input-path=torch_input.json
```


#### Catboost
1. Run catboost_titanic_binary_classification_model.py
2. This will create two files- a model file and input json file
3. Run the following command and pass in the path for model file and input file
```
jams predict catboost --model-path=catboost_titanic --input-path=catboost_input.json
```

#### LightGBM
1. Run lightgbm_iris_binary_classification_model.py
2. This will create two files- a model file and input json file
3. Run the following command(example) and pass in the path for model file and input file
```
jams predict lightgbm --model-path=lightgbm_iris.txt --input-path=lightgbm_input.json
```
---

## Docker
Please follow the following commands to start the server inside docker

1. git clone https://github.com/gagansingh894/jams-rs.git
2. `cd jams`
3. `docker build -t <your-tag> .`
4. `docker run --rm -p 3000:3000 <image_name>
   `