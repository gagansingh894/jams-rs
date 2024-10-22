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
- Config based deployment 🛠️
- Supports PyTorch* and Tensorflow Models via FFI Bindings 🤖
- Supports Tree Models - Catboost, LightGBM, (🚧) XGBoost via FFI Bindings 🌳
- Supports multiple backends for model stores - local file system, AWS S3, Azure Blob 🗳️
- Supports model store polling ⌛
- HTTP & gRPC API with ready to use clients in Python, Go, Rust, (🚧) TypeScript and (🚧) Java 🚀
- CLI 💻  

The project is divided into the following crates

- jams-core ![](https://img.shields.io/crates/v/jams-core)
- jams-serve ![](https://img.shields.io/crates/v/jams-serve)
- jams-proto ![](https://img.shields.io/crates/v/jams-proto)
- jams ![](https://img.shields.io/crates/v/jams)

`jams-core`
is the core library
which provides thin abstraction around common machine learning and deep learning models and model stores like AWS S3, 
Azure Blob Storage, Local Filesystem.
You can think of each component as a LEGO block which can be used to build a system depending on the requirements

`jams-proto` is provides the gRPC contract for jams-serve.

`jams-serve` is a http and gRPC API library for jams-core.
The API is highly configurable, allowing the user to select which components to use when setting up the model server.
Please refer to examples for different types of setup.

`jams` jams is an easy-to-use CLI application for interaction with J.A.M.S - Just Another Model Server.


(🚧) **J.A.M.S** also provides HTTP & gRPC client implementations in multiple languages. [See here](https://github.com/gagansingh894/jams-rs/tree/main/clients)

⚠️ **DISCLAIMER: jams is reaching stable version but may not run properly on ARM chips. Future releases will fix this.
For now use docker image or Linux x86_64 architecture. Only Pytorch 2.2.0 is supported for now due to dependencies on FFI bindings**



---

## Docker Setup
J.A.M.S is hosted on [DockerHub](https://hub.docker.com/r/gagansingh894/jams).

`docker pull gagansingh894/jams`

The easiest way to start J.A.M.S is by providing a config TOML file

##### Config File
```
[config]
protocol = "http"                               # Specifies the protocol to be used by the server.
                                                # Allowed values: "http", "grpc"

port = 3000                                     # Defines the port number on which the server will listen.
                                                # This should be an integer between 1 and 65535.
                                                # Example: 3000 for HTTP, 443 for HTTPS

model_store = "local"                           # Indicates the type of model store being used.
                                                # Allowed values:
                                                # - "local": Use local storage.
                                                # - "aws": Use AWS S3 for model storage.
                                                # - "azure": Use Azure Blob Storage.

model_dir = "<absolute path>"                   # Specifies the directory path where models are stored locally.
                                                # If `model_store` is set to "local", this directory is used
                                                # to store or load models.

azure_storage_container_name = "jamsmodelstore" # Specifies the name of the Azure Blob Storage container
                                                # used for storing models when `model_store` is set to "azure".
                                                # This should be a valid container name in Azure.

s3_bucket_name = "jamsmodelstore"               # Indicates the name of the S3 bucket used for storing models
                                                # when `model_store` is set to "aws".
                                                # This should be a valid bucket name in AWS S3.

poll_interval = 600                             # Defines the time interval (in seconds) for polling the model store
                                                # to check for updates.
                                                # Example: 600 means the application will poll every 10 minutes.

num_workers = 4                                 # Sets the number of Rayon threadpool worker threads
                                                # Example: 4 threads
```

Then Run

```
docker run --rm -v /your/path/to/model_store:/model_store -p 3000:3000 gagansingh894/jams start -f config.toml
```



There are other ways to start J.A.M.S.
Please follow the following commands to start the server inside docker.

If you want to disable polling, then do not pass `--poll-interval`

To run HTTP server, use
```
docker run --rm -v /your/path/to/model_store:/model_store -p 3000:3000 gagansingh894/jams start http --poll-interval 3600
```

To run gRPC server, use
```
docker run --rm -v /your/path/to/model_store:/model_store -p 4000:4000 gagansingh894/jams start grpc --poll-interval 3600
```

### To run with a S3 backend
- Create a S3 bucket with some models in it. Please refer to the structure of model store [here](https://github.com/gagansingh894/jams-rs?tab=readme-ov-file#model-store).
- Set the environment variables - `AWS_REGION`, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`. Alternatively if you have multiple AWS profiles then just set the `AWS_PROFILE-<profile_name>`
  You also need to set the bucket name. This can either be set via `S3_BUCKET_NAME` env variable or passed via `--s3-bucket-name` flag
- Run the command to start HTTP server with S3 model store. It assumes that bucket name is already set via `S3_BUCKET_NAME`

```
docker run --rm -p 3000:3000 gagansingh894/jams start http --with-s3-model-store=true --poll-interval 3600
```

- For gRPC server, use
```
docker run --rm -p 4000:4000 gagansingh894/jams start grpc --with-s3-model-store=true --poll-interval 3600
```

- If you want to pass bucket name, use
```
docker run --rm -p 3000:3000 gagansingh894/jams start http --with-s3-model-store=true --s3-bucket-name=<bucket_name>
```

### To run with a Azure Blob Storage backend
- Create a Azure Storage container with some models in it. Please refer to the structure of model store [here](https://github.com/gagansingh894/jams-rs?tab=readme-ov-file#model-store).
- Set the environment variables - `STORAGE_ACCOUNT`, `STORAGE_ACCESS_KEY`. You also need to set the azure container name. This can either be set via `AZURE_STORAGE_CONTAINER_NAME` env variable or passed via `--azure-storage-container-name` flag
- Run the command to start HTTP server with Azure model store. It assumes that container name is already set via `AZURE_STORAGE_CONTAINER_NAME`

```
docker run --rm -p 3000:3000 gagansingh894/jams start http --with-azure-model-store=true --poll-interval 3600
```

- For gRPC server, use
```
docker run --rm -p 4000:4000 gagansingh894/jams start grpc --with-azure-model-store=true --poll-interval 3600
```

- If you want to pass container name, use
```
docker run --rm -p 3000:3000 gagansingh894/jams start http --with-azure-model-store=true --azure-storage-container-name=<container_name> --poll-interval 3600
```

Please refer to [OpenAPI Spec](https://github.com/gagansingh894/jams-rs/blob/main/openapi.yml) for API endpoints. 
Alternatively, you can also refer to the [proto definition](https://github.com/gagansingh894/jams-rs/blob/main/internal/jams-proto/proto/api/v1/jams.proto).

---

## Local Setup
**Ensure that Cargo and Rust compiler are installed. Follow instructions [here](https://www.rust-lang.org/tools/install) if not installed**

This project relies on a couple of shared libraries. To easily set up, please follow the steps below


### Mac - Apple Silicon
1. Install [Homebrew](https://brew.sh/) if not already installed
2. Install [Rust](https://www.rust-lang.org/tools/install) if not already installed. **The MSRV is 1.81.**
3. Run the following command to install bazel, lightgbm, pytorch and tensorflow
```
brew install bazel lightgbm tensorflow
```
3. Download **Pytorch 2.2.0** from Pytorch website and set it to path
```
sudo sh -c '
    echo "Downloading libtorch for macOS (ARM64)...";
    wget https://download.pytorch.org/libtorch/cpu/libtorch-macos-arm64-2.2.0.zip -O /usr/local/lib/libtorch-macos-arm64-2.2.0.zip;

    echo "Unzipping libtorch into /usr/local/lib/libtorch2_2_0...";
    mkdir -p /usr/local/lib/libtorch2_2_0;
    unzip /usr/local/lib/libtorch-macos-arm64-2.2.0.zip -d /usr/local/lib/libtorch2_2_0;

    echo "Cleaning up by deleting the zip file...";
    rm /usr/local/lib/libtorch-macos-arm64-2.2.0.zip;

    echo "Setting up environment variables...";
    LIBTORCH_PATH="/usr/local/lib/libtorch2_2_0/libtorch";
    PROFILE_FILE="$HOME/.bash_profile";  # Change to .zshrc if using zsh
    if [ -n "$ZSH_VERSION" ]; then
        PROFILE_FILE="$HOME/.zshrc";
    fi

    if ! grep -q "export LIBTORCH=" "$PROFILE_FILE"; then
        echo "export LIBTORCH=$LIBTORCH_PATH" >> "$PROFILE_FILE";
        echo "LIBTORCH environment variable added to $PROFILE_FILE. Please run 'source $PROFILE_FILE' or restart your terminal.";
    else
        echo "LIBTORCH environment variable already set in $PROFILE_FILE.";
    fi

    echo "Libtorch installed in /usr/local/lib/libtorch2_2_0 and zip file deleted.";
'
```
4. Download catboost library(.dylib) directly from Github
```
sudo wget -q https://github.com/catboost/catboost/releases/download/v1.2.5/libcatboostmodel-darwin-universal2-1.2.5.dylib -O /usr/local/lib/libcatboostmodel.dylib
```
5. Copy lightGBM library(.dylib) to usr/local/lib
```
sudo cp /opt/homebrew/Cellar/lightgbm/$(brew list --versions lightgbm | awk '{print $2}')/lib/lib_lightgbm.dylib /usr/local/lib/
```
6. Add the following environment variables
```
export LIBTORCH=/usr/local/lib/libtorch2_2_0/libtorch
```
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

Alternatively, you can also refer to the [proto definition](https://github.com/gagansingh894/jams-rs/blob/main/internal/jams-proto/proto/api/v1/jams.proto). It provides the following **RPCs**

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

By default, the server runs on port `3000` and `2` workers in the rayon threadpool.You can override using
the `--port` and `--num-workers` flags respectively. The log level can also be changed to
`DEBUG` level using `--use-debug-level=true`.

#### Model Store
Below is the expected structure of model stores. 

- Notice the model naming convention
**<model_framework>-model_name.tar.gz**.
- The server unpacks and loads the model files.
- The server will warn about the unsupported formats and continue to load other models

```
└── model_store
    ├── catboost-my_awesome_binary_model.tar.gz
    ├── catboost-my_awesome_multiclass_model.tar.gz
    ├── catboost-my_awesome_regressor_model.tar.gz
    ├── catboost-titanic_model.tar.gz
    ├── lightgbm-my_awesome_binary_model_2.tar.gz
    ├── lightgbm-my_awesome_reg_model.tar.gz
    ├── lightgbm-my_awesome_xen_binary_model.tar.gz
    ├── lightgbm-my_awesome_xen_prob_model.tar.gz
    ├── pytorch-my_awesome_californiahousing_model.tar.gz
    ├── tensorflow-my_awesome_autompg_model.tar.gz
    ├── tensorflow-my_awesome_penguin_model.tar.gz
    ├── tensorflow-my_awesome_sequential_model.tar.gz
    └── torch-my_awesome_penguin_model.tar.gz

```

### predict
Use this command for making predictions via CLI for making predictions for the following models

- Tensorflow
- Torch
- Catboost
- LightGBM

This command does not expect the model format to be `.tar.gz`.

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