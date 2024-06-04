# JAMS-CLI
This crate is part of wider project called **J.A.M.S - Just Another Model Server**. Please refer [here](https://github.com/gagansingh894/jams-rs).

This crate provides a simple CLI for making predictions for the following models

- Tensorflow
- Torch
- Catboost
- LightGBM

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
cargo install jams-cli
```

## Examples

There are multiple python scripts in [examples folder]() which would allow you to generate different models and their
corresponding sample json input. Below are some examples

### Tensorflow
1. Run tensorflow_penguin_multiclass_classification_model.py
2. This will create two files - a model file and input json file 
3. Run the following command and pass in the path for model file and input file
```
jams-cli tensorflow --model-path=tensorflow_penguin_functional --input-path=tensorflow_input.json

```

### Torch
1. Run torch_penguin_multiclass_classification_model.py
2. This will create two files _ a model file and input json file
3. Run the following command and pass in the path for model file and input file
```
jams-cli torch --model-path=torch_penguin.pt --input-path=torch_input.json
```


### Catboost
1. Run catboost_titanic_binary_classification_model.py
2. This will create two files - a model file and input json file
3. Run the following command and pass in the path for model file and input file
```
jams-cli catboost --model-path=catboost_titanic --input-path=catboost_input.json
```

### LightGBM
1. Run lightgbm_iris_binary_classification_model.py
2. This will create two files - a model file and input json file
3. Run the following command(example) and pass in the path for model file and input file
```
jams-cli light-gbm --model-path=lightgbm_iris.txt --input-path=lightgbm_input.json

```

**DISCLAIMER: jams-cli is currently unstable and may not run properly on your machines. I have
tested the above on apple silicon. Future releases will fix this**
