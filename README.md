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



**J.A.M.S** acronym for Just Another Model Server aims to provide a fast, comprehensive and modular serving solution for tree based and deep learning models written in Rust 🦀

It is primarily targeted for software and data professionals for deploying their models in production

## Features
- Modular Design 📦
- Supports PyTorch and Tensorflow Models via FFI Bindings 🤖
- Support Tree Models - Catboost, LightGBM, (🚧) XGBoost via FFI Bindings 🌳
- (🚧) HTTP & gRPC API 🚀
- (🚧) CLI ⌨️ 


The project is divided into following crates

- jams-core ![](https://img.shields.io/crates/v/jams-core)
- jams-serve
- jams-cli ![](https://img.shields.io/crates/v/jams-cli)

(🚧)`jams-core` provides thin abstraction around common machine learning and deep learning models as well as databases like redis, dynamodb which can be used as real time feature stores. You can think of each component as a LEGO block which can be used to build a system depending on the requirements

(🚧)`jams-serve` is a http and gRPC API for jams-core. The API is highly configurable allowing the user to select which components to use when setting up the model server.
It provides a CLI for server configuration and starting the server. Please refer to examples for different types of setup.


(🚧)`jams-cli` is an easy-to-use CLI allowing user to make predictions by specifying model and an input string.

## Docker Setup
TODO

