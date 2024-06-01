# J.A.M.S 

[![Build](https://github.com/gagansingh894/jams-rs/actions/workflows/build.yml/badge.svg?branch=main)](https://github.com/gagansingh894/jams-rs/actions/workflows/build.yml)


J.A.M.S acronym for Just Another Model Server aims to provide a fast, comprehensive and modular serving solution for tree based and deep learning models written in Rust🦀

It is primarily targeted for software and data professionals for deploying their models in production

## Features
- Modular Design 📦
- Supports PyTorch and Tensorflow Models 🤖
- Support Tree Models - Catboost, LightGBM, XGBoost(in-progress) 🌳
- Fast with HTTP & gRPC API 🚀

The project is divided into following crates

- jams-core
- jams-serve
- jams-cli

`jams-core` provides thin abstraction around common machine learning and deep learning models as well as databases like redis, dynamodb which can be used as real time feature stores. You can think of each component as a LEGO block which can be used to build a system depending on the requirements

`jams-serve` provides an http and gRPC API  for jams-core. The API is highly configurable allowing the user to select which components to use when setting up the model server. Please refer to examples for different types of setup.

`jams-cli` is an easy to use CLI application for server configuration and monitoring.

## Docker Setup
<todo>
