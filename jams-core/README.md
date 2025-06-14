# JAMS-CORE
This library crate is part of a wider project called **J.A.M.S - Just Another Model Server**. Please refer [here](https://github.com/gagansingh894/jams-rs).

**The MSRV is 1.82.**

## Features

- Async
- Multiple Model Frameworks Supported
    - Tensorflow
    - Torch
    - Catboost
    - LightGBM
- Multiple Model Store Backends Supported
    - Local File System
    - AWS S3
    - Azure Blob Storage
    - MinIO
- Model Store Polling

### The following features are in progress 🚧
- Support XGBoost framework
- ModelSpec artefacts - Single source of information about models. This will assist in input validations
---


## Overview

Below diagram provides a high level overview of the crate

![Alt text](https://github.com/gagansingh894/jams-rs/blob/main/jams-core/overview.png?raw=true)