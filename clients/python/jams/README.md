# J.A.M.S Python Client


A HTTP & gRPC client for `J.A.M.S - Just Another Model Server`

## Installation
```
pip install jamspy
```

## Usage

Start `J.A.M.S` by following the instructions [here](https://github.com/gagansingh894/jams-rs?tab=readme-ov-file#docker-setup)

### HTTP

```
from jamspy.client.http import Client

# init client
client = Client('0.0.0.0:3000')

# healthcheck
client.health_check()

# predict   
model_name = "titanic_model"
model_input = json.dumps(
    {
        "pclass": ["1", "3"],
        "sex": ["male", "female"],
        "age": [22.0, 23.79929292929293],
        "sibsp": [
            "0",
            "1",
        ],
        "parch": ["0", "0"],
        "fare": [151.55, 14.4542],
        "embarked": ["S", "C"],
        "class": ["First", "Third"],
        "who": ["man", "woman"],
        "adult_male": ["True", "False"],
        "deck": ["Unknown", "Unknown"],
        "embark_town": ["Southampton", "Cherbourg"],
        "alone": ["True", "False"],
    }
)
prediction = client..predict(model_name=model_name, model_input=model_input)
prediction.values # use predictions


# add model
client.add_model(model_name='tensorflow-my_awesome_penguin_model') # <MODEL FRAMEWORK>-<MODEL_NAME>

# update model
client.update_model(model_name='my_awesome_penguin_model')

# delete model
client.delete_model(model_name='my_awesome_penguin_model')

# get models
models = client.get_models()
print(models)

```

### gRPC

```
from jamspy.client.grpc import Client

# init client
client = Client('0.0.0.0:4000')

# healthcheck
client.health_check()

# predict   
model_name = "titanic_model"
model_input = json.dumps(
    {
        "pclass": ["1", "3"],
        "sex": ["male", "female"],
        "age": [22.0, 23.79929292929293],
        "sibsp": [
            "0",
            "1",
        ],
        "parch": ["0", "0"],
        "fare": [151.55, 14.4542],
        "embarked": ["S", "C"],
        "class": ["First", "Third"],
        "who": ["man", "woman"],
        "adult_male": ["True", "False"],
        "deck": ["Unknown", "Unknown"],
        "embark_town": ["Southampton", "Cherbourg"],
        "alone": ["True", "False"],
    }
)
prediction = client..predict(model_name=model_name, model_input=model_input)
prediction.values # use predictions


# add model
client.add_model(model_name='tensorflow-my_awesome_penguin_model') # <MODEL FRAMEWORK>-<MODEL_NAME>

# update model
client.update_model(model_name='my_awesome_penguin_model')

# delete model
client.delete_model(model_name='my_awesome_penguin_model')

# get models
models = client.get_models()
print(models)
```