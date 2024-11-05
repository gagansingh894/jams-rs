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
import asyncio
import json
from jamspy.client.http import Client

# Initialize the async client
client = Client('0.0.0.0:3000')

async def main():
    # Perform a health check
    await client.health_check()

    # Define model and input data for prediction
    model_name = "titanic_model"
    model_input = json.dumps({
        "pclass": ["1", "3"],
        "sex": ["male", "female"],
        "age": [22.0, 23.79929292929293],
        "sibsp": ["0", "1"],
        "parch": ["0", "0"],
        "fare": [151.55, 14.4542],
        "embarked": ["S", "C"],
        "class": ["First", "Third"],
        "who": ["man", "woman"],
        "adult_male": ["True", "False"],
        "deck": ["Unknown", "Unknown"],
        "embark_town": ["Southampton", "Cherbourg"],
        "alone": ["True", "False"],
    })

    # Perform a prediction and await the response
    prediction = await client.predict(model_name=model_name, model_input=model_input)
    print(prediction.values)  # Use predictions

    # Add a model (model name format: <MODEL FRAMEWORK>-<MODEL_NAME>)
    await client.add_model(model_name='tensorflow-my_awesome_penguin_model')

    # Update the model
    await client.update_model(model_name='my_awesome_penguin_model')

    # Delete the model
    await client.delete_model(model_name='my_awesome_penguin_model')

    # Fetch the list of models and await the response
    models = await client.get_models()
    print(models)

# Run the main function as an async entry point
asyncio.run(main())

```

### gRPC

```
import json
import asyncio
from jamspy.client.grpc import Client

async def main():
    # Initialize the asynchronous client
    client = Client('0.0.0.0:4000')

    # Health check
    await client.health_check()

    # Define model and input data for prediction
    model_name = "titanic_model"
    model_input = json.dumps({
        "pclass": ["1", "3"],
        "sex": ["male", "female"],
        "age": [22.0, 23.79929292929293],
        "sibsp": ["0", "1"],
        "parch": ["0", "0"],
        "fare": [151.55, 14.4542],
        "embarked": ["S", "C"],
        "class": ["First", "Third"],
        "who": ["man", "woman"],
        "adult_male": ["True", "False"],
        "deck": ["Unknown", "Unknown"],
        "embark_town": ["Southampton", "Cherbourg"],
        "alone": ["True", "False"],
    })

    # Predict and await the response
    prediction = await client.predict(model_name=model_name, model_input=model_input)
    print(prediction.values)  # Use predictions

    # Add model
    await client.add_model(model_name='tensorflow-my_awesome_penguin_model') # <MODEL FRAMEWORK>-<MODEL_NAME>

    # Update model
    await client.update_model(model_name='my_awesome_penguin_model')

    # Delete model
    await client.delete_model(model_name='my_awesome_penguin_model')

    # Get models and await the response
    models = await client.get_models()
    print(models)

# Run the main function asynchronously
asyncio.run(main())
```