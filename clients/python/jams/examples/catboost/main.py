# ruff: noqa
# type: ignore

import json

import numpy as np

from jamspy.client import http

# a playground service is hosted here with some models
# latency will be high compared to actual production grade setup
URL = "https://jams-http.onrender.com"

if __name__ == '__main__':

    http_client = http.Client(base_url=URL)

    # health check
    try:
        http_client.health_check()
    except Exception as e:
        raise f'service is not running: {e}'

    # read request json
    with open('request.json', 'r') as f:
        request = json.load(f)

    # convert to string
    payload = json.dumps(request)

    # this is a binary classifier model and will return logits of each input record
    print('CATBOOST PREDICTIONS')
    catboost_preds = http_client.predict('titanic_model', payload)
    print(f'logits: {catboost_preds.values}')
    # calculate probabilities using sigmoid function
    sigmoid = lambda x: 1 / (1 + np.exp(-x))
    probabilities = sigmoid(np.array(catboost_preds.values))
    print(f'probabilities: {probabilities}')
    # assign class labels based on threshold (0.5)
    class_predictions = (probabilities >= 0.5).astype(int)
    print(f'class predictions: {class_predictions} \n')

