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

    # this will return a multiclass response for each input record. we can use np.argmax to get the index of the class
    print('TENSORFLOW PREDICTIONS')
    tf_preds = http_client.predict('my_awesome_penguin_model', payload)
    label = np.argmax(np.array(tf_preds.values), axis=1)
    print(f'penguin species label: {label} \n')