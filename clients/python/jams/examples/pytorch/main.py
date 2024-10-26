# ruff: noqa
# type: ignore
import asyncio
import json

from jamspy.client import http

# a playground service is hosted here with some models
# latency will be high compared to actual production grade setup
URL = "https://jams-http.onrender.com"


async def pytorch_example():
    http_client = http.Client(base_url=URL)

    # health check
    try:
        await http_client.health_check()
    except Exception as e:
        raise f'service is not running: {e}'

    # read request json
    with open('request.json', 'r') as f:
        request = json.load(f)

    # convert to string
    payload = json.dumps(request)

    # this is a regression model so output would be continous for each input record
    print('TORCH PREDICTIONS')
    torch_preds = await http_client.predict('my_awesome_californiahousing_model', payload)
    print(f'prices: {torch_preds.values} \n')


if __name__ == '__main__':
    asyncio.run(pytorch_example())
