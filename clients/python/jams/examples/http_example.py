from jamspy.client import http  # type: ignore

# a playground service is hosted here with some models
# latency will be high compared to actual production grade setup
URL = "https://jams-http.onrender.com"

http_client = http.Client(base_url=URL)

# health check
try:
    http_client.health_check()
except Exception as e:
    raise f'service is not running: {e}'  # type: ignore

# get models
models = http_client.get_models()
print(models.dict())
