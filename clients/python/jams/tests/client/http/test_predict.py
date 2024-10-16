import json

from src.jams.client import http
from src.jams.client.models import common

from tests.helper import get_http_url


def test_successfully_makes_predict_request() -> None:
    # Arrange
    base_url = get_http_url()
    http_client = http.Client(base_url)

    # Act
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
    resp = http_client.predict(model_name=model_name, model_input=model_input)

    # Assert
    # If the function errors out then we can assume the test has failed
    assert isinstance(resp, common.Prediction)
    assert 2 == len(resp.values)
