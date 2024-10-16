from typing import List

import json


class Prediction:
    """Class representing a 2D array of floats for predictions."""

    def __init__(self, data: str):
        """
        Create a Prediction instance from a JSON-encoded byte string.

        Args:
            data (bytes): JSON byte string representing a 2D array of floats.

        Returns:
            Prediction: A Prediction instance containing the 2D list of floats.

        Raises:
            ValueError: If the input data cannot be decoded or parsed.
        """
        try:
            self._values: List[List[float]] = json.loads(data)
        except (json.JSONDecodeError, TypeError) as e:
            raise ValueError(f"input data cannot be decoded or parsed: {e}")

    @property
    def values(self) -> List[List[float]]:
        return self._values
