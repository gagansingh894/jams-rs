from typing import Dict, List

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
            _values: Dict[str, List[List[float]]] = json.loads(data)
            # Loop over the map to get the value (since we know there is only one key)
            for value in _values.values():
                self._values: List[List[float]] = value
                break  # Stop after the first (and only) iteration
        except (json.JSONDecodeError, TypeError) as e:
            raise ValueError(f"input data cannot be decoded or parsed: {e}")

    @property
    def values(self) -> List[List[float]]:
        return self._values
