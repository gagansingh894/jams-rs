from typing import Dict

from pydantic import BaseModel


class Spec(BaseModel):
    dtype: str


class ModelSpec(BaseModel):
    framework: str
    spec: Dict[str, Spec]  # key -> feature_name
