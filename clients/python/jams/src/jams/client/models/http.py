from typing import List
from pydantic import BaseModel


class PredictRequest(BaseModel):
    model_name: str
    input: str


class PredictResponse(BaseModel):
    output: str


class ModelMetadata(BaseModel):
    name: str
    framework: str
    path: str
    last_updated: str


class GetModelsResponse(BaseModel):
    total: int
    models: List[ModelMetadata]


class AddModelsRequest(BaseModel):
    model_name: str


class UpdateModelsRequest(BaseModel):
    model_name: str
