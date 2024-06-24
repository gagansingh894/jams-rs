package http

type PredictRequest struct {
	ModelName string `json:"model_name"`
	Input     string `json:"input"`
}

type PredictResponse struct {
	Output string `json:"output"`
}

type AddModelRequest struct {
	ModelName string `json:"model_name"`
}

type UpdateModelRequest struct {
	ModelName string `json:"model_name"`
}

type GetModelsResponse struct {
	Total  int32            `json:"total"`
	Models []*ModelMetadata `json:"models"`
}

type ModelMetadata struct {
	Name        string `json:"name"`
	Framework   string `json:"framework"`
	Path        string `json:"path"`
	LastUpdated string `json:"last_updated"`
}
