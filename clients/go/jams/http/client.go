package http

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"github.com/gagansingh894/jams-rs/clients/go/jams/types"
	"net/http"
	"strings"
)

//go:generate mockery --name Client --output=../mocks/http
type Client interface {
	HealthCheck(ctx context.Context) error
	Predict(ctx context.Context, request *PredictRequest) (types.Prediction, error)
	AddModel(ctx context.Context, request *AddModelRequest) error
	UpdateModel(ctx context.Context, request *UpdateModelRequest) error
	DeleteModel(ctx context.Context, modelName string) error
	GetModels(ctx context.Context) (*GetModelsResponse, error)
}

// todo: Add batching
type client struct {
	baseURL string
	http.Client
}

func New(baseURL string) Client {
	if strings.HasPrefix(baseURL, "http://") || strings.HasPrefix(baseURL, "https://") {
		return &client{baseURL: baseURL}
	}

	return &client{baseURL: fmt.Sprintf("http://%s", baseURL)}
}

func (c *client) HealthCheck(ctx context.Context) error {
	url := fmt.Sprintf("%s/healthcheck", c.baseURL)
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return fmt.Errorf("failed to create HealthCheck request: %w", err)
	}

	res, err := c.Do(req)
	if err != nil {
		return fmt.Errorf("failed to do HealthCheck request: %w", err)
	}
	defer res.Body.Close()

	if res.StatusCode != http.StatusOK {
		return fmt.Errorf("failed to do HealthCheck request: %s", res.Status)
	}

	return nil
}

func (c *client) Predict(ctx context.Context, request *PredictRequest) (types.Prediction, error) {
	b, err := json.Marshal(request)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal Predict request: %w", err)
	}

	url := fmt.Sprintf("%s/api/predict", c.baseURL)
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewBuffer(b))
	if err != nil {
		return nil, fmt.Errorf("failed to create Predict request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	res, err := c.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to do Predict request: %w", err)
	}
	defer res.Body.Close()

	if res.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("failed to do Predict request: %s", res.Status)
	}

	out := PredictResponse{}
	err = json.NewDecoder(res.Body).Decode(&out)
	if err != nil {
		return nil, fmt.Errorf("failed to parse response from Predict request: %w", err)
	}

	prediction, err := types.NewPrediction([]byte(out.Output))
	if err != nil {
		return nil, fmt.Errorf("failed to parse PredictResponse into Prediction: %w", err)
	}

	return prediction, nil
}

func (c *client) AddModel(ctx context.Context, request *AddModelRequest) error {
	b, err := json.Marshal(request)
	if err != nil {
		return fmt.Errorf("failed to marshal AddModel request: %w", err)
	}

	url := fmt.Sprintf("%s/api/models", c.baseURL)
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewBuffer(b))
	if err != nil {
		return fmt.Errorf("failed to create AddModel request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	res, err := c.Do(req)
	if err != nil {
		return fmt.Errorf("failed to do AddModel request: %w", err)
	}
	defer res.Body.Close()

	if res.StatusCode != http.StatusOK {
		return fmt.Errorf("failed to do AddModel request: %s", res.Status)
	}

	return nil
}

func (c *client) UpdateModel(ctx context.Context, request *UpdateModelRequest) error {
	b, err := json.Marshal(request)
	if err != nil {
		return fmt.Errorf("failed to marshal UpdateModel request: %w", err)
	}

	url := fmt.Sprintf("%s/api/models", c.baseURL)
	req, err := http.NewRequestWithContext(ctx, http.MethodPut, url, bytes.NewBuffer(b))
	if err != nil {
		return fmt.Errorf("failed to create UpdateModel request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	res, err := c.Do(req)
	if err != nil {
		return fmt.Errorf("failed to do UpdateModel request: %w", err)
	}
	defer res.Body.Close()

	if res.StatusCode != http.StatusOK {
		return fmt.Errorf("failed to do UpdateModel request: %s", res.Status)
	}

	return nil
}

func (c *client) DeleteModel(ctx context.Context, modelName string) error {
	url := fmt.Sprintf("%s/api/models?model_name=%s", c.baseURL, modelName)
	req, err := http.NewRequestWithContext(ctx, http.MethodDelete, url, nil)
	if err != nil {
		return fmt.Errorf("failed to create DeleteModel request: %w", err)
	}

	res, err := c.Do(req)
	if err != nil {
		return fmt.Errorf("failed to do DeleteModel request: %w", err)
	}
	defer res.Body.Close()

	return nil
}

func (c *client) GetModels(ctx context.Context) (*GetModelsResponse, error) {
	url := fmt.Sprintf("%s/api/models", c.baseURL)
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to create GetModels request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	res, err := c.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to do GetModels request: %w", err)
	}
	defer res.Body.Close()

	if res.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("failed to do GetModels request: %s", res.Status)
	}

	models := &GetModelsResponse{}
	err = json.NewDecoder(res.Body).Decode(models)
	if err != nil {
		return nil, fmt.Errorf("failed to decode GetModels response: %w", err)
	}

	return models, nil
}
