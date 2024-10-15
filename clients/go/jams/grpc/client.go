package grpc

import (
	"context"
	"fmt"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/protobuf/types/known/emptypb"

	v1 "github.com/gagansingh894/jams-rs/clients/go/jams/pkg/pb/jams"
	"github.com/gagansingh894/jams-rs/clients/go/jams/types"
)

//go:generate mockery --name Client --output=../mocks/grpc
type Client interface {
	HealthCheck(ctx context.Context, in *emptypb.Empty, opts ...grpc.CallOption) error
	Predict(ctx context.Context, in *v1.PredictRequest, opts ...grpc.CallOption) (types.Prediction, error)
	AddModel(ctx context.Context, in *v1.AddModelRequest, opts ...grpc.CallOption) error
	UpdateModel(ctx context.Context, in *v1.UpdateModelRequest, opts ...grpc.CallOption) error
	DeleteModel(ctx context.Context, in *v1.DeleteModelRequest, opts ...grpc.CallOption) error
	GetModels(ctx context.Context, in *emptypb.Empty, opts ...grpc.CallOption) (*v1.GetModelsResponse, error)
}

// todo: Add batching
type client struct {
	client v1.ModelServerClient
}

func New(url string) (Client, error) {
	conn, err := grpc.NewClient(url, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		return nil, fmt.Errorf("failed to create client: %w", err)
	}
	defer conn.Close()

	return &client{
		client: v1.NewModelServerClient(conn),
	}, nil
}

func (c *client) HealthCheck(ctx context.Context, _ *emptypb.Empty, opts ...grpc.CallOption) error {
	_, err := c.client.HealthCheck(ctx, &emptypb.Empty{}, opts...)
	if err != nil {
		return fmt.Errorf("failed to check health: %w", err)
	}

	return nil
}

func (c *client) Predict(ctx context.Context, in *v1.PredictRequest, opts ...grpc.CallOption) (types.Prediction, error) {
	response, err := c.client.Predict(ctx, in, opts...)
	if err != nil {
		return nil, err
	}

	// parse response to type.Prediction
	prediction, err := types.NewPrediction([]byte(response.Output))
	if err != nil {
		return nil, fmt.Errorf("failed to parse prediction: %w", err)
	}

	return prediction, nil
}

func (c *client) AddModel(ctx context.Context, in *v1.AddModelRequest, opts ...grpc.CallOption) error {
	_, err := c.client.AddModel(ctx, in, opts...)
	if err != nil {
		return fmt.Errorf("failed to add model: %w", err)
	}

	return nil
}

func (c *client) UpdateModel(ctx context.Context, in *v1.UpdateModelRequest, opts ...grpc.CallOption) error {
	_, err := c.client.UpdateModel(ctx, in, opts...)
	if err != nil {
		return fmt.Errorf("failed to update model: %w", err)
	}

	return nil
}

func (c *client) DeleteModel(ctx context.Context, in *v1.DeleteModelRequest, opts ...grpc.CallOption) error {
	_, err := c.client.DeleteModel(ctx, in, opts...)
	if err != nil {
		return fmt.Errorf("failed to delete model: %w", err)
	}

	return nil
}

func (c *client) GetModels(ctx context.Context, in *emptypb.Empty, opts ...grpc.CallOption) (*v1.GetModelsResponse, error) {
	response, err := c.client.GetModels(ctx, in, opts...)
	if err != nil {
		return nil, fmt.Errorf("failed to get models: %w", err)
	}

	return response, nil
}
