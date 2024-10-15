package grpc

import (
	"context"
	"fmt"
	"os"
	"testing"

	"github.com/stretchr/testify/assert"
	"google.golang.org/protobuf/types/known/emptypb"

	"github.com/gagansingh894/jams-rs/clients/go/jams/pkg/pb/jams"
)

func getURL() string {
	hostname := os.Getenv("JAMS_GRPC_HOSTNAME")
	if hostname == "" {
		hostname = "0.0.0.0"
	}
	return fmt.Sprintf("%s:4000", hostname)
}

func TestHealthCheck(t *testing.T) {
	// Arrange
	ctx := context.Background()
	client, err := New(getURL())
	assert.Nil(t, err)

	// Act
	err = client.HealthCheck(ctx, &emptypb.Empty{})

	// Assert
	assert.NoError(t, err)
}

func TestGetModels(t *testing.T) {
	// Arrange
	ctx := context.Background()
	client, err := New(getURL())
	assert.Nil(t, err)

	// Act
	resp, err := client.GetModels(ctx, &emptypb.Empty{})

	// Assert
	assert.NoError(t, err)
	assert.NotNil(t, resp)
}

func TestDeleteModel(t *testing.T) {
	// Arrange
	ctx := context.Background()
	client, err := New(getURL())
	assert.Nil(t, err)

	// Act
	err = client.DeleteModel(ctx, &jams.DeleteModelRequest{ModelName: "my_awesome_californiahousing_model"})

	// Assert
	assert.NoError(t, err)
}

func TestAddModel(t *testing.T) {
	// Arrange
	ctx := context.Background()
	client, err := New(getURL())
	assert.Nil(t, err)

	// Act
	err = client.DeleteModel(ctx, &jams.DeleteModelRequest{ModelName: "my_awesome_penguin_model"})
	assert.NoError(t, err)
	err = client.AddModel(ctx, &jams.AddModelRequest{ModelName: "tensorflow-my_awesome_penguin_model"})

	// Assert
	assert.NoError(t, err)
}

func TestUpdateModel(t *testing.T) {
	// Arrange
	ctx := context.Background()
	client, err := New(getURL())
	assert.Nil(t, err)

	// Act
	err = client.UpdateModel(ctx, &jams.UpdateModelRequest{ModelName: "titanic_model"})

	// Assert
	assert.NoError(t, err)
}

func TestPredict(t *testing.T) {
	// Arrange
	ctx := context.Background()
	client, err := New(getURL())
	assert.Nil(t, err)

	// Act
	err = client.HealthCheck(ctx, &emptypb.Empty{})

	// Assert
	assert.NoError(t, err)
}
