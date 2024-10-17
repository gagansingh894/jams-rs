package http

import (
	"context"
	"fmt"
	"os"
	"testing"

	"github.com/stretchr/testify/assert"
)

func getURL() string {
	hostname := os.Getenv("JAMS_HTTP_HOSTNAME")
	if hostname == "" {
		hostname = "0.0.0.0"
	}
	return fmt.Sprintf("%s:3000", hostname)
}

func TestHealthCheck(t *testing.T) {
	// Arrange
	ctx := context.Background()
	client := New(getURL())

	// Act
	err := client.HealthCheck(ctx)

	// Assert
	assert.NoError(t, err)
}

func TestGetModels(t *testing.T) {
	// Arrange
	ctx := context.Background()
	client := New(getURL())

	// Act
	resp, err := client.GetModels(ctx)

	// Assert
	assert.NoError(t, err)
	assert.NotNil(t, resp)
}

func TestDeleteModel(t *testing.T) {
	// Arrange
	ctx := context.Background()
	client := New(getURL())

	// Act
	err := client.AddModel(ctx, &AddModelRequest{ModelName: "pytorch-my_awesome_californiahousing_model"})
	assert.NoError(t, err)
	err = client.DeleteModel(ctx, "my_awesome_californiahousing_model")

	// Assert
	assert.NoError(t, err)
}

func TestAddModel(t *testing.T) {
	// Arrange
	ctx := context.Background()
	client := New(getURL())

	// Act
	err := client.DeleteModel(ctx, "my_awesome_penguin_model")
	assert.NoError(t, err)
	err = client.AddModel(ctx, &AddModelRequest{ModelName: "tensorflow-my_awesome_penguin_model"})

	// Assert
	assert.NoError(t, err)
}

func TestUpdateModel(t *testing.T) {
	// Arrange
	ctx := context.Background()
	client := New(getURL())

	// Act
	err := client.UpdateModel(ctx, &UpdateModelRequest{ModelName: "titanic_model"})

	// Assert
	assert.NoError(t, err)
}

func TestPredict(t *testing.T) {
	// Arrange
	ctx := context.Background()
	client := New(getURL())

	// Act
	resp, err := client.Predict(ctx, &PredictRequest{
		ModelName: "titanic_model",
		Input:     "{\"adult_male\":[\"True\",\"False\"],\"age\":[22.0,23.79929292929293],\"alone\":[\"True\",\"False\"],\"class\":[\"First\",\"Third\"],\"deck\":[\"Unknown\",\"Unknown\"],\"embark_town\":[\"Southampton\",\"Cherbourg\"],\"embarked\":[\"S\",\"C\"],\"fare\":[151.55,14.4542],\"parch\":[\"0\",\"0\"],\"pclass\":[\"1\",\"3\"],\"sex\":[\"male\",\"female\"],\"sibsp\":[\"0\",\"1\"],\"who\":[\"man\",\"woman\"]}",
	})

	// Assert
	assert.NoError(t, err)
	assert.NotNil(t, resp)
}
