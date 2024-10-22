package main

import (
	"context"
	"fmt"
	"log"
	"math"
	"os"
	"time"

	"github.com/gagansingh894/jams-rs/clients/go/jams/http"
)

func main() {
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()

	client := http.New("https://jams-http.onrender.com")

	// health check
	err := client.HealthCheck(ctx)
	if err != nil {
		log.Fatal(err)
	}

	// read request json
	data, err := os.ReadFile("request.json")
	if err != nil {
		log.Fatalf("failed to read file: %v", err)
	}

	// predict
	predictions, err := client.Predict(ctx, &http.PredictRequest{
		ModelName: "titanic_model",
		Input:     string(data),
	})
	if err != nil {
		log.Fatal(err)
	}

	//this is a binary classifier model and will return logits of each input record
	fmt.Println("CATBOOST RESPONSE")
	fmt.Printf("logits: %+v\n", predictions.Values())

	// apply sigmoid to the 2D array to get the probabilities
	outputs := applySigmoid(predictions.Values())
	fmt.Printf("probabilities: %+v\n", outputs)

	// get class label
	fmt.Printf("class labels: %+v\n", applyClassLabel(outputs))
}

// Sigmoid function
func sigmoid(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}

func applySigmoid(inputs [][]float64) [][]float64 {
	// Create a new 2D slice to store the results
	outputs := make([][]float64, len(inputs))

	// Apply sigmoid function to each element in the 2D array
	for i, row := range inputs {
		outputs[i] = make([]float64, len(row))
		for j, value := range row {
			outputs[i][j] = sigmoid(value)
		}
	}
	return outputs
}

// get class labels from probabilities
func getClassLabel(input float64) int {
	if input >= 0.5 {
		return 1
	}

	return 0
}

func applyClassLabel(inputs [][]float64) []int {
	// Create a new 2D slice to store the results
	outputs := make([]int, len(inputs))

	// Apply sigmoid function to each element in the 2D array
	for i, row := range inputs {
		for _, value := range row {
			outputs[i] = getClassLabel(value)
		}
	}
	return outputs
}
