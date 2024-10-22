package main

import (
	"context"
	"fmt"
	"log"
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
		ModelName: "my_awesome_penguin_model",
		Input:     string(data),
	})
	if err != nil {
		log.Fatal(err)
	}

	// this will return a multiclass response for each input record. we can use argmax to get the index of the class
	fmt.Println("TENSORFLOW PREDICTIONS")
	fmt.Printf("penguin species labels: %+v\n", applyArgMax(predictions.Values()))
}

// Argmax function returns the index of the maximum value in a slice
func argmax(arr []float64) int {
	if len(arr) == 0 {
		return -1 // Return -1 for an empty array (no valid index)
	}

	maxIndex := 0
	maxValue := arr[0]

	for i, value := range arr {
		if value > maxValue {
			maxValue = value
			maxIndex = i
		}
	}

	return maxIndex
}

func applyArgMax(inputs [][]float64) []int {
	// Create a new 2D slice to store the results
	outputs := make([]int, len(inputs))

	// Apply sigmoid function to each element in the 2D array
	for i, row := range inputs {
		outputs[i] = argmax(row)
	}

	return outputs
}
