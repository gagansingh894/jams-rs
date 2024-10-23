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
		ModelName: "my_awesome_californiahousing_model",
		Input:     string(data),
	})
	if err != nil {
		log.Fatal(err)
	}

	// this is a regression model so output would be continous for each input record
	fmt.Println("TORCH PREDICTIONS")
	fmt.Printf("valuess: %+v\n", predictions.Values())

}
