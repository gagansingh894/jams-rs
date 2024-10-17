# J.A.M.S Go Client


A HTTP & gRPC client for `J.A.M.S - Just Another Model Server`

## Installation
```
go get github.com/gagansingh894/jams-rs/clients/go/jams
```

## Usage

Start `J.A.M.S` by following the instructions [here](https://github.com/gagansingh894/jams-rs?tab=readme-ov-file#docker-setup)

### HTTP

```
package main

import (
	"context"
	"fmt"
	"log"
	"time"

	"github.com/gagansingh894/jams-rs/clients/go/jams/http"
)

func main() {
	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()

	client := http.New("0.0.0.0:3000")

	// health check
	err := client.HealthCheck(ctx)
	if err != nil {
		log.Fatal(err)
	}

	// predict
	predictions, err := client.Predict(ctx, &http.PredictRequest{
		ModelName: "titanic_model",
		Input:     "{\"adult_male\":[\"True\",\"False\"],\"age\":[22.0,23.79929292929293],\"alone\":[\"True\",\"False\"],\"class\":[\"First\",\"Third\"],\"deck\":[\"Unknown\",\"Unknown\"],\"embark_town\":[\"Southampton\",\"Cherbourg\"],\"embarked\":[\"S\",\"C\"],\"fare\":[151.55,14.4542],\"parch\":[\"0\",\"0\"],\"pclass\":[\"1\",\"3\"],\"sex\":[\"male\",\"female\"],\"sibsp\":[\"0\",\"1\"],\"who\":[\"man\",\"woman\"]}",
	})
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(predictions.Values()) // use values

	// add model
	err = client.AddModel(ctx, &http.AddModelRequest{ModelName: "catboost-titanic_model"})
	if err != nil {
		log.Fatal(err)
	}

	// update model
	err = client.UpdateModel(ctx, &http.UpdateModelRequest{ModelName: "titanic_model"})
	if err != nil {
		log.Fatal(err)
	}

	// delete model
	err = client.DeleteModel(ctx, "my_awesome_californiahousing_model")
	if err != nil {
		log.Fatal(err)
	}

	// get models
	models, err := client.GetModels(ctx)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("%+v \n", models)
}
```

### gRPC

```
package main

import (
	"context"
	"fmt"
	"log"
	"time"

	"google.golang.org/protobuf/types/known/emptypb"

	"github.com/gagansingh894/jams-rs/clients/go/jams/grpc"
	pb "github.com/gagansingh894/jams-rs/clients/go/jams/pkg/pb/jams"
)

func main() {
	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()

	client, err := grpc.New("0.0.0.0:4000")
	if err != nil {
		log.Fatal(err)
	}

	// health check
	err = client.HealthCheck(ctx, &emptypb.Empty{})
	if err != nil {
		log.Fatal(err)
	}

	// predict
	predictions, err := client.Predict(ctx, &pb.PredictRequest{
		ModelName: "titanic_model",
		Input:     "{\"adult_male\":[\"True\",\"False\"],\"age\":[22.0,23.79929292929293],\"alone\":[\"True\",\"False\"],\"class\":[\"First\",\"Third\"],\"deck\":[\"Unknown\",\"Unknown\"],\"embark_town\":[\"Southampton\",\"Cherbourg\"],\"embarked\":[\"S\",\"C\"],\"fare\":[151.55,14.4542],\"parch\":[\"0\",\"0\"],\"pclass\":[\"1\",\"3\"],\"sex\":[\"male\",\"female\"],\"sibsp\":[\"0\",\"1\"],\"who\":[\"man\",\"woman\"]}",
	})
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(predictions.Values()) // use values

	// add model
	err = client.AddModel(ctx, &pb.AddModelRequest{ModelName: "catboost-titanic_model"})
	if err != nil {
		log.Fatal(err)
	}

	// update model
	err = client.UpdateModel(ctx, &pb.UpdateModelRequest{ModelName: "titanic_model"})
	if err != nil {
		log.Fatal(err)
	}

	// delete model
	err = client.DeleteModel(ctx, &pb.DeleteModelRequest{ModelName: "my_awesome_californiahousing_model"})
	if err != nil {
		log.Fatal(err)
	}

	// get models
	models, err := client.GetModels(ctx, &emptypb.Empty{})
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("%+v \n", models)
}

```