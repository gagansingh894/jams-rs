package types

import (
	"encoding/json"
	"fmt"
)

type Prediction [][]float64

func NewPrediction(in []byte) (Prediction, error) {
	want := Prediction{}
	err := json.Unmarshal(in, &want)
	if err != nil {
		return nil, fmt.Errorf("failed to unmarshal: %w", err)
	}

	return want, nil
}
