package types

import (
	"encoding/json"
	"fmt"
)

type Prediction map[string][][]float64

func NewPrediction(in []byte) (Prediction, error) {
	want := Prediction{}
	err := json.Unmarshal(in, &want)
	if err != nil {
		return nil, fmt.Errorf("failed to unmarshal: %w", err)
	}

	return want, nil
}

func (p Prediction) Values() [][]float64 {
	var value [][]float64

	// Loop over the map to get the value (since we know there is only one key)
	for _, v := range p {
		value = v
		break // Stop after the first (and only) iteration
	}

	return value
}
