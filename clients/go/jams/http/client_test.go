package http

import (
	"context"
	"github.com/gagansingh894/jams-rs/clients/go/jams/types"
	"net/http"
	"reflect"
	"testing"
)

func TestNew(t *testing.T) {
	type args struct {
		baseURL string
	}
	tests := []struct {
		name string
		args args
		want Client
	}{
		// TODO: Add test cases.
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := New(tt.args.baseURL); !reflect.DeepEqual(got, tt.want) {
				t.Errorf("New() = %v, want %v", got, tt.want)
			}
		})
	}
}

func Test_client_AddModel(t *testing.T) {
	type fields struct {
		baseURL string
		Client  http.Client
	}
	type args struct {
		ctx     context.Context
		request *AddModelRequest
	}
	tests := []struct {
		name    string
		fields  fields
		args    args
		wantErr bool
	}{
		// TODO: Add test cases.
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			c := &client{
				baseURL: tt.fields.baseURL,
				Client:  tt.fields.Client,
			}
			if err := c.AddModel(tt.args.ctx, tt.args.request); (err != nil) != tt.wantErr {
				t.Errorf("AddModel() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

func Test_client_DeleteModel(t *testing.T) {
	type fields struct {
		baseURL string
		Client  http.Client
	}
	type args struct {
		ctx       context.Context
		modelName string
	}
	tests := []struct {
		name    string
		fields  fields
		args    args
		wantErr bool
	}{
		// TODO: Add test cases.
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			c := &client{
				baseURL: tt.fields.baseURL,
				Client:  tt.fields.Client,
			}
			if err := c.DeleteModel(tt.args.ctx, tt.args.modelName); (err != nil) != tt.wantErr {
				t.Errorf("DeleteModel() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

func Test_client_GetModels(t *testing.T) {
	type fields struct {
		baseURL string
		Client  http.Client
	}
	type args struct {
		ctx context.Context
	}
	tests := []struct {
		name    string
		fields  fields
		args    args
		want    *GetModelsResponse
		wantErr bool
	}{
		// TODO: Add test cases.
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			c := &client{
				baseURL: tt.fields.baseURL,
				Client:  tt.fields.Client,
			}
			got, err := c.GetModels(tt.args.ctx)
			if (err != nil) != tt.wantErr {
				t.Errorf("GetModels() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("GetModels() got = %v, want %v", got, tt.want)
			}
		})
	}
}

func Test_client_HealthCheck(t *testing.T) {
	type fields struct {
		baseURL string
		Client  http.Client
	}
	type args struct {
		ctx context.Context
	}
	tests := []struct {
		name    string
		fields  fields
		args    args
		wantErr bool
	}{
		// TODO: Add test cases.
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			c := &client{
				baseURL: tt.fields.baseURL,
				Client:  tt.fields.Client,
			}
			if err := c.HealthCheck(tt.args.ctx); (err != nil) != tt.wantErr {
				t.Errorf("HealthCheck() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

func Test_client_Predict(t *testing.T) {
	type fields struct {
		baseURL string
		Client  http.Client
	}
	type args struct {
		ctx     context.Context
		request *PredictRequest
	}
	tests := []struct {
		name    string
		fields  fields
		args    args
		want    types.Prediction
		wantErr bool
	}{
		// TODO: Add test cases.
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			c := &client{
				baseURL: tt.fields.baseURL,
				Client:  tt.fields.Client,
			}
			got, err := c.Predict(tt.args.ctx, tt.args.request)
			if (err != nil) != tt.wantErr {
				t.Errorf("Predict() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("Predict() got = %v, want %v", got, tt.want)
			}
		})
	}
}

func Test_client_UpdateModel(t *testing.T) {
	type fields struct {
		baseURL string
		Client  http.Client
	}
	type args struct {
		ctx     context.Context
		request *UpdateModelRequest
	}
	tests := []struct {
		name    string
		fields  fields
		args    args
		wantErr bool
	}{
		// TODO: Add test cases.
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			c := &client{
				baseURL: tt.fields.baseURL,
				Client:  tt.fields.Client,
			}
			if err := c.UpdateModel(tt.args.ctx, tt.args.request); (err != nil) != tt.wantErr {
				t.Errorf("UpdateModel() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}
