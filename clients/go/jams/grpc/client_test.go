package grpc

import (
	"context"
	v1 "github.com/gagansingh894/jams-rs/clients/go/jams/pkg/pb/jams"
	"github.com/gagansingh894/jams-rs/clients/go/jams/types"
	"google.golang.org/grpc"
	"google.golang.org/protobuf/types/known/emptypb"
	"reflect"
	"testing"
)

func TestNew(t *testing.T) {
	type args struct {
		url string
	}
	tests := []struct {
		name    string
		args    args
		want    Client
		wantErr bool
	}{
		// TODO: Add test cases.
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := New(tt.args.url)
			if (err != nil) != tt.wantErr {
				t.Errorf("New() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("New() got = %v, want %v", got, tt.want)
			}
		})
	}
}

func Test_client_AddModel(t *testing.T) {
	type fields struct {
		client v1.ModelServerClient
	}
	type args struct {
		ctx  context.Context
		in   *v1.AddModelRequest
		opts []grpc.CallOption
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
				client: tt.fields.client,
			}
			if err := c.AddModel(tt.args.ctx, tt.args.in, tt.args.opts...); (err != nil) != tt.wantErr {
				t.Errorf("AddModel() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

func Test_client_DeleteModel(t *testing.T) {
	type fields struct {
		client v1.ModelServerClient
	}
	type args struct {
		ctx  context.Context
		in   *v1.DeleteModelRequest
		opts []grpc.CallOption
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
				client: tt.fields.client,
			}
			if err := c.DeleteModel(tt.args.ctx, tt.args.in, tt.args.opts...); (err != nil) != tt.wantErr {
				t.Errorf("DeleteModel() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

func Test_client_GetModels(t *testing.T) {
	type fields struct {
		client v1.ModelServerClient
	}
	type args struct {
		ctx  context.Context
		in   *emptypb.Empty
		opts []grpc.CallOption
	}
	tests := []struct {
		name    string
		fields  fields
		args    args
		want    *v1.GetModelsResponse
		wantErr bool
	}{
		// TODO: Add test cases.
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			c := &client{
				client: tt.fields.client,
			}
			got, err := c.GetModels(tt.args.ctx, tt.args.in, tt.args.opts...)
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
		client v1.ModelServerClient
	}
	type args struct {
		ctx  context.Context
		in1  *emptypb.Empty
		opts []grpc.CallOption
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
				client: tt.fields.client,
			}
			if err := c.HealthCheck(tt.args.ctx, tt.args.in1, tt.args.opts...); (err != nil) != tt.wantErr {
				t.Errorf("HealthCheck() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

func Test_client_Predict(t *testing.T) {
	type fields struct {
		client v1.ModelServerClient
	}
	type args struct {
		ctx  context.Context
		in   *v1.PredictRequest
		opts []grpc.CallOption
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
				client: tt.fields.client,
			}
			got, err := c.Predict(tt.args.ctx, tt.args.in, tt.args.opts...)
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
		client v1.ModelServerClient
	}
	type args struct {
		ctx  context.Context
		in   *v1.UpdateModelRequest
		opts []grpc.CallOption
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
				client: tt.fields.client,
			}
			if err := c.UpdateModel(tt.args.ctx, tt.args.in, tt.args.opts...); (err != nil) != tt.wantErr {
				t.Errorf("UpdateModel() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}
