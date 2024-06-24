// Code generated by mockery v2.43.2. DO NOT EDIT.

package mocks

import (
	context "context"

	grpc "google.golang.org/grpc"
	emptypb "google.golang.org/protobuf/types/known/emptypb"

	jams "github.com/gagansingh894/jams-rs/clients/go/jams/pkg/pb/jams"

	mock "github.com/stretchr/testify/mock"

	types "github.com/gagansingh894/jams-rs/clients/go/jams/types"
)

// Client is an autogenerated mock type for the Client type
type Client struct {
	mock.Mock
}

// AddModel provides a mock function with given fields: ctx, in, opts
func (_m *Client) AddModel(ctx context.Context, in *jams.AddModelRequest, opts ...grpc.CallOption) error {
	_va := make([]interface{}, len(opts))
	for _i := range opts {
		_va[_i] = opts[_i]
	}
	var _ca []interface{}
	_ca = append(_ca, ctx, in)
	_ca = append(_ca, _va...)
	ret := _m.Called(_ca...)

	if len(ret) == 0 {
		panic("no return value specified for AddModel")
	}

	var r0 error
	if rf, ok := ret.Get(0).(func(context.Context, *jams.AddModelRequest, ...grpc.CallOption) error); ok {
		r0 = rf(ctx, in, opts...)
	} else {
		r0 = ret.Error(0)
	}

	return r0
}

// DeleteModel provides a mock function with given fields: ctx, in, opts
func (_m *Client) DeleteModel(ctx context.Context, in *jams.DeleteModelRequest, opts ...grpc.CallOption) error {
	_va := make([]interface{}, len(opts))
	for _i := range opts {
		_va[_i] = opts[_i]
	}
	var _ca []interface{}
	_ca = append(_ca, ctx, in)
	_ca = append(_ca, _va...)
	ret := _m.Called(_ca...)

	if len(ret) == 0 {
		panic("no return value specified for DeleteModel")
	}

	var r0 error
	if rf, ok := ret.Get(0).(func(context.Context, *jams.DeleteModelRequest, ...grpc.CallOption) error); ok {
		r0 = rf(ctx, in, opts...)
	} else {
		r0 = ret.Error(0)
	}

	return r0
}

// GetModels provides a mock function with given fields: ctx, in, opts
func (_m *Client) GetModels(ctx context.Context, in *emptypb.Empty, opts ...grpc.CallOption) (*jams.GetModelsResponse, error) {
	_va := make([]interface{}, len(opts))
	for _i := range opts {
		_va[_i] = opts[_i]
	}
	var _ca []interface{}
	_ca = append(_ca, ctx, in)
	_ca = append(_ca, _va...)
	ret := _m.Called(_ca...)

	if len(ret) == 0 {
		panic("no return value specified for GetModels")
	}

	var r0 *jams.GetModelsResponse
	var r1 error
	if rf, ok := ret.Get(0).(func(context.Context, *emptypb.Empty, ...grpc.CallOption) (*jams.GetModelsResponse, error)); ok {
		return rf(ctx, in, opts...)
	}
	if rf, ok := ret.Get(0).(func(context.Context, *emptypb.Empty, ...grpc.CallOption) *jams.GetModelsResponse); ok {
		r0 = rf(ctx, in, opts...)
	} else {
		if ret.Get(0) != nil {
			r0 = ret.Get(0).(*jams.GetModelsResponse)
		}
	}

	if rf, ok := ret.Get(1).(func(context.Context, *emptypb.Empty, ...grpc.CallOption) error); ok {
		r1 = rf(ctx, in, opts...)
	} else {
		r1 = ret.Error(1)
	}

	return r0, r1
}

// HealthCheck provides a mock function with given fields: ctx, in, opts
func (_m *Client) HealthCheck(ctx context.Context, in *emptypb.Empty, opts ...grpc.CallOption) error {
	_va := make([]interface{}, len(opts))
	for _i := range opts {
		_va[_i] = opts[_i]
	}
	var _ca []interface{}
	_ca = append(_ca, ctx, in)
	_ca = append(_ca, _va...)
	ret := _m.Called(_ca...)

	if len(ret) == 0 {
		panic("no return value specified for HealthCheck")
	}

	var r0 error
	if rf, ok := ret.Get(0).(func(context.Context, *emptypb.Empty, ...grpc.CallOption) error); ok {
		r0 = rf(ctx, in, opts...)
	} else {
		r0 = ret.Error(0)
	}

	return r0
}

// Predict provides a mock function with given fields: ctx, in, opts
func (_m *Client) Predict(ctx context.Context, in *jams.PredictRequest, opts ...grpc.CallOption) (types.Prediction, error) {
	_va := make([]interface{}, len(opts))
	for _i := range opts {
		_va[_i] = opts[_i]
	}
	var _ca []interface{}
	_ca = append(_ca, ctx, in)
	_ca = append(_ca, _va...)
	ret := _m.Called(_ca...)

	if len(ret) == 0 {
		panic("no return value specified for Predict")
	}

	var r0 types.Prediction
	var r1 error
	if rf, ok := ret.Get(0).(func(context.Context, *jams.PredictRequest, ...grpc.CallOption) (types.Prediction, error)); ok {
		return rf(ctx, in, opts...)
	}
	if rf, ok := ret.Get(0).(func(context.Context, *jams.PredictRequest, ...grpc.CallOption) types.Prediction); ok {
		r0 = rf(ctx, in, opts...)
	} else {
		if ret.Get(0) != nil {
			r0 = ret.Get(0).(types.Prediction)
		}
	}

	if rf, ok := ret.Get(1).(func(context.Context, *jams.PredictRequest, ...grpc.CallOption) error); ok {
		r1 = rf(ctx, in, opts...)
	} else {
		r1 = ret.Error(1)
	}

	return r0, r1
}

// UpdateModel provides a mock function with given fields: ctx, in, opts
func (_m *Client) UpdateModel(ctx context.Context, in *jams.UpdateModelRequest, opts ...grpc.CallOption) error {
	_va := make([]interface{}, len(opts))
	for _i := range opts {
		_va[_i] = opts[_i]
	}
	var _ca []interface{}
	_ca = append(_ca, ctx, in)
	_ca = append(_ca, _va...)
	ret := _m.Called(_ca...)

	if len(ret) == 0 {
		panic("no return value specified for UpdateModel")
	}

	var r0 error
	if rf, ok := ret.Get(0).(func(context.Context, *jams.UpdateModelRequest, ...grpc.CallOption) error); ok {
		r0 = rf(ctx, in, opts...)
	} else {
		r0 = ret.Error(0)
	}

	return r0
}

// NewClient creates a new instance of Client. It also registers a testing interface on the mock and a cleanup function to assert the mocks expectations.
// The first argument is typically a *testing.T value.
func NewClient(t interface {
	mock.TestingT
	Cleanup(func())
}) *Client {
	mock := &Client{}
	mock.Mock.Test(t)

	t.Cleanup(func() { mock.AssertExpectations(t) })

	return mock
}
