// Code generated by protoc-gen-go-grpc. DO NOT EDIT.
// versions:
// - protoc-gen-go-grpc v1.4.0
// - protoc             v5.27.1
// source: jams.proto

package jams

import (
	context "context"
	grpc "google.golang.org/grpc"
	codes "google.golang.org/grpc/codes"
	status "google.golang.org/grpc/status"
	emptypb "google.golang.org/protobuf/types/known/emptypb"
)

// This is a compile-time assertion to ensure that this generated file
// is compatible with the grpc package it is being compiled against.
// Requires gRPC-Go v1.62.0 or later.
const _ = grpc.SupportPackageIsVersion8

const (
	ModelServer_HealthCheck_FullMethodName = "/jams_v1.ModelServer/HealthCheck"
	ModelServer_Predict_FullMethodName     = "/jams_v1.ModelServer/Predict"
	ModelServer_GetModels_FullMethodName   = "/jams_v1.ModelServer/GetModels"
	ModelServer_AddModel_FullMethodName    = "/jams_v1.ModelServer/AddModel"
	ModelServer_UpdateModel_FullMethodName = "/jams_v1.ModelServer/UpdateModel"
	ModelServer_DeleteModel_FullMethodName = "/jams_v1.ModelServer/DeleteModel"
)

// ModelServerClient is the client API for ModelServer service.
//
// For semantics around ctx use and closing/ending streaming RPCs, please refer to https://pkg.go.dev/google.golang.org/grpc/?tab=doc#ClientConn.NewStream.
//
// Service definition for model server.
type ModelServerClient interface {
	// HealthCheck is used to check the server health
	HealthCheck(ctx context.Context, in *emptypb.Empty, opts ...grpc.CallOption) (*emptypb.Empty, error)
	// Predict is used to make predictions based on provided input.
	Predict(ctx context.Context, in *PredictRequest, opts ...grpc.CallOption) (*PredictResponse, error)
	// GetModels is used to get the list of models which are loaded into memory.
	GetModels(ctx context.Context, in *emptypb.Empty, opts ...grpc.CallOption) (*GetModelsResponse, error)
	// AddModel adds a new model to the model server.
	AddModel(ctx context.Context, in *AddModelRequest, opts ...grpc.CallOption) (*emptypb.Empty, error)
	// UpdateModel updates an existing model in the model server.
	UpdateModel(ctx context.Context, in *UpdateModelRequest, opts ...grpc.CallOption) (*emptypb.Empty, error)
	// DeleteModel deletes an existing model from the server.
	DeleteModel(ctx context.Context, in *DeleteModelRequest, opts ...grpc.CallOption) (*emptypb.Empty, error)
}

type modelServerClient struct {
	cc grpc.ClientConnInterface
}

func NewModelServerClient(cc grpc.ClientConnInterface) ModelServerClient {
	return &modelServerClient{cc}
}

func (c *modelServerClient) HealthCheck(ctx context.Context, in *emptypb.Empty, opts ...grpc.CallOption) (*emptypb.Empty, error) {
	cOpts := append([]grpc.CallOption{grpc.StaticMethod()}, opts...)
	out := new(emptypb.Empty)
	err := c.cc.Invoke(ctx, ModelServer_HealthCheck_FullMethodName, in, out, cOpts...)
	if err != nil {
		return nil, err
	}
	return out, nil
}

func (c *modelServerClient) Predict(ctx context.Context, in *PredictRequest, opts ...grpc.CallOption) (*PredictResponse, error) {
	cOpts := append([]grpc.CallOption{grpc.StaticMethod()}, opts...)
	out := new(PredictResponse)
	err := c.cc.Invoke(ctx, ModelServer_Predict_FullMethodName, in, out, cOpts...)
	if err != nil {
		return nil, err
	}
	return out, nil
}

func (c *modelServerClient) GetModels(ctx context.Context, in *emptypb.Empty, opts ...grpc.CallOption) (*GetModelsResponse, error) {
	cOpts := append([]grpc.CallOption{grpc.StaticMethod()}, opts...)
	out := new(GetModelsResponse)
	err := c.cc.Invoke(ctx, ModelServer_GetModels_FullMethodName, in, out, cOpts...)
	if err != nil {
		return nil, err
	}
	return out, nil
}

func (c *modelServerClient) AddModel(ctx context.Context, in *AddModelRequest, opts ...grpc.CallOption) (*emptypb.Empty, error) {
	cOpts := append([]grpc.CallOption{grpc.StaticMethod()}, opts...)
	out := new(emptypb.Empty)
	err := c.cc.Invoke(ctx, ModelServer_AddModel_FullMethodName, in, out, cOpts...)
	if err != nil {
		return nil, err
	}
	return out, nil
}

func (c *modelServerClient) UpdateModel(ctx context.Context, in *UpdateModelRequest, opts ...grpc.CallOption) (*emptypb.Empty, error) {
	cOpts := append([]grpc.CallOption{grpc.StaticMethod()}, opts...)
	out := new(emptypb.Empty)
	err := c.cc.Invoke(ctx, ModelServer_UpdateModel_FullMethodName, in, out, cOpts...)
	if err != nil {
		return nil, err
	}
	return out, nil
}

func (c *modelServerClient) DeleteModel(ctx context.Context, in *DeleteModelRequest, opts ...grpc.CallOption) (*emptypb.Empty, error) {
	cOpts := append([]grpc.CallOption{grpc.StaticMethod()}, opts...)
	out := new(emptypb.Empty)
	err := c.cc.Invoke(ctx, ModelServer_DeleteModel_FullMethodName, in, out, cOpts...)
	if err != nil {
		return nil, err
	}
	return out, nil
}

// ModelServerServer is the server API for ModelServer service.
// All implementations must embed UnimplementedModelServerServer
// for forward compatibility
//
// Service definition for model server.
type ModelServerServer interface {
	// HealthCheck is used to check the server health
	HealthCheck(context.Context, *emptypb.Empty) (*emptypb.Empty, error)
	// Predict is used to make predictions based on provided input.
	Predict(context.Context, *PredictRequest) (*PredictResponse, error)
	// GetModels is used to get the list of models which are loaded into memory.
	GetModels(context.Context, *emptypb.Empty) (*GetModelsResponse, error)
	// AddModel adds a new model to the model server.
	AddModel(context.Context, *AddModelRequest) (*emptypb.Empty, error)
	// UpdateModel updates an existing model in the model server.
	UpdateModel(context.Context, *UpdateModelRequest) (*emptypb.Empty, error)
	// DeleteModel deletes an existing model from the server.
	DeleteModel(context.Context, *DeleteModelRequest) (*emptypb.Empty, error)
	mustEmbedUnimplementedModelServerServer()
}

// UnimplementedModelServerServer must be embedded to have forward compatible implementations.
type UnimplementedModelServerServer struct {
}

func (UnimplementedModelServerServer) HealthCheck(context.Context, *emptypb.Empty) (*emptypb.Empty, error) {
	return nil, status.Errorf(codes.Unimplemented, "method HealthCheck not implemented")
}
func (UnimplementedModelServerServer) Predict(context.Context, *PredictRequest) (*PredictResponse, error) {
	return nil, status.Errorf(codes.Unimplemented, "method Predict not implemented")
}
func (UnimplementedModelServerServer) GetModels(context.Context, *emptypb.Empty) (*GetModelsResponse, error) {
	return nil, status.Errorf(codes.Unimplemented, "method GetModels not implemented")
}
func (UnimplementedModelServerServer) AddModel(context.Context, *AddModelRequest) (*emptypb.Empty, error) {
	return nil, status.Errorf(codes.Unimplemented, "method AddModel not implemented")
}
func (UnimplementedModelServerServer) UpdateModel(context.Context, *UpdateModelRequest) (*emptypb.Empty, error) {
	return nil, status.Errorf(codes.Unimplemented, "method UpdateModel not implemented")
}
func (UnimplementedModelServerServer) DeleteModel(context.Context, *DeleteModelRequest) (*emptypb.Empty, error) {
	return nil, status.Errorf(codes.Unimplemented, "method DeleteModel not implemented")
}
func (UnimplementedModelServerServer) mustEmbedUnimplementedModelServerServer() {}

// UnsafeModelServerServer may be embedded to opt out of forward compatibility for this service.
// Use of this interface is not recommended, as added methods to ModelServerServer will
// result in compilation errors.
type UnsafeModelServerServer interface {
	mustEmbedUnimplementedModelServerServer()
}

func RegisterModelServerServer(s grpc.ServiceRegistrar, srv ModelServerServer) {
	s.RegisterService(&ModelServer_ServiceDesc, srv)
}

func _ModelServer_HealthCheck_Handler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(emptypb.Empty)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(ModelServerServer).HealthCheck(ctx, in)
	}
	info := &grpc.UnaryServerInfo{
		Server:     srv,
		FullMethod: ModelServer_HealthCheck_FullMethodName,
	}
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(ModelServerServer).HealthCheck(ctx, req.(*emptypb.Empty))
	}
	return interceptor(ctx, in, info, handler)
}

func _ModelServer_Predict_Handler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(PredictRequest)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(ModelServerServer).Predict(ctx, in)
	}
	info := &grpc.UnaryServerInfo{
		Server:     srv,
		FullMethod: ModelServer_Predict_FullMethodName,
	}
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(ModelServerServer).Predict(ctx, req.(*PredictRequest))
	}
	return interceptor(ctx, in, info, handler)
}

func _ModelServer_GetModels_Handler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(emptypb.Empty)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(ModelServerServer).GetModels(ctx, in)
	}
	info := &grpc.UnaryServerInfo{
		Server:     srv,
		FullMethod: ModelServer_GetModels_FullMethodName,
	}
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(ModelServerServer).GetModels(ctx, req.(*emptypb.Empty))
	}
	return interceptor(ctx, in, info, handler)
}

func _ModelServer_AddModel_Handler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(AddModelRequest)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(ModelServerServer).AddModel(ctx, in)
	}
	info := &grpc.UnaryServerInfo{
		Server:     srv,
		FullMethod: ModelServer_AddModel_FullMethodName,
	}
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(ModelServerServer).AddModel(ctx, req.(*AddModelRequest))
	}
	return interceptor(ctx, in, info, handler)
}

func _ModelServer_UpdateModel_Handler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(UpdateModelRequest)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(ModelServerServer).UpdateModel(ctx, in)
	}
	info := &grpc.UnaryServerInfo{
		Server:     srv,
		FullMethod: ModelServer_UpdateModel_FullMethodName,
	}
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(ModelServerServer).UpdateModel(ctx, req.(*UpdateModelRequest))
	}
	return interceptor(ctx, in, info, handler)
}

func _ModelServer_DeleteModel_Handler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(DeleteModelRequest)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(ModelServerServer).DeleteModel(ctx, in)
	}
	info := &grpc.UnaryServerInfo{
		Server:     srv,
		FullMethod: ModelServer_DeleteModel_FullMethodName,
	}
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(ModelServerServer).DeleteModel(ctx, req.(*DeleteModelRequest))
	}
	return interceptor(ctx, in, info, handler)
}

// ModelServer_ServiceDesc is the grpc.ServiceDesc for ModelServer service.
// It's only intended for direct use with grpc.RegisterService,
// and not to be introspected or modified (even as a copy)
var ModelServer_ServiceDesc = grpc.ServiceDesc{
	ServiceName: "jams_v1.ModelServer",
	HandlerType: (*ModelServerServer)(nil),
	Methods: []grpc.MethodDesc{
		{
			MethodName: "HealthCheck",
			Handler:    _ModelServer_HealthCheck_Handler,
		},
		{
			MethodName: "Predict",
			Handler:    _ModelServer_Predict_Handler,
		},
		{
			MethodName: "GetModels",
			Handler:    _ModelServer_GetModels_Handler,
		},
		{
			MethodName: "AddModel",
			Handler:    _ModelServer_AddModel_Handler,
		},
		{
			MethodName: "UpdateModel",
			Handler:    _ModelServer_UpdateModel_Handler,
		},
		{
			MethodName: "DeleteModel",
			Handler:    _ModelServer_DeleteModel_Handler,
		},
	},
	Streams:  []grpc.StreamDesc{},
	Metadata: "jams.proto",
}
