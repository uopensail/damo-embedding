package service

import (
	"context"
	"damo-embedding/api"
	"io"

	"github.com/gin-gonic/gin"
	"github.com/uopensail/ulib/prome"
	"github.com/uopensail/ulib/zlog"
	"go.uber.org/zap"
	"google.golang.org/grpc"
	"google.golang.org/grpc/health/grpc_health_v1"
)

var __GITHASH__ = ""

const (
	kOpenDB int32 = iota + 1
	kCloseDB
	kPull
	kPush
	kDump
	kCheckPoint
	kLoadCheckPoint
)

type Service struct {
	api.UnimplementedDamoEmbeddingServer
}

func NewService() *Service {
	srv := &Service{}
	return srv
}

func (srv *Service) Close() {
	s := DamoEmbeddingServer{}
	s.CloseDB()
}

func (srv *Service) GRPCAPIRegister(s *grpc.Server) {
	api.RegisterDamoEmbeddingServer(s, srv)
}

func (srv *Service) GinAPIRegister(e *gin.Engine) {
	e.POST("/call", srv.CallHandler)
	e.GET("/version", srv.VersionHandler)
	e.GET("/ping", srv.PingHandler)
}

func (srv *Service) CallHandler(c *gin.Context) {
	stat := prome.NewStat("Service.CallHandler")
	defer stat.End()

	request := &api.Request{}
	data, err := io.ReadAll(c.Request.Body)
	
	if err != nil {
		stat.MarkErr()
		zlog.LOG.Error("request read byte error: ", zap.Error(err))
		return
	}
	c.
	s := DamoEmbeddingServer{}
	resp := &api.Response{}
	switch request.Cmd {
	case kOpenDB:
		s.OpenDB(request.Ttl, request.Path)
	case kCloseDB:
		s.CloseDB()
	case kPull:
		weight := s.Pull(int(request.Group), request.Keys, int(request.DataLength))
		resp.
	}

	resp, err := app.Rank(context.Background(), request)
	if err != nil {
		zlog.LOG.Error("rank error: ", zap.Error(err))
		c.JSON(404, err.Error())
		return
	}
	c.JSON(200, resp)
	return
}

func (srv *Service) PingHandler(c *gin.Context) {
	c.String(200, "PONG")
}

func (srv *Service) VersionHandler(c *gin.Context) {
	c.String(200, __GITHASH__)
}

func (srv *Service) Check(ctx context.Context, req *grpc_health_v1.HealthCheckRequest) (*grpc_health_v1.HealthCheckResponse, error) {
	return &grpc_health_v1.HealthCheckResponse{
		Status: grpc_health_v1.HealthCheckResponse_SERVING,
	}, nil
}

func (srv *Service) Watch(req *grpc_health_v1.HealthCheckRequest, server grpc_health_v1.Health_WatchServer) error {
	server.Send(&grpc_health_v1.HealthCheckResponse{
		Status: grpc_health_v1.HealthCheckResponse_SERVING,
	})
	return nil
}
