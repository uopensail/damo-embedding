package service

/*
#cgo darwin,amd64 pkg-config: ${SRCDIR}/../third/damo-embedding-darwin-amd64.pc
#cgo darwin,arm64 pkg-config: ${SRCDIR}/../third/damo-embedding-darwin-arm64.pc
#cgo linux,amd64 pkg-config: ${SRCDIR}/../third/damo-embedding-linux-amd64.pc
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include "damo-embedding.h"
*/
import "C"

import (
	"reflect"
	"unsafe"
)

type DamoEmbeddingServer struct{}

func (s *DamoEmbeddingServer) OpenDB(ttl int32, path string) {
	C.damo_embedding_open(C.int(ttl), (*C.char)(unsafe.Pointer(&s2b(path)[0])), C.int(len(path)))
}

func (s *DamoEmbeddingServer) CloseDB() {
	C.damo_embedding_close()
}

func (s *DamoEmbeddingServer) CreateEmbedding(params string) {
	C.damo_embedding_new((*C.char)(unsafe.Pointer(&s2b(params)[0])), C.int(len(params)))
}

func (s *DamoEmbeddingServer) Dump(dir string) {
	C.damo_embedding_dump((*C.char)(unsafe.Pointer(&s2b(dir)[0])), C.int(len(dir)))
}

func (s *DamoEmbeddingServer) Checkpoint(dir string) {
	C.damo_embedding_checkpoint((*C.char)(unsafe.Pointer(&s2b(dir)[0])), C.int(len(dir)))
}

func (s *DamoEmbeddingServer) Load(dir string) {
	C.damo_embedding_load((*C.char)(unsafe.Pointer(&s2b(dir)[0])), C.int(len(dir)))
}

func (s *DamoEmbeddingServer) Pull(group int, keys []int64, size int) []float32 {
	weights := make([]float32, size)
	C.damo_embedding_pull(
		C.int(group),
		unsafe.Pointer(&keys[0]),
		C.int(len(keys)),
		unsafe.Pointer(&weights[0]),
		C.int(size),
	)
	return weights
}

func (s *DamoEmbeddingServer) Push(group int, keys []int64, gds []float32) {
	C.damo_embedding_push(
		C.int(group),
		unsafe.Pointer(&keys[0]),
		C.int(len(keys)),
		unsafe.Pointer(&gds[0]),
		C.int(len(gds)),
	)
}

func s2b(s string) (b []byte) {
	/* #nosec G103 */
	bh := (*reflect.SliceHeader)(unsafe.Pointer(&b))
	/* #nosec G103 */
	sh := (*reflect.StringHeader)(unsafe.Pointer(&s))
	bh.Data = sh.Data
	bh.Cap = sh.Len
	bh.Len = sh.Len
	return b
}
