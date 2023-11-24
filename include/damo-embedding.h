//
// `Damo-Embedding` - 'c++ tool for sparse parameter server'
// Copyright (C) 2019 - present timepi <timepi123@gmail.com>
// `Damo-Embedding` is provided under: GNU Affero General Public License
// (AGPL3.0) https://www.gnu.org/licenses/agpl-3.0.html unless stated otherwise.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License as
// published by the Free Software Foundation.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Affero General Public License for more details.
//

#ifndef DAMO_EMBEDDING_H
#define DAMO_EMBEDDING_H

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

void damo_embedding_open(int ttl, char *path, int len);
void damo_embedding_close();
void damo_embedding_new(char *data, int len);
void damo_embedding_dump(char *dir, int len);
void damo_embedding_checkpoint(char *dir, int len);
void damo_embedding_load(char *dir, int len);
void damo_embedding_pull(int group, void *keys, int klen, void *w, int wlen);
void damo_embedding_push(int group, void *keys, int klen, void *g, int glen);

#ifdef __cplusplus
} /* end extern "C"*/
#endif

#endif // DAMO_EMBEDDING_H