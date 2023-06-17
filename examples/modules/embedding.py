#!/usr/bin/python
# -*- coding: UTF-8 -*-
#
# `Damo-Embedding` - 'c++ tool for sparse parameter server'
# Copyright (C) 2019 - present timepi <timepi123@gmail.com>
# `Damo-Embedding` is provided under: GNU Affero General Public License
# (AGPL3.0) https:#www.gnu.org/licenses/agpl-3.0.html unless stated otherwise.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as
# published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#

import damo
import numpy as np

# create storage
storage = damo.PyStorage("/tmp/data_dir", 0)

# create initializer
init_params = damo.Parameters()
init_params.insert("name", "truncate_normal")
init_params.insert("mean", 0.0)
init_params.insert("stddev", 1.0)
initializer = damo.PyInitializer(init_params)

# create optimizer
optm_params = damo.Parameters()
optm_params.insert("name", "sgd")
optm_params.insert("gamma", 0.001)
optm_params.insert("lambda", 0.0)
optimizer = damo.PyOptimizer(optm_params)

dim = 16
group = 0
embedding = damo.PyEmbedding(storage, optimizer, initializer, dim, group)

keys = np.zeros(1, dtype=np.uint64)
keys[0] = 1234567890
w = np.zeros(dim * keys.shape[0], dtype=np.float32)
gds = np.random.random(dim * keys.shape[0]).astype(np.float32)

print(w)
embedding.lookup(keys, w)
embedding.apply_gradients(keys, gds)
embedding.lookup(keys, w)
print(w)
