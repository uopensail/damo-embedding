#
# `Damo-Embedding` - 'c++ tool for sparse parameter server'
# Copyright(C) 2019 - present timepi < timepi123@gmail.com >
#
# This file is part of `Damo-Embedding`.
#
# `Damo-Embedding` is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# `Damo-Embedding` is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY
# without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with `Damo-Embedding`.  If not, see < http: # www.gnu.org/licenses/>.
#

import damo
import numpy as np

# create storage
storage = damo.PyStorage("/tmp/data_dir", 86400*100)

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
w = np.zeros(dim*keys.shape[0], dtype=np.float32)
gds = np.random.random(dim*keys.shape[0]).astype(np.float32)

print('original weight: ', w)
embedding.lookup(keys, w)
print('initialized weight: ', w)
print('new weights', w - gds*0.001)
print('gradients: ', gds)
embedding.apply_gradients(keys, gds)
embedding.lookup(keys, w)
print('apply gradients weight: ', w)
embedding.lookup(keys, w)
print('apply gradients weight: ', w)
