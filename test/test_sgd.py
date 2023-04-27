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
import unittest
from test import support
import damo
import numpy as np


class SGDTestCase(unittest.TestCase):
    def setUp(self):
        self.storage = damo.PyStorage("/tmp/data_dir", 86400)
        init_params = damo.Parameters()
        init_params.insert("name", "truncate_normal")
        init_params.insert("mean", 0.0)
        init_params.insert("stddev", 1.0)
        self.initializer = damo.PyInitializer(init_params)
        optm_params = damo.Parameters()
        optm_params.insert("name", "sgd")
        self.lr = 0.001
        optm_params.insert("gamma", self.lr)
        optm_params.insert("lambda", 0.0)
        self.optimizer = damo.PyOptimizer(optm_params)

        self.dim = 16
        group = 0
        self.embedding = damo.PyEmbedding(
            self.storage, self.optimizer, self.initializer, self.dim, group)

    def tearDown(self):
        pass

    def test(self):
        n = 8
        keys = np.zeros(n, dtype=np.uint64)
        for i in range(n):
            keys[i] = i+1
        w = np.zeros(self.dim*keys.shape[0], dtype=np.float32)
        gds = np.random.random(self.dim*keys.shape[0]).astype(np.float32)
        self.embedding.lookup(keys, w)
        a = w - self.lr*gds
        self.embedding.apply_gradients(keys, gds)
        self.embedding.lookup(keys, w)
        assert(np.linalg.norm(a - w) == 0.0)


if __name__ == '__main__':
    unittest.main()
