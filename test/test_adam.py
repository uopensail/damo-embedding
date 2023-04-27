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
import os


class SGDTestCase(unittest.TestCase):
    def setUp(self):
        if os.path.exists("/tmp/data_dir"):
            os.remove("/tmp/data_dir")
        self.storage = damo.PyStorage("/tmp/data_dir", 86400)
        init_params = damo.Parameters()
        init_params.insert("name", "truncate_normal")
        init_params.insert("mean", 0.0)
        init_params.insert("stddev", 1.0)
        self.initializer = damo.PyInitializer(init_params)
        optm_params = damo.Parameters()
        optm_params.insert("name", "adam")
        self.gamma = 0.001
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.lambda_ = 0.0
        self.epsilon = 1e-8
        optm_params.insert("gamma", self.gamma)
        optm_params.insert("beta1", self.beta1)
        optm_params.insert("beta2", self.beta2)
        optm_params.insert("lambda", self.lambda_)
        optm_params.insert("epsilon", self.epsilon)
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
        m = np.zeros(self.dim*keys.shape[0], dtype=np.float32)
        v = np.zeros(self.dim*keys.shape[0], dtype=np.float32)
        w = np.zeros(self.dim*keys.shape[0], dtype=np.float32)
        gds = np.random.random(self.dim*keys.shape[0]).astype(np.float32)
        m = self.beta1*m + (1.0-self.beta1)*gds
        v = self.beta2*v + (1.0-self.beta2)*gds*gds
        self.embedding.lookup(keys, w)

        m_t = m/(1.0-self.beta1)
        v_t = v/(1.0-self.beta2)
        a = w - self.gamma*m_t/(np.sqrt(v_t)+self.epsilon)

        self.embedding.apply_gradients(keys, gds)
        self.embedding.lookup(keys, w)

        assert(np.linalg.norm(a - w) == 0.0)


if __name__ == '__main__':
    unittest.main()
