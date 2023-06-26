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
import unittest
import damo
import numpy as np
import os
import shutil


class FTRLTestCase(unittest.TestCase):
    def setUp(self):
        if os.path.exists("/tmp/data_dir"):
            shutil.rmtree("/tmp/data_dir")
        self.storage = damo.PyStorage("/tmp/data_dir", 86400)
        init_params = damo.Parameters()
        init_params.insert("name", "zeros")
        self.initializer = damo.PyInitializer(init_params)
        optm_params = damo.Parameters()
        optm_params.insert("name", "ftrl")
        self.gamma = 0.005
        self.beta = 0.0
        self.lambda1 = 0.0
        self.lambda2 = 0.0
        optm_params.insert("gamma", self.gamma)
        optm_params.insert("beta", self.beta)
        optm_params.insert("lambda1", self.lambda1)
        optm_params.insert("lambda2", self.lambda2)
        self.optimizer = damo.PyOptimizer(optm_params)

        self.dim = 16
        group = 0
        self.embedding = damo.PyEmbedding(
            self.storage, self.optimizer, self.initializer, self.dim, group
        )

    def tearDown(self):
        pass

    def test(self):
        n = 8
        keys = np.random.randint(1, 10000 + 1, n, dtype=np.int64)
        N = np.zeros(self.dim * n, dtype=np.float32)
        z = np.zeros(self.dim * n, dtype=np.float32)
        w = np.zeros(self.dim * n, dtype=np.float32)
        new_w = np.zeros(self.dim * n, dtype=np.float32)
        for i in range(100):
            self.embedding.lookup(keys, w)
            gds = np.random.random(self.dim * n).astype(np.float32)
            n1 = N + gds * gds
            delta = (1.0 / self.gamma) * (np.sqrt(n1) - np.sqrt(N))
            z = z + gds - w * delta
            N = n1
            tmp = np.where(np.abs(z) < self.lambda1, 0, z - np.sign(z) * self.lambda1)
            new_w = -1.0 / ((self.beta + np.sqrt(N)) / self.gamma + self.lambda2) * tmp
            self.embedding.apply_gradients(keys, gds)
            self.embedding.lookup(keys, w)
            print(np.linalg.norm(w - new_w))


if __name__ == "__main__":
    unittest.main()