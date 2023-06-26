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


class LionTestCase(unittest.TestCase):
    def setUp(self):
        if os.path.exists("/tmp/data_dir"):
            shutil.rmtree("/tmp/data_dir")
        self.storage = damo.PyStorage("/tmp/data_dir", 86400)
        init_params = damo.Parameters()
        init_params.insert("name", "truncate_normal")
        init_params.insert("mean", 0.0)
        init_params.insert("stddev", 1.0)
        self.initializer = damo.PyInitializer(init_params)
        optm_params = damo.Parameters()
        optm_params.insert("name", "lion")
        self.eta = 0.003
        self.beta1 = 0.9
        self.beta2 = 0.99
        self.lambda_ = 0.01
        optm_params.insert("eta", self.eta)
        optm_params.insert("beta1", self.beta1)
        optm_params.insert("beta2", self.beta2)
        optm_params.insert("lambda", self.lambda_)
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
        w = np.zeros(self.dim * n, dtype=np.float32)
        self.embedding.lookup(keys, w)
        m = np.zeros(self.dim * n).astype(np.float32)
        gds = np.random.random(self.dim * n).astype(np.float32)
        tmp_mu = np.sign(self.beta1 * m + (1.0 - self.beta1)
                         * gds) + w * self.lambda_
        m = self.beta2 * +(1.0 - self.beta2) * gds
        self.embedding.apply_gradients(keys, gds)
        new_w = w - self.eta * tmp_mu
        self.embedding.lookup(keys, w)

        assert np.linalg.norm(new_w - w) == 0.0


if __name__ == "__main__":
    unittest.main()
