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
import json
import os
import shutil
import unittest

import damo
import numpy as np


class LionTestCase(unittest.TestCase):
    def setUp(self):
        if os.path.exists("/tmp/data_dir"):
            shutil.rmtree("/tmp/data_dir")
        self.dim = 16
        self.group = 0
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.lambda_ = 0.0
        self.epsilon = 1e-8
        self.eta = 0.003
        self.configure = {
            "ttl": 86400,
            "dir": "/tmp/data_dir",
            "embeddings": [
                {
                    "dim": self.dim,
                    "group": self.group,
                    "initializer": {
                        "name": "truncate_normal",
                        "mean": 0.0,
                        "stddev": 1.0,
                    },
                    "optimizer": {
                        "name": "lion",
                        "eta": self.eta,
                        "beta1": self.beta1,
                        "beta2": self.beta2,
                        "lambda": self.lambda_,
                        "epsilon": self.epsilon,
                    },
                }
            ],
        }

        with open("/tmp/damo-configure.json", "w") as f:
            json.dump(self.configure, f)
        self.damo = damo.PyDamo("/tmp/damo-configure.json")

    def tearDown(self):
        pass

    def test(self):
        n = 8
        keys = np.random.randint(1, 10000 + 1, n, dtype=np.int64)
        w = np.zeros(self.dim * n, dtype=np.float32)
        self.damo.pull(self.group, keys, w)
        m = np.zeros(self.dim * n).astype(np.float32)
        gds = np.random.random(self.dim * n).astype(np.float32)
        tmp_mu = np.sign(self.beta1 * m + (1.0 - self.beta1) * gds) + w * self.lambda_
        m = self.beta2 * +(1.0 - self.beta2) * gds
        self.damo.push(self.group, keys, gds)
        new_w = w - self.eta * tmp_mu
        self.damo.pull(self.group, keys, w)

        assert np.linalg.norm(new_w - w) == 0.0


if __name__ == "__main__":
    unittest.main()
