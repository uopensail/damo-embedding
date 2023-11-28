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


class SGDTestCase(unittest.TestCase):
    def setUp(self):
        if os.path.exists("/tmp/data_dir"):
            shutil.rmtree("/tmp/data_dir")
        self.dim = 16
        self.group = 0
        self.lr = 0.001

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
                        "name": "sgd",
                        "gemma": self.lr,
                        "lambda": 0.0,
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
        keys = np.zeros(n, dtype=np.int64)
        for i in range(n):
            keys[i] = i + 1
        w = np.zeros(self.dim * keys.shape[0], dtype=np.float32)
        gds = np.random.random(self.dim * keys.shape[0]).astype(np.float32)
        self.damo.pull(self.group, keys, w)
        a = w - self.lr * gds
        self.damo.push(self.group, keys, gds)
        self.damo.pull(self.group, keys, w)
        assert np.linalg.norm(a - w) == 0.0


if __name__ == "__main__":
    unittest.main()
