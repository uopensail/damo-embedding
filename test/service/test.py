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

import damo_embedding


import json
import os
import shutil
import unittest

import damo
import numpy as np
import torch
import time


class DamoServiceTestCase(unittest.TestCase):
    def setUp(self) -> None:
        if os.path.exists("/tmp/data_dir"):
            shutil.rmtree("/tmp/data_dir")
        self.dim = 16
        self.group = 0
        self.gamma = 0.01
        self.eta = 0.0
        self.lambda_ = 0.0
        self.epsilon = 1e-10
        self.configure = {
            "ttl": 86400,
            "dir": "/tmp/data_dir",
            "port": 9275,
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
                        "name": "adagrad",
                        "gamma": self.gamma,
                        "lambda": self.lambda_,
                        "epsilon": self.epsilon,
                    },
                }
            ],
        }
        with open("/tmp/damo-configure.json", "w") as f:
            json.dump(self.configure, f)
        damo_embedding.run_damo_embedding_service("/tmp/damo-configure.json",9275)

    def tearDown(self) -> None:
        damo_embedding.stop_damo_embeding_service()
        return super().tearDown()

    def test(self):
        # in test case, we use torch to test the results
        n = 8
        keys = np.random.randint(1, 10000 + 1, n, dtype=np.int64)
        w = np.zeros(self.dim * n).astype(np.float32)
        damo_embedding.util.pull(self.group, keys, w)
        # print("w", w)
        gds = np.random.random(self.dim * n).astype(np.float32)
        x = torch.tensor(w, dtype=torch.float32, requires_grad=True)
        x.grad = torch.tensor(gds, dtype=torch.float32)
        opt = torch.optim.Adagrad(
            [x],
            lr=self.gamma,
            eps=self.epsilon,
        )

        damo_embedding.util.push(self.group, keys, gds)
        opt.step()
        damo_embedding.util.pull(self.group, keys, w)
        tmp = (w - x.detach().numpy()).astype(np.float32)
        tmp = tmp.reshape((self.dim, n))
        norms = np.linalg.norm(tmp, axis=-1)
        print(norms)


if __name__ == "__main__":
    unittest.main()
