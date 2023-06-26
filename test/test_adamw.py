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
import numpy as np
import torch
import damo
import shutil
import os


class AdamWTestCase(unittest.TestCase):
    def setUp(self) -> None:
        if os.path.exists("/tmp/data_dir"):
            shutil.rmtree("/tmp/data_dir")
        self.storage = damo.PyStorage("/tmp/data_dir", 86400)
        init_params = damo.Parameters()
        init_params.insert("name", "truncate_normal")
        init_params.insert("mean", 0.0)
        init_params.insert("stddev", 1.0)
        self.initializer = damo.PyInitializer(init_params)
        optm_params = damo.Parameters()
        optm_params.insert("name", "adamw")
        self.gamma = 0.001
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.lambda_ = 0.01
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
            self.storage, self.optimizer, self.initializer, self.dim, group
        )

    def test(self):
        # in test case, we use torch to test the results
        n = 8
        keys = np.random.randint(1, 10000 + 1, n, dtype=np.int64)

        w = np.zeros(self.dim * n).astype(np.float32)
        self.embedding.lookup(keys, w)
        gds = np.random.random(self.dim * n).astype(np.float32)
        x = torch.tensor(w, dtype=torch.float32, requires_grad=True)
        x.grad = torch.tensor(gds, dtype=torch.float32)

        y = torch.tensor(w, dtype=torch.float32, requires_grad=True)
        y.grad = torch.tensor(gds, dtype=torch.float32)

        opt1 = torch.optim.AdamW(
            [x],
            lr=self.gamma,
            betas=(self.beta1, self.beta2),
            weight_decay=self.lambda_,
            eps=self.epsilon,
        )

        # opt2 = torch.optim.AdamW(
        #     [y],
        #     lr=self.gamma,
        #     betas=(self.beta1, self.beta2),
        #     weight_decay=0.0,
        #     eps=self.epsilon,
        # )

        self.embedding.apply_gradients(keys, gds)
        opt1.step()
        # opt2.step()

        x0 = x.detach().numpy()
        # y0 = y.detach().numpy()

        # the diffenence between y0 and x0 is - w * self.gamma * self.lambda_
        # print(y0 - x0 - w * self.gamma * self.lambda_)

        self.embedding.lookup(keys, w)
        tmp = (w - x.detach().numpy()).astype(np.float32)
        tmp = tmp.reshape((self.dim, n))
        norms = np.linalg.norm(tmp, axis=-1)
        print(norms)


if __name__ == "__main__":
    unittest.main()
