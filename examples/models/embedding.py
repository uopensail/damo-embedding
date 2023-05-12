#!/usr/bin/python
# -*- coding: UTF-8 -*-
#
# `Damo-Embedding` - 'c++ tool for sparse parameter server'
# Copyright(C) 2019 - present timepi <timepi123@gmail.com>
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
import torch
import numpy as np
from typing import Union
from collections import defaultdict


class Storage(object):
    """singleton storage class."""

    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = object.__new__(cls)
            cls._instance.dir = kwargs.get("dir", "./embeddings")
            cls._instance.ttl = kwargs.get("ttl", 8640000)
            cls._instance.storage = damo.PyStorage(
                cls._instance.dir, cls._instance.ttl)
        return cls._instance

    @staticmethod
    def checkpoint(path: str):
        assert Storage._instance is not None
        Storage._instance.storage.checkpoint(path)

    @staticmethod
    def dump(path: str):
        assert Storage._instance is not None
        Storage._instance.storage.dump(path)

    @staticmethod
    def load_from_checkpoint(path: str):
        assert Storage._instance is not None
        Storage._instance.storage.load_from_checkpoint(path)


class Embedding(torch.nn.Module):
    _group = -1

    def __init__(self, dim: int, initializer={}, optimizer={}, group=-1, **kwargs):
        super(Embedding, self).__init__()
        self.dim = dim
        if group != -1:
            self.group = group
            assert 0 <= self.group < 256
        else:
            Embedding._group += 1
            self.group = Embedding._group
            assert 0 <= self.group < 256
        self.storage = Storage(**kwargs).storage

        # create initializer
        init_params = damo.Parameters()
        for k, v in initializer.items():
            init_params.insert(k, v)
        self.initializer = damo.PyInitializer(init_params)

        # create optimizer
        opt_params = damo.Parameters()
        for k, v in optimizer.items():
            opt_params.insert(k, v)
        self.optimizer = damo.PyOptimizer(opt_params)

        self.embedding = damo.PyEmbedding(
            self.storage, self.optimizer, self.initializer, self.dim, self.group
        )

    def forward(self, inputs: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        """embedding lookup

        Args:
            inputs (Union[torch.Tensor, np.ndarray]): input values

        Returns:
            torch.Tensor: embedding values (inputs.shape[0], inputs.shape[1], self.dim)
        """

        data = inputs
        if isinstance(inputs, torch.Tensor):
            data = inputs.numpy().astype(np.uint64)
        elif isinstance(inputs, np.ndarray):
            if data.type != np.uint64:
                data = inputs.astype(np.uint64)

        batch_size, width = data.shape
        keys = np.unique(np.concatenate(data)).astype(np.uint64)
        length = keys.shape[0]
        weights = np.zeros(length * self.dim, dtype=np.float32)
        self.embedding.lookup(keys, weights)
        weights = weights.reshape((length, self.dim))
        weight_dict = {k: v for k, v in zip(keys, weights)}
        values = np.zeros(
            shape=(batch_size, width, self.dim), dtype=np.float32)

        for i in range(batch_size):
            for j in range(width):
                key = data[i][j]
                # 0 is padding value
                if key != 0:
                    values[i][j] = weight_dict[key]

        def apply_gradients(gradients):
            grad = gradients.numpy()
            grad = grad.reshape((batch_size, width, self.dim))
            grad_dict = defaultdict(
                lambda: np.zeros(self.dim, dtype=np.float32))
            for i in range(batch_size):
                for j in range(width):
                    key = data[i][j]
                    if key != 0:
                        grad_dict[key] += grad[i][j]

            values = np.zeros(length * self.dim, dtype=np.float32)
            for i in range(length):
                values[i * self.dim: (i + 1) * self.dim] = (
                    grad_dict[keys[i]] / batch_size
                )

            self.embedding.apply_gradients(keys, values)

        ret = torch.from_numpy(values)
        ret.requires_grad_()
        ret.register_hook(apply_gradients)
        return ret
