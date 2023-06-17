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
            cls._instance.storage = damo.PyStorage(cls._instance.dir, cls._instance.ttl)
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
            assert 0 <= self.group < 1024
        else:
            Embedding._group += 1
            self.group = Embedding._group
            assert 0 <= self.group < 1024
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

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """embedding lookup

        Args:
            input (torch.Tensor): input values

        Returns:
            torch.Tensor: embedding values (input.shape[0], input.shape[1], self.dim)
        """

        data = input.numpy().astype(np.uint64)
        batch_size, width = data.shape
        keys = np.unique(np.concatenate(data)).astype(np.uint64)
        length = keys.shape[0]
        weights = np.zeros(length * self.dim, dtype=np.float32)
        self.embedding.lookup(keys, weights)
        weights = weights.reshape((length, self.dim))
        weight_dict = {k: v for k, v in zip(keys, weights)}
        values = np.zeros(shape=(batch_size, width, self.dim), dtype=np.float32)

        for i in range(batch_size):
            for j in range(width):
                key = data[i][j]
                # 0 is padding value
                if key != 0:
                    values[i][j] = weight_dict[key]

        def apply_gradients(gradients):
            grad = gradients.numpy()
            grad = grad.reshape((batch_size, width, self.dim))
            grad_dict = defaultdict(lambda: np.zeros(self.dim, dtype=np.float32))
            for i in range(batch_size):
                for j in range(width):
                    key = data[i][j]
                    if key != 0:
                        grad_dict[key] += grad[i][j]

            values = np.zeros(length * self.dim, dtype=np.float32)
            for i in range(length):
                values[i * self.dim : (i + 1) * self.dim] = (
                    grad_dict[keys[i]] / batch_size
                )

            self.embedding.apply_gradients(keys, values)

        ret = torch.from_numpy(values)
        ret.requires_grad_()
        ret.register_hook(apply_gradients)
        return ret
