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
import torch
import numpy as np
import damo
from collections import defaultdict
from .config import IS_MULTIPROCESSING_TRAINING


class Embedding(torch.nn.Module):
    """embedding module for training."""

    def __init__(
        self,
        dim: int,
        group: int,
        initializer={},
        optimizer={},
        load=False,
    ):
        super(Embedding, self).__init__()
        self.dim = dim
        self.group = group
        self.init_params = initializer
        self.opt_params = optimizer
        assert self.group >= 0
        self.client = damo

        if not load:
            damo.embedding(
                json.dumps(
                    {
                        "dim": self.dim,
                        "group": self.group,
                        "initializer": self.init_params,
                        "optimizer": self.opt_params,
                    }
                )
            )

    def pull(self, keys: np.ndarray):
        pass

    def push(self, keys: np.ndarray, gds: np.ndarray):
        pass

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """embedding lookup

        Args:
            input (torch.Tensor): input values

        Returns:
            torch.Tensor: embedding values (input.shape[0], input.shape[1], self.dim)
        """

        data = input.numpy().astype(np.int64)
        batch_size, width = data.shape
        keys = np.unique(np.concatenate(data)).astype(np.int64)
        length = keys.shape[0]
        weights = np.zeros(self.dim * length, dtype=np.float32)
        damo.pull(self.group, keys, weights)
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

            damo.push(self.group, keys, values)

        ret = torch.from_numpy(values)
        ret.requires_grad_()
        ret.register_hook(apply_gradients)
        return ret
