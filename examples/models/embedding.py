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
import os
import damo
import struct
import torch
import numpy as np
from typing import Tuple, Dict, List
from collections import defaultdict

__all__ = [
    "Embedding",
    "save_model",
    "checkpoint",
    "load_from_checkpoint",
]


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
    """embedding module for training."""

    _group = -1

    def __init__(self, dim: int, initializer={}, optimizer={}, **kwargs):
        super(Embedding, self).__init__()
        self.dim = dim
        Embedding._group += 1
        self.group = Embedding._group
        assert self.group >= 0
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
            inputs (torch.Tensor): input values

        Returns:
            torch.Tensor: embedding values (input.shape[0], input.shape[1], self.dim)
        """

        data = input.numpy().astype(np.int64)
        batch_size, width = data.shape
        keys = np.unique(np.concatenate(data)).astype(np.int64)
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


class KeyMapper(torch.nn.Module):
    """hashed key to embedding key."""

    def __init__(self, keys: Dict[int, int]):
        super(KeyMapper, self).__init__()
        self.dict: Dict[int, int] = keys

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        keys = input.flatten()
        values: List[int] = [self.dict.get(
            int(keys[i]), 0) for i in range(len(keys))]
        ret = torch.tensor(values, dtype=torch.int64, requires_grad=False)
        return ret.reshape(input.shape)


class DummyEmbedding(torch.nn.Module):
    """embedding module for inference."""

    def __init__(self, keys: Dict[int, int], data: np.ndarray):
        super(DummyEmbedding, self).__init__()
        self.keymapper = KeyMapper(keys)
        self.embedding = torch.nn.Embedding(data.shape[0],
                                            data.shape[1], padding_idx=0)
        self.embedding.weight.data = torch.from_numpy(data)
        self.embedding.requires_grad_ = False

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.embedding(self.keymapper(input))


def sparse_to_numpy(sparse_path: str, groups: Dict[int, int]) -> Tuple[
        Dict[int, np.ndarray], Dict[int, Dict[int, int]]]:
    """read from the sparse file, which dumped by Storage

    Args:
        sparse_path (str): data path
        groups (Dict[int, int]): group and it's dim
    Returns:
        Tuple[Dict[int, np.ndarray], Dict[int, Dict[int, int]]]: data and ids
    """
    group_data, group_index, group_ids = {}, {}, {}
    with open(sparse_path, "rb") as f:
        size = struct.unpack("@i", f.read(4))[0]
        groups = struct.unpack(f"@{size}i", f.read(size * 4))

        dims = struct.unpack(f"@{size}i", f.read(size * 4))
        counts = struct.unpack(f"@{size}q", f.read(size * 8))

        for i in range(size):
            group, dim, count = groups[i], dims[i], counts[i]
            group_data[group] = np.zeros((count + 1, dim), dtype=np.float32)
            group_index[group] = 1
            group_ids[group] = {}

        buffer = f.read(8)
        while buffer:
            key = struct.unpack("@q", buffer)[0]
            group = struct.unpack("@i", f.read(4))[0]
            weight = struct.unpack(
                f"@{dims[group]}f", f.read(4 * dims[group]))
            group_data[group][group_index[group]] = np.array(
                weight, dtype=np.float32)
            group_ids[group][key] = group_index[group]
            group_index[group] += 1
            buffer = f.read(8)
    return group_data, group_ids


def load_from_checkpoint(path: str):
    """load from checkpoint

    Args:
        path (str): checkpoint file path
    """
    Storage.load_from_checkpoint(path)


def checkpoint(path: str):
    """do a checkpoint

    Args:
        path (str): checkpoint file path
    """
    Storage.checkpoint(path)


def save_model(model: torch.nn.Module, output_dir: str) -> None:
    """save mode to dir

    Args:
        model (torch.nn.Module): torch module
        output_dir (str): output dir
    """
    sparse_path = os.path.join(output_dir, "sparse.dat")
    Storage.dump(sparse_path)
    groups = {}
    for k, v in model.__dict__['_modules'].items():
        if isinstance(v, Embedding):
            groups[v.group] = v.dim
        elif isinstance(v, torch.nn.ModuleList):
            for m in v:
                if isinstance(m, Embedding):
                    groups[m.group] = m.dim
        elif isinstance(v, torch.nn.ModuleDict):
            for _, m in v.items():
                if isinstance(m, Embedding):
                    groups[m.group] = m.dim

    group_data, group_ids = sparse_to_numpy(sparse_path, groups)
    # change modules for saving
    original_modules = {}
    for k, v in model.__dict__['_modules'].items():
        original_modules[k] = v
        if isinstance(v, Embedding):
            model.__dict__['_modules'][k] = DummyEmbedding(group_ids[v.group],
                                                           group_data[v.group])
        elif isinstance(v, torch.nn.ModuleList):
            modules = torch.nn.ModuleList()
            for m in v:
                if isinstance(m, Embedding):
                    modules.append(DummyEmbedding(group_ids[m.group],
                                                  group_data[m.group]))
                else:
                    modules.append(m)
            model.__dict__['_modules'][k] = modules
        elif isinstance(v, torch.nn.ModuleDict):
            modules = torch.nn.ModuleDict()
            for n, m in v.items():
                if isinstance(m, Embedding):
                    modules[n] = DummyEmbedding(group_ids[m.group],
                                                group_data[m.group])
                else:
                    modules[n] = m
            model.__dict__['_modules'][k] = modules

    model_scripted = torch.jit.script(model)
    model_scripted.save(os.path.join(output_dir, "model.pt"))

    # recover
    for k, _ in model.__dict__['_modules'].items():
        model.__dict__['_modules'][k] = original_modules[k]

    os.remove(sparse_path)
