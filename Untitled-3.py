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
import json
import struct
import torch
import numpy as np
import shutil
from typing import Tuple, Dict, List
from collections import defaultdict


__all__ = [
    "Embedding",
    "save_model",
    "load_model",
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
        print("Loading From Checkpoint: %s" % path)
        Storage._instance.storage.load_from_checkpoint(path)


class Embedding(torch.nn.Module):
    """embedding module for training."""

    _group = -1

    def __init__(self, dim: int, initializer={}, optimizer={}, **kwargs):
        super(Embedding, self).__init__()
        self.dim = dim
        if "group" in kwargs:
            self.group = kwargs["group"]
            Embedding._group = max(Embedding._group, self.group)
        else:
            # if `group` not in arguments, use default
            Embedding._group += 1
            self.group = Embedding._group

        assert self.group >= 0
        self.kwargs = kwargs
        self.storage = Storage(**kwargs).storage

        # create initializer
        self.init_params = initializer
        init_params = damo.Parameters()
        for k, v in initializer.items():
            init_params.insert(k, v)
        self.initializer = damo.PyInitializer(init_params)

        # create optimizer
        opt_params = damo.Parameters()
        self.opt_params = optimizer
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

        data = input.numpy().astype(np.int64)
        batch_size, width = data.shape
        keys = np.unique(np.concatenate(data)).astype(np.int64)
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


class KeyMapper(torch.nn.Module):
    """hashed key to embedding key."""

    def __init__(self, keys: Dict[int, int]):
        super(KeyMapper, self).__init__()
        self.dict: Dict[int, int] = keys

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """keys lookup, different embeddings have different keymap

        Args:
            input (torch.Tensor): _description_

        Returns:
            torch.Tensor: translate keys to embedding keys
        """
        keys = input.flatten()
        values: List[int] = [self.dict.get(int(keys[i]), 0) for i in range(len(keys))]
        ret = torch.tensor(values, dtype=torch.int64, requires_grad=False)
        return ret.reshape(input.shape)


class DummyEmbedding(torch.nn.Module):
    """embedding module for inference."""

    def __init__(self, keys: Dict[int, int], data: np.ndarray):
        super(DummyEmbedding, self).__init__()
        self.keymapper = KeyMapper(keys)
        self.embedding = torch.nn.Embedding(data.shape[0], data.shape[1], padding_idx=0)
        self.embedding.weight.data = torch.from_numpy(data)
        self.embedding.requires_grad_ = False

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.embedding(self.keymapper(input))


def sparse_to_numpy(
    sparse_path: str, groups: Dict[int, int]
) -> Tuple[Dict[int, np.ndarray], Dict[int, Dict[int, int]]]:
    """read from the sparse file, which dumped by Storage

    Args:
        sparse_path (str): data path
        groups (Dict[int, int]): group and it's dim
    Returns:
        Tuple[Dict[int, np.ndarray], Dict[int, Dict[int, int]]]: data and ids
    """
    group_data, group_index, group_ids, group_dims = {}, {}, {}, {}
    with open(sparse_path, "rb") as f:
        size = struct.unpack("@i", f.read(4))[0]
        groups = struct.unpack(f"@{size}i", f.read(size * 4))
        dims = struct.unpack(f"@{size}i", f.read(size * 4))
        counts = struct.unpack(f"@{size}q", f.read(size * 8))
        for group, dim, count in zip(groups, dims, counts):
            group_dims[group] = dim
            group_data[group] = np.zeros((count + 1, dim), dtype=np.float32)
            group_index[group] = 1
            group_ids[group] = {}

        buffer = f.read(8)
        while buffer:
            key = struct.unpack("@q", buffer)[0]
            group = struct.unpack("@i", f.read(4))[0]
            weight = struct.unpack(
                f"@{group_dims[group]}f", f.read(4 * group_dims[group])
            )
            group_data[group][group_index[group]] = np.array(weight, dtype=np.float32)
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


def save_model_for_training(model: torch.nn.Module, output_dir: str):
    """save model to dir, for training again

    Args:
        model (torch.nn.Module): torch module
        output_dir (str): output dir
    """
    # remove output dir
    output_dir = os.path.join(output_dir, "save_model")
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    checkpoint_dir = os.path.join(output_dir, "checkpoint")
    checkpoint(checkpoint_dir)

    sparse_embedding_params = {}
    original_modules = {}
    for k, v in model.__dict__["_modules"].items():
        original_modules[k] = v
        if isinstance(v, Embedding):
            sparse_embedding_params["e_%s" % k] = {
                "init_params": v.init_params,
                "opt_params": v.opt_params,
                "kwargs": v.kwargs,
                "dim": v.dim,
                "group": v.group,
            }
            model.__dict__["_modules"][k] = torch.nn.Embedding(1, v.dim, padding_idx=0)
        elif isinstance(v, torch.nn.ModuleList):
            modules = torch.nn.ModuleList()
            for i, m in enumerate(v):
                if isinstance(m, Embedding):
                    sparse_embedding_params["l_%s_%d" % (k, i)] = {
                        "init_params": v.init_params,
                        "opt_params": v.opt_params,
                        "kwargs": v.kwargs,
                        "dim": v.dim,
                        "group": v.group,
                    }
                    modules.append(torch.nn.Embedding(1, v.dim, padding_idx=0))
                else:
                    modules.append(m)
            model.__dict__["_modules"][k] = modules
        elif isinstance(v, torch.nn.ModuleDict):
            modules = torch.nn.ModuleDict()
            for n, m in v.items():
                if isinstance(m, Embedding):
                    sparse_embedding_params["d_%s_%s" % (n, k)] = {
                        "init_params": v.init_params,
                        "opt_params": v.opt_params,
                        "kwargs": v.kwargs,
                        "dim": v.dim,
                        "group": v.group,
                    }
                    modules[n] = torch.nn.Embedding(1, v.dim, padding_idx=0)
                else:
                    modules[n] = m
            model.__dict__["_modules"][k] = modules

    torch.save(model, os.path.join(output_dir, "model.pt"))
    json.dump(
        sparse_embedding_params,
        open(os.path.join(output_dir, "sparse_config.json"), "w"),
    )

    # recover
    for k, _ in model.__dict__["_modules"].items():
        model.__dict__["_modules"][k] = original_modules[k]


def load_model(dir: str) -> torch.nn.Module:
    """load model to for training again

    Args:
        dir (str): model directory

    Returns:
        torch.nn.Module: model to train
    """
    sparse_embedding_params = json.load(
        open(os.path.join(dir, "sparse_config.json"), "r")
    )
    model = torch.load(os.path.join(dir, "model.pt"))

    for file in os.listdir(dir):
        if file.startswith("checkpoint"):
            if Storage._instance is None:
                for _, v in sparse_embedding_params.items():
                    Storage(**v["kwargs"])
                    break
            load_from_checkpoint(os.path.join(dir, file))
            Embedding._group = -1
            break
    for k, v in model.__dict__["_modules"].items():
        if isinstance(v, torch.nn.Embedding):
            key = "e_%s" % k
            if key not in sparse_embedding_params:
                continue
            config = sparse_embedding_params[key]
            model.__dict__["_modules"][k] = Embedding(
                config["dim"],
                config["init_params"],
                config["opt_params"],
                group=config["group"],
                **config["kwargs"],
            )
        elif isinstance(v, torch.nn.ModuleList):
            modules = torch.nn.ModuleList()
            for i, m in enumerate(v):
                key = "l_%s_%d" % (k, i)
                if key not in sparse_embedding_params:
                    modules.append(m)
                else:
                    config = sparse_embedding_params[key]
                    modules.append(
                        Embedding(
                            config["dim"],
                            config["init_params"],
                            config["opt_params"],
                            group=config["group"],
                            **config["kwargs"],
                        )
                    )
            model.__dict__["_modules"][k] = modules
        elif isinstance(v, torch.nn.ModuleDict):
            modules = torch.nn.ModuleDict()
            for n, m in v.items():
                key = "d_%s_%s" % (n, k)
                if key not in sparse_embedding_params:
                    modules[n] = m
                else:
                    config = sparse_embedding_params[key]
                    modules[n] = Embedding(
                        config["dim"],
                        config["init_params"],
                        config["opt_params"],
                        group=config["group"],
                        **config["kwargs"],
                    )
            model.__dict__["_modules"][k] = modules
    return model


def save_model_for_inference(model: torch.nn.Module, output_dir: str) -> None:
    """save model to dir for inference

    Args:
        model (torch.nn.Module): torch module
        output_dir (str): output dir
    """
    sparse_path = os.path.join(output_dir, ".sparse.dat")
    Storage.dump(sparse_path)
    groups = {}
    for k, v in model.__dict__["_modules"].items():
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
    for k, v in model.__dict__["_modules"].items():
        original_modules[k] = v
        if isinstance(v, Embedding):
            model.__dict__["_modules"][k] = DummyEmbedding(
                group_ids[v.group], group_data[v.group]
            )
        elif isinstance(v, torch.nn.ModuleList):
            modules = torch.nn.ModuleList()
            for m in v:
                if isinstance(m, Embedding):
                    modules.append(
                        DummyEmbedding(group_ids[m.group], group_data[m.group])
                    )
                else:
                    modules.append(m)
            model.__dict__["_modules"][k] = modules
        elif isinstance(v, torch.nn.ModuleDict):
            modules = torch.nn.ModuleDict()
            for n, m in v.items():
                if isinstance(m, Embedding):
                    modules[n] = DummyEmbedding(group_ids[m.group], group_data[m.group])
                else:
                    modules[n] = m
            model.__dict__["_modules"][k] = modules

    model_scripted = torch.jit.script(model)
    model_scripted.save(os.path.join(output_dir, "model.pt"))

    # recover
    for k, _ in model.__dict__["_modules"].items():
        model.__dict__["_modules"][k] = original_modules[k]

    os.remove(sparse_path)


def save_model(model: torch.nn.Module, output_dir: str, training: bool = True):
    """save mode

    Args:
        model (torch.nn.Module): model
        output_dir (str): model directory
        training (bool, optional): training or inference. Defaults to True.
    """
    if training:
        save_model_for_training(model, output_dir)
    else:
        save_model_for_inference(model, output_dir)
