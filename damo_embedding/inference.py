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
import struct
from typing import Dict, List, Tuple

import numpy as np
import torch

from .graph import update_model_graph
from .util import Embedding, dump


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


def save_model_for_inference(
    model: torch.nn.Module, output_dir: str, graph_update: bool = False
) -> None:
    """save model to dir for inference

    Args:
        model (torch.nn.Module):  torch module
        output_dir (str): output dir
        graph_update (bool, optional): update graph or not
    """
    output_dir = os.path.join(output_dir, "inference")
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    sparse_path = os.path.join(output_dir, "sparse.dat")
    dump(sparse_path)
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
    model_path = os.path.join(output_dir, "model.pt")
    meta_path = os.path.join(output_dir, "meta.json")
    if graph_update:
        update_model_graph(
            model=model_scripted,
            model_path=model_path,
            meta_path=meta_path,
        )
    else:
        model_scripted.save(model_path)
        json.dump({"sparse": 0}, open(meta_path, "w"))

    # recover
    for k, _ in model.__dict__["_modules"].items():
        model.__dict__["_modules"][k] = original_modules[k]

    os.remove(sparse_path)
