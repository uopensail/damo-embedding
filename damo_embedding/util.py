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
import struct
import subprocess
from collections import defaultdict
from multiprocessing import Process
from typing import List

import numpy as np
import requests
import torch

from .config import DAMO_INSTANCE, DAMO_SERVICE_ADDRESS


class Embedding(torch.nn.Module):
    """embedding module for training."""

    def __init__(
        self,
        dim: int,
        initializer={},
        optimizer={},
    ):
        super(Embedding, self).__init__()
        self.dim = dim
        self.initializer = initializer
        self.optimizer = optimizer
        self.group = -1

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
        pull(self.group, keys, weights)
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

            push(self.group, keys, values)

        ret = torch.from_numpy(values)
        ret.requires_grad_()
        ret.register_hook(apply_gradients)
        return ret


def list_all_sparse_embeddings(model: torch.nn.Module) -> List[Embedding]:
    """list all sparse embeddings

    Args:
        model (torch.nn.Module): torch module

    Returns:
        List[Embedding]: all embeddings list
    """
    embeddings = []

    for k, v in model.__dict__["_modules"].items():
        if isinstance(v, Embedding):
            embeddings.append((k, v))
        elif isinstance(v, torch.nn.ModuleList):
            for i, m in enumerate(v):
                if isinstance(m, Embedding):
                    embeddings.append((f"{k}-{i}", m))
        elif isinstance(v, torch.nn.ModuleDict):
            for i, m in v.items():
                if isinstance(m, Embedding):
                    embeddings.append((f"{k}-{i}", m))
    embeddings.sort(key=lambda x: x[0])
    return list(map(lambda x: x[1], embeddings))


def get_damo_embedding_configure(model: torch.nn.Module) -> dict:
    """get damo embedding configure for damo service or damo.PyDamo

    Args:
        model (torch.nn.Module): torch model

    Returns:
        dict: configure dict
    """
    embeddings = list_all_sparse_embeddings(model)
    configure = {"embeddings": []}
    for i, embedding in enumerate(embeddings):
        group = i
        if hasattr(embedding, "group"):
            group = getattr(embedding, "group")
        else:
            setattr(embedding, "group", i)
        configure["embeddings"].append(
            {
                "dim": embedding.dim,
                "group": group,
                "initializer": embedding.optimizer,
                "optimizer": embedding.optimizer,
            }
        )
    return configure


def damo_embedding_service(server: str, configure: str):
    """damo embedding service

    Args:
        server (str): server bin file path
        configure (str): configure path
    """
    subprocess.run(f"bin/damo-server -c {configure}")


def run_damo_embedding_service(server: str, configure: str):
    """run damo embedding service as daemon processor

    Args:
        server (str): server bin file path
        configure (str): configure path
    """
    p = Process(target=damo_embedding_service, args=(server, configure))
    p.daemon = True
    p.start()


def stop_damo_embeding_service():
    global DAMO_SERVICE_ADDRESS
    requests.get(f"{DAMO_SERVICE_ADDRESS}/stop")


def push(group: int, keys: np.ndarray, gradients: np.ndarray):
    """push gradients to damo embedding

    Args:
        group (int): embedding group
        keys (np.ndarray): push keys
        gradients (np.ndarray): gradients of keys
    """
    global DAMO_INSTANCE, DAMO_SERVICE_ADDRESS
    if DAMO_INSTANCE is not None:
        DAMO_INSTANCE.pull(group, keys, gradients)
    else:
        buffer = struct.pack(
            f"@ii{keys.shape[0]}q{gradients.shape[0]}f",
            group,
            keys.shape[0],
            *keys,
            *gradients,
        )
        headers = {
            "Content-Type": "application/octet-stream",
        }

        requests.post(f"{DAMO_SERVICE_ADDRESS}/push", buffer, headers=headers)


def pull(group: int, keys: np.ndarray, weights: np.ndarray):
    """pull weights from damo embedding

    Args:
        group (int): embedding group
        keys (np.ndarray): pull keys
        weights (np.ndarray): keys weights
    """
    global DAMO_INSTANCE, DAMO_SERVICE_ADDRESS
    if DAMO_INSTANCE is not None:
        DAMO_INSTANCE.pull(group, keys, weights)
    else:
        buffer = struct.pack(f"@ii{keys.shape[0]}q", group, keys.shape[0], *keys)
        headers = {
            "Content-Type": "application/octet-stream",
        }

        r = requests.post(f"{DAMO_SERVICE_ADDRESS}/pull", buffer, headers=headers)
        tmp = np.frombuffer(r.raw, dtype=np.float32)
        assert weights.shape[0] == tmp.shape[0]
        weights[:] = tmp[:]


def dump(dir: str):
    """dump spare embedding

    Args:
        dir (str): spare embedding path
    """
    global DAMO_INSTANCE, DAMO_SERVICE_ADDRESS
    if DAMO_INSTANCE is not None:
        DAMO_INSTANCE.dump(dir)
    else:
        headers = {
            "Content-Type": "text/plain",
        }

        requests.post(f"{DAMO_SERVICE_ADDRESS}/dump", data=dir, headers=headers)


def checkpoint(dir: str):
    """do checkpoint

    Args:
        dir (str): checkpoint path
    """
    global DAMO_INSTANCE, DAMO_SERVICE_ADDRESS
    if DAMO_INSTANCE is not None:
        DAMO_INSTANCE.checkpoint(dir)
    else:
        headers = {
            "Content-Type": "text/plain",
        }

        requests.post(f"{DAMO_SERVICE_ADDRESS}/checkpoint", data=dir, headers=headers)
