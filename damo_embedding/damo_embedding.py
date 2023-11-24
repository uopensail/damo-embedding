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
import json
import struct
import torch
import numpy as np
import zmq
import damo
import requests
import shutil
from typing import Tuple, Dict, List
from enum import IntEnum
from collections import defaultdict
import tornado.web
import tornado.ioloop

__all__ = [
    "initialize",
    "close",
    "Embedding",
    "save_model",
    "load_model",
    "checkpoint",
    "load_from_checkpoint",
]


DAMO_EMBEDDING_SERVICE_ADDRESS = "http://localhost:9275"

EMBEDDING_DEFAULT_TTL = 86400 * 30
EMBEDDING_DEFAULT_PATH = "./embeddings"


class DamoEmbeddingServer(tornado.web.Application):
    def __init__(
        self,
        ttl: int = EMBEDDING_DEFAULT_TTL,
        dir: str = EMBEDDING_DEFAULT_PATH,
        **kwargs,
    ):
        self.ttl = ttl
        self.dir = dir
        damo.opendb(self.ttl, self.dir)
        super(DamoEmbeddingServer, self).__init__(**kwargs)


class PullHandler(tornado.web.RequestHandler):
    def post(self, *args, **kwargs):
        group = self.get_body_argument("group")
        keys = np.array(self.get_body_argument("keys"), dtype=np.int64)
        weights = np.array(self.get_body_argument("weights"), dtype=np.float32)

        damo.pull(group, keys, weights)
        self.write("Hello World, My name is 张岩林")


class PushHandler(tornado.web.RequestHandler):
    def post(self, *args, **kwargs):
        group = self.get_body_argument("group")
        keys = np.array(self.get_body_argument("keys"), dtype=np.int64)
        gds = np.array(self.get_body_argument("gds"), dtype=np.float32)
        damo.push(group, keys, gds)
        self.set_status(200)


application = DamoEmbeddingServer(
    [
        (r"/push", PushHandler),
    ]
)

if __name__ == "__main__":
    application.listen(8080)
    tornado.ioloop.IOLoop.instance().start()


def open(dir: str = "./embeddings", ttl: int = 86400 * 30, del_old: bool = False):
    """open rocksdb

    Args:
        dir (str, optional): data dir. Defaults to "./embeddings".
        ttl (int, optional): expire time. Defaults to 86400*30.
        del_old (bool, optional): delete old path. Defaults to False.
    """
    if del_old and os.path.exists(dir):
        shutil.rmtree(dir)
    damo.open(ttl, dir)


def close():
    damo.close()


def pull(group: int, keys: np.ndarray, weights: np.ndarray):
    damo.pull(group, keys, weights)


def dump(path: str):
    """dump for inference

    Args:
        path (str): dump file path
    """
    damo.dump(path)


def load_from_checkpoint(path: str):
    """load from checkpoint

    Args:
        path (str): checkpoint file path
    """
    damo.load(path)


def checkpoint(path: str):
    """do a checkpoint

    Args:
        path (str): checkpoint file path
    """
    damo.checkpoint(path)


class CommandType(IntEnum):
    """define command type"""

    kCreateEmbedding = 1
    kPull = 2
    kPush = 3
    kDump = 4
    kCheckPoint = 5
    kLoadCheckPoint = 6


ZMQ_DEFAULT_ADDRESS = "tcp://localhost:9275"


class ZMQClient(object):
    def __init__(self, address: str = ZMQ_DEFAULT_ADDRESS) -> None:
        self.context = zmq.Context()
        self.address = address
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect(self.address)

    def __call__(self, msg: str) -> dict:
        print(msg, len(msg))
        self.socket.send_string(msg)
        ret = self.socket.recv_string()
        return json.loads(ret)


def zmq_pull(client: ZMQClient, group: int, keys: np.ndarray) -> np.ndarray:
    """pull weight from server

    Args:
        client (ZMQClient): zeromq client
        group (int): embedding group
        keys (np.ndarray): keys to pull

    Returns:
        np.ndarray: weights
    """
    keys = keys.tolist()
    params = {
        "cmd": CommandType.kPull,
        "keys": keys,
        "group": group,
        "n": len(keys),
    }
    ret = client(json.dumps(params))
    status = ret["status"] == 0
    if status:
        return np.array(ret["data"], dtype=np.float32)
    return np.zeros(len(keys), dtype=np.float32)


def zmq_push(client: ZMQClient, group: int, keys: np.ndarray, gds: np.ndarray) -> bool:
    """push gradients to server

    Args:
        client (ZMQClient): zeromq client
        group (int): group
        keys (np.ndarray): key to push
        gds (np.ndarray): gradients

    Returns:
        bool: pus ok or not
    """
    keys = keys.tolist()
    gds = gds.tolist()
    params = {
        "cmd": CommandType.kPull,
        "keys": keys,
        "gds": keys,
        "group": group,
    }
    ret = client(json.dumps(params))
    return ret["status"] == 0


def zmq_create_embedding(
    client: ZMQClient,
    dim: int,
    group: int,
    initializer: Dict[str, any] = {},
    optimizer: Dict[str, any] = {},
    scheduler: Dict[str, any] = {},
) -> bool:
    """_summary_

    Args:
        client (ZMQClient): zeromq client
        dim (int): dim of embedding
        group (int): group of embedding
        initializer (Dict[str, any], optional): initializer params. Defaults to {}.
        optimizer (Dict[str, any], optional): optimizer params. Defaults to {}.
        scheduler (Dict[str, any], optional): scheduler params. Defaults to {}.

    Returns:
        bool: create ok or not
    """
    params = {
        "cmd": CommandType.kCreateEmbedding,
        "dim": dim,
        "group": group,
        "initializer": initializer,
        "optimizer": optimizer,
    }
    if len(scheduler) > 0:
        params["scheduler"] = scheduler
    ret = client(json.dumps(params))
    return ret["status"] == 0


class Embedding(torch.nn.Module):
    """embedding module for training."""

    def __init__(
        self,
        dim: int,
        group: int,
        initializer={},
        optimizer={},
        scheduler={},
        load=False,
    ):
        super(Embedding, self).__init__()
        self.dim = dim
        self.group = group
        self.init_params = initializer
        self.opt_params = optimizer
        self.sch_params = scheduler
        assert self.group >= 0

        self.client = None
        if not load:
            assert zmq_create_embedding(
                self.client, self.dim, self.group, initializer, optimizer, scheduler
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
        weights = zmq_pull(self.client, self.group, keys)
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

            zmq_push(self.client, self.group, keys, values)

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
    damo.load(path)


def checkpoint(path: str):
    """do a checkpoint

    Args:
        path (str): checkpoint file path
    """
    damo.checkpoint(path)


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

    def get_embedding_params(embedding: Embedding):
        return {
            "init_params": embedding.init_params,
            "opt_params": embedding.opt_params,
            "sch_params": embedding.sch_params,
            "dim": embedding.dim,
            "group": embedding.group,
        }

    sparse_embedding_params = {}
    original_modules = {}
    for k, v in model.__dict__["_modules"].items():
        original_modules[k] = v
        if isinstance(v, Embedding):
            sparse_embedding_params["e_%s" % k] = get_embedding_params(v)
            model.__dict__["_modules"][k] = torch.nn.Embedding(1, v.dim, padding_idx=0)
        elif isinstance(v, torch.nn.ModuleList):
            modules = torch.nn.ModuleList()
            for i, m in enumerate(v):
                if isinstance(m, Embedding):
                    sparse_embedding_params["l_%s_%d" % (k, i)] = get_embedding_params(
                        v
                    )
                    modules.append(torch.nn.Embedding(1, v.dim, padding_idx=0))
                else:
                    modules.append(m)
            model.__dict__["_modules"][k] = modules
        elif isinstance(v, torch.nn.ModuleDict):
            modules = torch.nn.ModuleDict()
            for n, m in v.items():
                if isinstance(m, Embedding):
                    sparse_embedding_params["d_%s_%s" % (n, k)] = get_embedding_params(
                        v
                    )
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
    dir = os.path.join(dir, "save_model")
    sparse_embedding_params = json.load(
        open(os.path.join(dir, "sparse_config.json"), "r")
    )
    model = torch.load(os.path.join(dir, "model.pt"))

    def create_embedding(params):
        return Embedding(
            dim=params["dim"],
            group=params["group"],
            initializer=params["init_params"],
            optimizer=params["opt_params"],
            scheduler=params["sch_params"],
            load=True,
        )

    load_from_checkpoint(os.path.join(dir, "checkpoint"))

    for k, v in model.__dict__["_modules"].items():
        if isinstance(v, torch.nn.Embedding):
            key = "e_%s" % k
            if key not in sparse_embedding_params:
                continue
            config = sparse_embedding_params[key]
            model.__dict__["_modules"][k] = create_embedding(config)
        elif isinstance(v, torch.nn.ModuleList):
            modules = torch.nn.ModuleList()
            for i, m in enumerate(v):
                key = "l_%s_%d" % (k, i)
                if key not in sparse_embedding_params:
                    modules.append(m)
                else:
                    config = sparse_embedding_params[key]
                    modules.append(create_embedding(config))
            model.__dict__["_modules"][k] = modules
        elif isinstance(v, torch.nn.ModuleDict):
            modules = torch.nn.ModuleDict()
            for n, m in v.items():
                key = "d_%s_%s" % (n, k)
                if key not in sparse_embedding_params:
                    modules[n] = m
                else:
                    config = sparse_embedding_params[key]
                    modules[n] = create_embedding(config)
            model.__dict__["_modules"][k] = modules
    return model


def save_model_for_inference(model: torch.nn.Module, output_dir: str) -> None:
    """save model to dir for inference

    Args:
        model (torch.nn.Module): torch module
        output_dir (str): output dir
    """
    sparse_path = os.path.join(output_dir, ".sparse.dat")
    damo.dump(sparse_path)
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
