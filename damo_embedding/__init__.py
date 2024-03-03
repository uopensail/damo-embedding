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
__version__ = "1.1"
__all__ = [
    "damo_embedding_init",
    "damo_embedding_close",
    "Embedding",
    "save_model",
    "load_model",
    "update_model",
    "damo_set_global_step_control"
]

import json
import os

import damo
import torch

from . import config
from .graph import update_model
from .inference import save_model_for_inference
from .trainning import load_model, save_model_for_training
from .util import (
    Embedding,
    get_damo_embedding_configure,
    run_damo_embedding_service,
    stop_damo_embeding_service,
)


def damo_embedding_init(
    model: torch.nn.Module,
    ttl: int,
    dir: str,
    reload: str = "",
    port: int = 9275,
    mode: str = "",
    run_service = True,
):
    """initial of damo embedding

    Args:
        model (torch.nn.Module): torch model
        ttl (int): key ttl
        dir (str): rocksdb dir
        port (int, optional): server port. Defaults to 9275.
        reload (str, optional): reload from from checkpont path. Defaults to "".
    """
    configure = get_damo_embedding_configure(model)
    configure["port"] = port
    configure["ttl"] = ttl
    configure["dir"] = dir
    if reload != "":
        configure["reload_dir"] = reload
    config_path = f"/tmp/damo-configure-{os.getpid()}.json"
    json.dump(configure, open(config_path, "w"))
    mode = mode.lower()
    if mode == "":
        config.DAMO_INSTANCE = damo.PyDamo(config_path)
    elif mode == "service":
        config.DAMO_SERVICE_ADDRESS = f"http://localhost:{port}"
        if run_service: run_damo_embedding_service(config_path, port)


def damo_embedding_close():
    """close rocksdb"""
    if config.DAMO_INSTANCE is None:
        stop_damo_embeding_service()

def damo_set_global_step_control(train_id:int, step:int):
    config.set_global_step_control(train_id,step)

def save_model(
    model: torch.nn.Module, output_dir: str, training: bool = True, **kwargs
):
    """save mode

    Args:
        model (torch.nn.Module): model
        output_dir (str): model directory
        training (bool, optional): training or inference. Defaults to True.
    """
    if training:
        save_model_for_training(model, output_dir)
    else:
        graph_update = kwargs.get("graph_update", False)
        save_model_for_inference(model, output_dir, graph_update)
