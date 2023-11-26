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

from .config import *
import torch
from .inference import save_model_for_inference
from .trainning import save_model_for_training, load_model, list_all_sparse_embeddings
import shutil
import damo
import os
import json

GLOBAL_DAMO_INSTANCE = None


def damo_embedding_init(
    model: torch.Module,
    ttl: int = EMBEDDING_DEFAULT_TTL,
    dir: str = EMBEDDING_DEFAULT_PATH,
    del_old: bool = False,
    method: str = EMBEDDING_DEFAULT_METHOD,
    recover: bool = True,
):
    global GLOBAL_DAMO_INSTANCE
    embeddings = list_all_sparse_embeddings(model=model)
    configure = {"ttl": ttl, "dir": dir, "embeddings": []}
    for i, embedding in enumerate(embeddings):
        setattr(embedding, "group", i)
        configure["embeddings"].append(
            {
                "dim": embedding.dim,
                "group": embedding.group,
                "initializer": embedding.optimizer,
                "optimizer": embedding.optimizer,
            }
        )
    config_path = f"/tmp/damo-configure-{os.getpid()}.json"
    json.dump(configure, open(config_path, "w"))
    GLOBAL_DAMO_INSTANCE = damo.PyDamo(config_path)


def damo_embedding_close():
    """close rocksdb"""
    damo.close()


def set_training_status(is_multiprocessing_training: bool):
    global IS_MULTIPROCESSING_TRAINING
    IS_MULTIPROCESSING_TRAINING = is_multiprocessing_training


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
