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
from .trainning import save_model_for_training, load_model
import shutil
import damo
import os


def damo_embedding_init(
    ttl: int = EMBEDDING_DEFAULT_TTL,
    dir: str = EMBEDDING_DEFAULT_PATH,
    del_old: bool = False,
):
    """open rocksdb

    Args:
        dir (str, optional): data dir. Defaults to "./embeddings".
        ttl (int, optional): expire time. Defaults to 86400*30.
        del_old (bool, optional): delete old path. Defaults to False.
    """
    if del_old and os.path.exists(dir):
        shutil.rmtree(dir)
    damo.open(ttl, dir)


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
