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
import torch
import damo
import shutil
from typing import List

from .config import IS_MULTIPROCESSING_TRAINING

if IS_MULTIPROCESSING_TRAINING:
    from .embedding import xx as Embedding
else:
    from .embedding import Embedding as Embedding


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
    damo.checkpoint(checkpoint_dir)

    def get_embedding_params(embedding: Embedding):
        return {
            "init_params": embedding.init_params,
            "opt_params": embedding.opt_params,
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
            load=True,
        )

    damo.load(os.path.join(dir, "checkpoint"))

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
