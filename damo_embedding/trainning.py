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

import torch

from .util import Embedding, checkpoint


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
            "init_params": embedding.initializer,
            "opt_params": embedding.optimizer,
            "dim": embedding.dim,
            "group": embedding.group,
        }

    sparse_embedding_params = {}
    original_modules = {}
    for k, v in model.__dict__["_modules"].items():
        original_modules[k] = v
        if isinstance(v, Embedding):
            sparse_embedding_params[k] = get_embedding_params(v)
            model.__dict__["_modules"][k] = torch.nn.Embedding(1, v.dim, padding_idx=0)
        elif isinstance(v, torch.nn.ModuleList):
            modules = torch.nn.ModuleList()
            for i, m in enumerate(v):
                if isinstance(m, Embedding):
                    sparse_embedding_params[f"{k}-{i}"] = get_embedding_params(v)
                    modules.append(torch.nn.Embedding(1, v.dim, padding_idx=0))
                else:
                    modules.append(m)
            model.__dict__["_modules"][k] = modules
        elif isinstance(v, torch.nn.ModuleDict):
            modules = torch.nn.ModuleDict()
            for i, m in v.items():
                if isinstance(m, Embedding):
                    sparse_embedding_params[f"{k}-{i}"] = get_embedding_params(v)
                    modules[i] = torch.nn.Embedding(1, v.dim, padding_idx=0)
                else:
                    modules[i] = m
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
        embedding = Embedding(
            dim=params["dim"],
            initializer=params["init_params"],
            optimizer=params["opt_params"],
        )
        setattr(embedding, "group", params["group"])
        return embedding

    for k, v in model.__dict__["_modules"].items():
        if isinstance(v, torch.nn.Embedding):
            key = k
            if key not in sparse_embedding_params:
                continue
            config = sparse_embedding_params[key]
            model.__dict__["_modules"][k] = create_embedding(config)
        elif isinstance(v, torch.nn.ModuleList):
            modules = torch.nn.ModuleList()
            for i, m in enumerate(v):
                key = f"{k}-{i}"
                if key not in sparse_embedding_params:
                    modules.append(m)
                else:
                    config = sparse_embedding_params[key]
                    modules.append(create_embedding(config))
            model.__dict__["_modules"][k] = modules
        elif isinstance(v, torch.nn.ModuleDict):
            modules = torch.nn.ModuleDict()
            for i, m in v.items():
                key = f"{k}-{i}"
                if key not in sparse_embedding_params:
                    modules[i] = m
                else:
                    config = sparse_embedding_params[key]
                    modules[i] = create_embedding(config)
            model.__dict__["_modules"][k] = modules
    return model
