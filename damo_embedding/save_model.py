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
import copy
import json
import os
import shutil
from typing import Tuple

import torch

from .damo_helper import Embedding, checkpoint


def save_model_for_training(model: torch.nn.Module, output_dir: str) -> None:
    """Save model architecture and sparse embedding configurations for resumable training

    Preserves complete model structure while replacing Damo embeddings with placeholder layers,
    storing original embedding configurations separately for later restoration.

    @param model: Model containing Damo embedding layers
    @type model: torch.nn.Module
    @param output_dir: Root directory for saved artifacts
    @type output_dir: str

    @throws PermissionError: If unable to write to target directory
    @throws RuntimeError: If model restoration fails after saving

    @note:
        1. Maintains original model structure through deep copy
        2. Handles nested ModuleList/ModuleDict recursively
        3. Atomic save operation with temporary directory
        4. Preserves device placement and dtype information

    @example:
        >>> model = create_damo_model()
        >>> save_model_for_training(model, "/saved_models/v1")
        >>> # Later loading:
        >>> model = load_model_for_training("/saved_models/v1")
    """
    try:
        # Create temporary working directory
        temp_dir = os.path.join(output_dir, "_temp_save")
        final_dir = os.path.join(output_dir, "save_model")

        # Clean/create directories atomically
        with AtomicDirectory(final_dir, temp_dir) as work_dir:
            # 1. Save checkpoint
            checkpoint(os.path.join(work_dir, "checkpoint"))

            # 2. Create model copy and replace embeddings
            model_copy, embedding_config = _create_savable_model_copy(model)

            # 3. Save modified model and config
            torch.save(model_copy, os.path.join(work_dir, "model.pt"))
            _save_embedding_config(embedding_config, work_dir)

            # 4. Preserve original model state dict
            torch.save(model.state_dict(), os.path.join(work_dir, "original_state.pth"))

    except Exception as e:
        raise RuntimeError(f"Model save failed: {str(e)}") from e


def _create_savable_model_copy(model: torch.nn.Module) -> Tuple[torch.nn.Module, dict]:
    """Create a safe-to-save model copy with placeholder embeddings"""
    model_copy = copy.deepcopy(model)
    embedding_config = {}

    # Recursive module replacement
    def _replace_embeddings(module, path=""):
        for name, child in module.named_children():
            full_path = f"{path}.{name}" if path else name

            if isinstance(child, Embedding):
                # Store original configuration
                embedding_config[full_path] = {
                    "initializer": child.initializer,
                    "optimizer": child.optimizer,
                    "dim": child.dim,
                    "group": child.group,
                }

                # Create placeholder embedding
                placeholder = torch.nn.Embedding(1, child.dim, padding_idx=0)
                setattr(module, name, placeholder)

            else:
                _replace_embeddings(child, full_path)

    _replace_embeddings(model_copy)
    return model_copy, embedding_config


def _save_embedding_config(config: dict, save_dir: str) -> None:
    """Save embedding configuration with versioning"""
    config_path = os.path.join(save_dir, "sparse_config.json")

    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)


class AtomicDirectory:
    """Context manager for atomic directory operations"""

    def __init__(self, final_path: str, temp_path: str):
        self.final_path = final_path
        self.temp_path = temp_path

    def __enter__(self):
        if os.path.exists(self.temp_path):
            shutil.rmtree(self.temp_path)
        os.makedirs(self.temp_path)
        return self.temp_path

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            if os.path.exists(self.final_path):
                shutil.rmtree(self.final_path)
            os.rename(self.temp_path, self.final_path)
        else:
            shutil.rmtree(self.temp_path)
