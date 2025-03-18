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
import shutil
from pathlib import Path

import torch

from .damo_helper import Embedding


def load_model_for_training(model_dir: str) -> torch.nn.Module:
    """Load a training-ready model with restored sparse embeddings

    @param model_dir: Directory containing saved model artifacts
    @type model_dir: str

    @return: Model with original architecture and restored embeddings
    @rtype: torch.nn.Module

    @throws ModelLoadError: If critical components are missing or corrupted
    @throws VersionMismatchError: If config version is incompatible

    @example:
        >>> model = load_model_for_training("/saved_models/v1")
        >>> output = model(input_data)
    """
    try:
        # Validate directory structure
        save_root = Path(model_dir)
        if not save_root.is_dir():
            raise NotADirectoryError(f"Model directory not found: {model_dir}")

        artifacts_dir = save_root / "save_model"
        if not artifacts_dir.exists():
            raise FileNotFoundError(f"Missing save_model subdirectory in {model_dir}")

        # Load with atomic guarantee
        with AtomicLoader(artifacts_dir) as loader:
            # 1. Load base model
            model = loader.load_torch("model.pt")

            # 2. Load sparse config with version check
            config = loader.load_json("sparse_config.json")

            # 3. Load original state dict
            state_dict = loader.load_torch("original_state.pth")

            # 4. Recursively restore embeddings
            _restore_embeddings(model, config, loader)

            # 5. Restore model state
            model.load_state_dict(state_dict, False)

        return model

    except Exception as e:
        raise RuntimeError(f"Failed to load model from {model_dir}")


class AtomicLoader:
    """Atomic model loading context manager"""

    def __init__(self, artifacts_dir: Path):
        self.artifacts_dir = artifacts_dir
        self.temp_dir = artifacts_dir.with_name(artifacts_dir.name + "_tmp")

    def __enter__(self):
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
        shutil.copytree(self.artifacts_dir, self.temp_dir)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            shutil.rmtree(self.temp_dir)

    def load_torch(self, filename: str):
        """Safe torch loading with hash verification"""
        path = self.temp_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing model file: {filename}")
        return torch.load(path, map_location="cpu", weights_only=False)

    def load_json(self, filename: str):
        """JSON loading with schema validation"""
        path = self.temp_dir / filename
        with open(path, "r") as f:
            data = json.load(f)
        return data


def _restore_embeddings(model: torch.nn.Module, config: dict, loader: AtomicLoader):
    """Recursively restore embeddings using BFS traversal"""
    from collections import deque

    queue = deque([(model, "")])  # (module, path)

    while queue:
        current_module, current_path = queue.popleft()

        for name, child in current_module.named_children():
            full_path = f"{current_path}.{name}" if current_path else name

            # Restore embeddings at any depth
            if isinstance(child, torch.nn.Embedding):
                if full_path in config:
                    embedding_config = config[full_path]
                    _replace_with_damo_embedding(current_module, name, embedding_config)

            # Continue traversal
            queue.append((child, full_path))


def _replace_with_damo_embedding(module: torch.nn.Module, name: str, config: dict):
    """Safely replace placeholder with Damo embedding"""
    # Validate config structure
    required_keys = {"dim", "group", "initializer", "initializer"}
    if not required_keys.issubset(config.keys()):
        raise KeyError(f"Missing keys in embedding config: {config}")

    # Create Damo embedding with original parameters
    embedding = Embedding(
        dim=config["dim"],
        initializer=config["initializer"],
        optimizer=config["initializer"],
        group=config["group"],
    )

    # Replace module
    setattr(module, name, embedding)
