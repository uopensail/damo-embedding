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
__version__ = "1.2"
__all__ = [
    "Embedding",
    "damo_embedding_init",
    "save_model",
    "load_model",
]

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Optional

import damo
import torch

from . import config as cf
from .damo_helper import Embedding, get_damo_embedding_configure
from .inference import save_model_for_inference
from .load_model import load_model_for_training as load_model
from .save_model import save_model_for_training


def damo_embedding_init(
    model: Optional[torch.nn.Module], workdir: str, ttl: int, **kwargs
) -> None:
    """!
    @brief Initialize DAMO embedding system with enhanced safety and reliability

    @details Features:
    - Thread-safe singleton initialization
    - Automatic model loading fallback
    - Secure temp file handling
    - Comprehensive error reporting

    @param model Target PyTorch model (can be None if model_dir provided)
    @param workdir Working directory for DAMO storage
    @param ttl Time-to-live for cached embeddings (seconds)
    @param kwargs Additional options:
        - model_dir: Fallback model directory path

    @throws FileNotFoundError If config file is missing
    @throws RuntimeError For file operation failures
    @throws ValueError For initialization failures
    """

    # 1. Model Loading Logic
    if model is None:
        model_dir = kwargs.get("model_dir")
        if not model_dir:
            raise ValueError("Must provide either model or model_dir")
        try:
            model = load_model_for_training(model_dir=model_dir)
        except Exception as e:
            raise RuntimeError(f"Model loading failed: {str(e)}") from e

    # 2. Configuration Generation
    try:
        config = get_damo_embedding_configure(model)
        config.update({"dir": os.path.abspath(workdir), "ttl": ttl})
    except Exception as e:
        raise RuntimeError("Configuration generation failed") from e

    # 3. Temp File Handling
    config_file = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".json",
            prefix="damo_config_",
            delete=False,
            dir=os.path.dirname(workdir),  # Same partition for atomic writes
        ) as temp_file:
            json.dump(config, temp_file, indent=2)
            config_file = temp_file.name
    except (IOError, PermissionError) as e:
        raise RuntimeError(f"Config write failed: {str(e)}") from e
    except json.JSONEncodeError as e:
        raise ValueError(f"Invalid config format: {str(e)}") from e

    # 4. Singleton Initialization with Double-Checked Locking
    if cf._singleton_damo_instance is None:
        with cf._singleton_lock:
            if cf._singleton_damo_instance is None:
                try:
                    # Validate config file integrity
                    if not os.path.isfile(config_file):
                        raise FileNotFoundError(f"Missing config: {config_file}")
                    if os.path.getsize(config_file) == 0:
                        raise ValueError("Empty config file")

                    # Core initialization
                    cf._singleton_damo_instance = damo.Damo(
                        config_file=config_file,
                    )

                except Exception as e:
                    raise ValueError(f"DAMO init error: {str(e)}") from e
                finally:
                    # Cleanup in all cases
                    if config_file and os.path.exists(config_file):
                        try:
                            os.unlink(config_file)
                        except OSError as e:
                            print(f"Temp cleanup warning: {str(e)}")
    else:
        # Handle duplicate initialization
        print("DAMO already initialized, skipping...")


def save_model(
    model: torch.nn.Module, output_dir: str, training: bool = True, **kwargs: Any
) -> None:
    """!
    @brief Save a PyTorch model in training or inference format

    @details This function provides a unified interface for saving models in different modes:
    - Training mode: Preserves full model structure and training capabilities
    - Inference mode: Optimizes model for production deployment

    @param model The PyTorch model to save
    @param output_dir Target directory path for model artifacts
    @param training Save mode selector (True for training, False for inference)
    @param kwargs Additional parameters forwarded to underlying save functions:
        - For training: See save_model_for_training documentation
        - For inference: See save_model_for_inference documentation

    @throws ValueError If input validation fails
    @throws IOError If directory creation or file writing fails
    @throws RuntimeError For TorchScript conversion errors in inference mode

    @example
    ```python
    # Save training checkpoint
    save_model(model, "checkpoints/", training=True)

    # Export production model
    save_model(model, "deploy/", training=False)
    ```
    """
    # Parameter validation
    if not isinstance(model, torch.nn.Module):
        raise ValueError("Invalid model type: expected torch.nn.Module")

    if not output_dir:
        raise ValueError("Output directory path cannot be empty")

    # Convert to Path object and ensure directory exists
    save_path = Path(output_dir).resolve()
    try:
        save_path.mkdir(parents=True, exist_ok=True)
    except PermissionError as e:
        raise IOError(f"Permission denied creating directory: {save_path}") from e

    # Dispatch to appropriate save function
    try:
        if training:
            save_model_for_training(model=model, output_dir=str(save_path), **kwargs)
        else:
            save_model_for_inference(model=model, output_dir=str(save_path), **kwargs)
    except Exception as e:
        context = "training" if training else "inference"
        raise RuntimeError(f"Failed to save {context} model") from e

    # Optional: Verify output files
    if not any(save_path.iterdir()):
        raise RuntimeError("No model files generated in target directory")
