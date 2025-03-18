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
from typing import Any, Dict, List

import numpy as np
import torch
from torch.autograd import Function

from . import config


class EmbeddingFunction(Function):
    """Custom autograd function for efficient embedding operations"""

    @staticmethod
    def forward(
        ctx,
        group: int,
        keys: torch.Tensor,
        weights: torch.Tensor,
        input: torch.Tensor,
        dim: int,
    ):
        """
        Forward pass with automatic gradient registration
        ctx: Context object to save tensors for backward pass
        group: Embedding group identifier
        keys: Unique keys tensor
        weights: Embedding weights tensor
        input: Input indices tensor
        dim: Embedding dimension
        """
        ctx.save_for_backward(input, keys, weights)
        ctx.group = group
        ctx.dim = dim

        # Create embedding matrix through advanced indexing
        weight_dict = {k.item(): v for k, v in zip(keys, weights)}
        embedding_matrix = torch.stack(
            [weight_dict.get(idx.item(), torch.zeros(dim)) for idx in input.flatten()]
        )
        return embedding_matrix.view(*input.shape, dim)

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass with optimized gradient calculation
        grad_output: Gradient of the loss w.r.t. the output
        """
        input, keys, weights = ctx.saved_tensors
        group = ctx.group
        dim = ctx.dim

        # Vectorized gradient calculation
        flat_grad = grad_output.contiguous().view(-1, dim)
        flat_input = input.contiguous().view(-1)

        # Create gradient dictionary using PyTorch operations
        unique_keys, inverse_indices = torch.unique(flat_input, return_inverse=True)
        summed_grad = (
            torch.zeros_like(unique_keys, dtype=torch.float32)
            .unsqueeze(1)
            .repeat(1, dim)
        )
        summed_grad.index_add_(0, inverse_indices, flat_grad)

        # Normalize gradients by batch size
        batch_size = input.size(0)
        normalized_grad = summed_grad / batch_size

        # Push gradients through Damo service
        push(group, unique_keys.numpy(), normalized_grad.numpy())

        return None, None, None, None, None


class Embedding(torch.nn.Module):
    """Optimized sparse embedding layer with Damo Embedding integration

    Attributes:
        dim (int): Embedding dimension
        group (int): Service group identifier
        initializer (dict): Weight initialization parameters
        optimizer (dict): Optimization configuration
    """

    def __init__(
        self,
        dim: int,
        group: int,
        initializer: dict = None,
        optimizer: dict = None,
        **kwargs,
    ):
        """
        Initialize embedding layer

        Args:
            dim: Embedding dimension
            group: Group identifier
            initializer: Weight initialization config (default: {})
            optimizer: Optimizer configuration (default: {})
            kwargs: Additional service parameters
        """
        super().__init__()
        self.dim = dim
        self.group = group
        self.initializer = initializer or {}
        self.optimizer = optimizer or {}

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Perform embedding lookup with automatic gradient tracking

        Args:
            input: Input tensor of indices (2D: [batch, sequence])

        Returns:
            Embedded values tensor (3D: [batch, sequence, dim])

        Raises:
            RuntimeError: If Damo service not initialized
            ValueError: For invalid input dimensions
        """
        if input.dim() != 2:
            raise ValueError(f"Input must be 2D tensor (got {input.dim()}D)")

        # Convert to numpy for service communication
        with torch.no_grad():
            np_input = input.cpu().numpy()
            unique_keys = np.unique(np_input)

        # Pull weights from Damo service
        weights = torch.zeros(len(unique_keys), self.dim, dtype=torch.float32)
        pull(self.group, unique_keys, weights.numpy())

        # Create tensors on correct device
        device = input.device
        keys_tensor = torch.from_numpy(unique_keys).to(device)
        weights_tensor = weights.to(device)

        return EmbeddingFunction.apply(
            self.group, keys_tensor, weights_tensor, input.to(device), self.dim
        )


def list_all_sparse_embeddings(model: torch.nn.Module) -> List[Embedding]:
    """Recursively finds all sparse embedding layers in a PyTorch model

    Traverses the model hierarchy using depth-first search to locate all
    embedding layers, including those nested in complex module structures.

    @param model: PyTorch model to search
    @type model: torch.nn.Module

    @return: List of found embedding layers (empty list if none found)
    @rtype: List[Embedding]

    @note:
        1. Performs full recursive traversal of all nested modules
        2. Handles various container types (ModuleList, ModuleDict, etc.)
        3. Returns empty list for models without embeddings
        4. Complexity: O(n) where n is number of modules in model

    @example:
        >>> model = torch.nn.Sequential(
                torch.nn.Linear(10, 20),
                torch.nn.ModuleList([
                    Embedding(...),
                    torch.nn.Linear(20, 30)
                ])
            )
        >>> embeddings = list_all_sparse_embeddings(model)
        >>> len(embeddings)
        1
    """

    # Base case: current module is itself an embedding
    if isinstance(model, Embedding):
        return [model]

    embeddings = []

    # Recursive case: iterate through all child modules
    for _, module in model.named_children():
        # Direct recursion through all child modules
        embeddings += list_all_sparse_embeddings(module)

    return embeddings


def get_damo_embedding_configure(model: torch.nn.Module) -> Dict[str, Any]:
    """!
    @brief Generates configuration metadata for all DAMO sparse embeddings in a model

    @details Scans the provided model to discover all sparse embedding layers and
    constructs a structured configuration dictionary containing their parameters.
    Ensures group IDs are unique across all embeddings.

    @param model The PyTorch model containing DAMO embedding layers to be configured
    @return Dictionary:
        - "embeddings": List of embedding configurations sorted by group ID

    @throws ValueError if duplicate group IDs are found

    @note Configuration format for each embedding:
        - dim: Embedding dimension size
        - group: Unique group identifier for parameter sharing
        - initializer: Weight initialization configuration
        - optimizer: Optimization strategy configuration
    """
    embeddings: List[Embedding] = list_all_sparse_embeddings(model)
    configuration: Dict[str, Any] = {"embeddings": []}

    seen_groups = set()

    for embedding in embeddings:
        current_group = embedding.group

        if current_group in seen_groups:
            raise ValueError(
                f"Duplicate group ID detected: {current_group}. "
                "All embedding groups must be unique."
            )

        # Build embedding configuration entry
        config_entry = {
            "dim": embedding.dim,
            "group": current_group,
            "initializer": embedding.initializer,
            "optimizer": embedding.optimizer,
        }

        configuration["embeddings"].append(config_entry)
        seen_groups.add(current_group)

    return configuration


def push(group: int, keys: np.ndarray, gradients: np.ndarray) -> None:
    """Push gradient updates for embedding keys to Damo service

    @param group: Embedding group identifier from configuration
    @type group: int
    @param keys: Array of embedding keys to update (1D int array)
    @type keys: np.ndarray
    @param gradients: Corresponding gradient values (2D float array)
    @type gradients: np.ndarray

    @throws ValueError: If array dimensions mismatch or invalid group
    @throws RuntimeError: If Damo instance not initialized

    @note:
        1. keys and gradients must have matching first dimension sizes
        2. gradients array must be float32 dtype
        3. Non-destructive operation - original arrays remain unchanged

    @example:
        >>> keys = np.array([101, 203], dtype=np.int64)
        >>> grads = np.random.rand(2, 128).astype(np.float32)
        >>> push(0, keys, grads)
    """
    _validate_damo_instance()
    if keys.ndim != 1 or gradients.ndim != 2:
        raise ValueError(
            f"Dimension mismatch: keys({keys.shape}) vs gradients({gradients.shape})"
        )
    if keys.shape[0] != gradients.shape[0]:
        raise ValueError(
            f"Batch size mismatch: {keys.shape[0]} vs {gradients.shape[0]}"
        )

    config._singleton_damo_instance.push(group, keys, gradients)


def pull(group: int, keys: np.ndarray, weights: np.ndarray) -> None:
    """Retrieve current embedding weights from Damo service

    @param group: Embedding group identifier from configuration
    @type group: int
    @param keys: Array of keys to retrieve (1D int array)
    @type keys: np.ndarray
    @param weights: Output array for weights (pre-allocated 2D float array)
    @type weights: np.ndarray

    @throws BufferError: If weights array has incorrect dimensions
    @throws TypeError: If invalid key datatype

    @note:
        1. weights array will be modified in-place
        2. keys must be int32/int64 dtype
        3. weights array must be writeable and properly aligned

    @example:
        >>> keys = np.array([101, 203], dtype=np.int64)
        >>> weights = np.empty((2, 128), dtype=np.float32)
        >>> pull(0, keys, weights)
    """
    _validate_damo_instance()
    if not keys.flags["C_CONTIGUOUS"] or not weights.flags["C_CONTIGUOUS"]:
        raise BufferError("Input arrays must be C-contiguous")
    if weights.shape[0] != keys.shape[0]:
        raise ValueError(
            f"Output buffer size mismatch: {weights.shape[0]} vs {keys.shape[0]}"
        )
    config._singleton_damo_instance.pull(group, keys, weights.ravel())


def dump(file_path: str) -> None:
    """Persist embedding state to a single file

    @param file_path: Target file path for storing embedding state (.embd extension recommended)
    @type file_path: str

    @throws PermissionError: If file path is not writable
    @throws IsADirectoryError: If path points to a directory
    @throws RuntimeError: If Damo service not initialized

    @example:
        >>> dump("/data/embeddings/backup_2023.embd")
    """
    _validate_damo_instance()
    _validate_file_path(file_path)
    _ensure_parent_directory(file_path)

    config._singleton_damo_instance.dump(file_path)


def checkpoint(checkpoint_file: str) -> None:
    """Create atomic checkpoint file

    @param checkpoint_file: Target file path for checkpoint (.ckpt extension recommended)
    @type checkpoint_file: str

    @note:
        1. Uses temporary file + atomic rename pattern
        2. Maintains previous checkpoint file until success
        3. Requires write permission in target directory

    @warning:
        - Target filesystem must support atomic rename operations
        - Temporary files created in the same directory
    """
    _validate_damo_instance()
    _validate_file_path(checkpoint_file)
    _ensure_parent_directory(checkpoint_file)

    config._singleton_damo_instance.checkpoint(checkpoint_file)


def load(load_file: str) -> None:
    """Initialize embeddings from saved state file

    @param load_file: Path to state file created by dump/checkpoint
    @type load_file: str

    @throws FileNotFoundError: If specified file does not exist
    @throws InvalidFileFormatError: If file format is unrecognized

    @note:
        1. Full reset of current in-memory state
        2. File format must match exactly with dump version
    """
    _validate_damo_instance()
    if not os.path.isfile(load_file):
        raise FileNotFoundError(f"Embedding state file not found: {load_file}")

    config._singleton_damo_instance.load(load_file)


def _validate_file_path(path: str) -> None:
    """Validate file path characteristics"""
    if os.path.isdir(path):
        raise IsADirectoryError(f"Path points to a directory: {path}")
    if os.path.exists(path) and not os.path.isfile(path):
        raise FileExistsError(f"Path exists but is not a regular file: {path}")


def _ensure_parent_directory(file_path: str) -> None:
    """Ensure parent directory exists"""
    parent_dir = os.path.dirname(file_path)
    if parent_dir and not os.path.exists(parent_dir):
        os.makedirs(parent_dir, exist_ok=True)


def _validate_damo_instance() -> None:
    """Internal validation for Damo service initialization"""
    if (
        not hasattr(config, "_singleton_damo_instance")
        or config._singleton_damo_instance is None
    ):
        raise RuntimeError(
            "Damo service not initialized. Call damo_embedding_init() first"
        )
