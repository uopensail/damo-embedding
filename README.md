# Damo-Embedding

[![Deploy to GitHub Pages](https://github.com/uopensail/damo-embedding/actions/workflows/gh-pages.yml/badge.svg)](https://uopensail.github.io/damo-embedding/docs/Intro) [![Build and upload to PyPI](https://github.com/uopensail/damo-embedding/actions/workflows/main.yml/badge.svg?event=release)](https://pypi.org/project/damo-embedding/)
# Quick Install

```shell
pip install damo-embedding
```

# Example

## Embedding
```python


import damo
import torch
import numpy as np
from typing import Union
from collections import defaultdict


class Storage(object):
    """singleton storage class."""

    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = object.__new__(cls)
            cls._instance.dir = kwargs.get("dir", "./embeddings")
            cls._instance.ttl = kwargs.get("ttl", 8640000)
            cls._instance.storage = damo.PyStorage(cls._instance.dir, cls._instance.ttl)
        return cls._instance

    @staticmethod
    def checkpoint(path: str):
        assert Storage._instance is not None
        Storage._instance.storage.checkpoint(path)

    @staticmethod
    def dump(path: str):
        assert Storage._instance is not None
        Storage._instance.storage.dump(path)

    @staticmethod
    def load_from_checkpoint(path: str):
        assert Storage._instance is not None
        Storage._instance.storage.load_from_checkpoint(path)


class Embedding(torch.nn.Module):
    _group = -1

    def __init__(self, dim: int, initializer={}, optimizer={}, group=-1, **kwargs):
        super(Embedding, self).__init__()
        self.dim = dim
        if group != -1:
            self.group = group
            assert 0 <= self.group < 256
        else:
            Embedding._group += 1
            self.group = Embedding._group
            assert 0 <= self.group < 256
        self.storage = Storage(**kwargs).storage

        # create initializer
        init_params = damo.Parameters()
        for k, v in initializer.items():
            init_params.insert(k, v)
        self.initializer = damo.PyInitializer(init_params)

        # create optimizer
        opt_params = damo.Parameters()
        for k, v in optimizer.items():
            opt_params.insert(k, v)
        self.optimizer = damo.PyOptimizer(opt_params)

        self.embedding = damo.PyEmbedding(
            self.storage, self.optimizer, self.initializer, self.dim, self.group
        )

    def forward(self, inputs: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        """embedding lookup

        Args:
            inputs (Union[torch.Tensor, np.ndarray]): input values

        Returns:
            torch.Tensor: embedding values (inputs.shape[0], inputs.shape[1], self.dim)
        """

        data = inputs
        if isinstance(inputs, torch.Tensor):
            data = inputs.numpy().astype(np.uint64)
        elif isinstance(inputs, np.ndarray):
            if data.type != np.uint64:
                data = inputs.astype(np.uint64)

        batch_size, width = data.shape
        keys = np.unique(np.concatenate(data)).astype(np.uint64)
        length = keys.shape[0]
        weights = np.zeros(length * self.dim, dtype=np.float32)
        self.embedding.lookup(keys, weights)
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

            self.embedding.apply_gradients(keys, values)

        ret = torch.from_numpy(values)
        ret.requires_grad_()
        ret.register_hook(apply_gradients)
        return ret


```

## DeepFM

```python

import torch
import torch.nn as nn
import numpy as np
from typing import Union
from embedding import Embedding


class DeepFM(torch.nn.Module):
    def __init__(
        self,
        emb_size: int,
        fea_size: int,
        hid_dims=[256, 128],
        num_classes=1,
        dropout=[0.2, 0.2],
        **kwargs,
    ):
        super(DeepFM, self).__init__()
        self.emb_size = emb_size
        self.fea_size = fea_size

        initializer = {
            "name": "truncate_normal",
            "mean": float(kwargs.get("mean", 0.0)),
            "stddev": float(kwargs.get("stddev", 0.0001)),
        }

        optimizer = {
            "name": "adam",
            "gamma": float(kwargs.get("gamma", 0.001)),
            "beta1": float(kwargs.get("beta1", 0.9)),
            "beta2": float(kwargs.get("beta2", 0.999)),
            "lambda": float(kwargs.get("lambda", 0.0)),
            "epsilon": float(kwargs.get("epsilon", 1e-8)),
        }

        self.w = Embedding(
            1,
            initializer=initializer,
            optimizer=optimizer,
            group=0,
            **kwargs,
        )

        self.v = Embedding(
            self.emb_size,
            initializer=initializer,
            optimizer=optimizer,
            group=1,
            **kwargs,
        )
        self.w0 = torch.zeros(1, dtype=torch.float32, requires_grad=True)
        self.dims = [fea_size * emb_size] + hid_dims

        self.layers = nn.ModuleList()
        for i in range(1, len(self.dims)):
            self.layers.append(nn.Linear(self.dims[i - 1], self.dims[i]))
            self.layers.append(nn.BatchNorm1d(self.dims[i]))
            self.layers.append(nn.BatchNorm1d(self.dims[i]))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Dropout(dropout[i - 1]))
        self.layers.append(nn.Linear(self.dims[-1], num_classes))
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        """forward

        Args:
            inputs (Union[torch.Tensor, np.ndarray]): input tensor

        Returns:
            tensor.Tensor: deepfm forward values
        """
        assert inputs.shape[1] == self.fea_size
        w = self.w.forward(inputs)
        v = self.v.forward(inputs)
        square_of_sum = torch.pow(torch.sum(v, dim=1), 2)
        sum_of_square = torch.sum(v * v, dim=1)
        fm_out = (
            torch.sum((square_of_sum - sum_of_square) * 0.5, dim=1, keepdim=True)
            + torch.sum(w, dim=1)
            + self.w0
        )

        dnn_out = torch.flatten(v, 1)
        for layer in self.layers:
            dnn_out = layer(dnn_out)
        out = fm_out + dnn_out
        out = self.sigmoid(out)
        return out

```

# Document
[Doc Website](https://uopensail.github.io/damo-embedding/docs/Intro)