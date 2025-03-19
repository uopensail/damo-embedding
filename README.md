# Damo-Embedding

This project is mainly aimed at the model training scenario of small companies, because small companies may be limited in machine resources, and it is not easy to apply for large memory machines or distributed clusters. In addition, most small companies do not need distributed training when training machine learning/deep learning models. On the one hand, because small companies do not have enough data to train distributed large models. On the other hand, training distributed model is a relatively complex project, with high requirements for engineers, and the cost of machines is also high. However, if stand-alone training is used, Out-Of-Memory (OOM) and Out-Of-Vocabulary (OOV) problems often arise. Damo-Embedding is a project designed to solve these problems.

## Out-Of-Memory(OOM)

When using the machine learning framework (TensorFlow/Pytorch) to train the model, creating a new embedding is usually necessary to specify the dimension and size in advance. And also, their implementations are based on memory. If the embedding size is too large, there will be no enough memory. So why do you need such a large Embedding? Because in some scenarios, especially in search, recommmend or ads scenarios, the number of users and materials is usually very large, and engineers will do some manual cross-features, which will lead to exponential expansion of the number of features.

## Out-Of-Vocabulary(OOV)

In the online training model, some new features often appear, such as new user ids, new material ids, etc., which have never appeared before. This will cause the problem of OOV.

## Solutions

The reason for the OOV problem is mainly because the embedding in the training framework is implemented in the form of an array. Once the feature id is out of range, the problem of OOV will appear. We use [rocksdb](https://rocksdb.org/) to store embedding, which naturally avoids the problems of OOV and OOM, because rocksdb uses KV storage, which is similar to hash table and its capacity is only limited by the local disk.


[![Deploy to GitHub Pages](https://github.com/uopensail/damo-embedding/actions/workflows/gh-pages.yml/badge.svg)](https://uopensail.github.io/damo-embedding/docs/Intro) [![Build and upload to PyPI](https://github.com/uopensail/damo-embedding/actions/workflows/main.yml/badge.svg?event=release)](https://pypi.org/project/damo-embedding/)
# Quick Install

```shell
pip install damo-embedding
```

# Example

## DeepFM

```python
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

import torch
import torch.nn as nn

from damo_embedding import Embedding


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
            "name": "uniform",
        }

        optimizer = {
            "name": "adamw",
            "gamma": float(kwargs.get("gamma", 0.0001)),
            "beta1": float(kwargs.get("beta1", 0.9)),
            "beta2": float(kwargs.get("beta2", 0.999)),
            "lambda": float(kwargs.get("lambda", 0.0)),
            "epsilon": float(kwargs.get("epsilon", 1e-8)),
        }

        self.v = Embedding(
            dim=self.emb_size,
            group=0,
            initializer=initializer,
            optimizer=optimizer,
            **kwargs,
        )
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

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """forward

        Args:
            input (torch.Tensor): input tensor

        Returns:
            tensor.Tensor: deepfm forward values
        """
        assert input.shape[1] == self.fea_size
        v = self.v.forward(input)
        square_of_sum = torch.pow(torch.sum(v, dim=1), 2)
        sum_of_square = torch.sum(v * v, dim=1)
        fm_out = torch.sum((square_of_sum - sum_of_square) * 0.5, dim=1, keepdim=True)

        dnn_out = torch.flatten(v, 1)
        for layer in self.layers:
            dnn_out = layer(dnn_out)
        out = fm_out + dnn_out
        out = self.sigmoid(out)
        return out

```

## Save Model

```python
from damo_embedding import save_model
model = DeepFM(8, 39)

# ... other codes

save_model(model, "train", True)


# save onnx model for inference 
save_model(
    model,
    "eval",
    False,
    dummy_input=torch.randint(low=1, high=1000, size=(100, 39), dtype=torch.int64),
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={'input' : {0 : 'batch_size'},
                        'output' : {0 : 'batch_size'}}
)
```
# Document
[Doc Website](https://uopensail.github.io/damo-embedding/docs/Intro)