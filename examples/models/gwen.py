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

from typing import List

import torch
import torch.nn as nn

from damo_embedding import Embedding


class GroupWiseEmbeddingNetwork(torch.nn.Module):
    def __init__(
        self,
        emb_sizes: List[int],
        hid_dims=[256, 128],
        num_classes=1,
        dropout=[0.2, 0.2],
        **kwargs,
    ):
        super(GroupWiseEmbeddingNetwork, self).__init__()
        self.emb_sizes = emb_sizes
        self.groups: int = len(self.emb_sizes)
        initializer = {
            "name": "truncate_normal",
            "mean": float(kwargs.get("mean", 0.0)),
            "stddev": float(kwargs.get("stddev", 0.0001)),
        }

        optimizer = {
            "name": "adam",
            "gamma": float(kwargs.get("gamma", 1e-3)),
            "beta1": float(kwargs.get("beta1", 0.9)),
            "beta2": float(kwargs.get("beta2", 0.999)),
            "lambda": float(kwargs.get("lambda", 0.0)),
            "epsilon": float(kwargs.get("epsilon", 1e-8)),
        }

        self.embeddings = nn.ModuleList()
        for emb_size in emb_sizes:
            embedding = Embedding(
                emb_size,
                initializer=initializer,
                optimizer=optimizer,
                **kwargs,
            )
            self.embeddings.append(embedding)

        self.dims = [sum(emb_sizes)] + hid_dims
        self.layers = nn.ModuleList()
        for i in range(1, len(self.dims)):
            self.layers.append(nn.Linear(self.dims[i - 1], self.dims[i]))
            self.layers.append(nn.BatchNorm1d(self.dims[i]))
            self.layers.append(nn.BatchNorm1d(self.dims[i]))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Dropout(dropout[i - 1]))
        self.layers.append(nn.Linear(self.dims[-1], num_classes))
        self.sigmoid = nn.Sigmoid()

    def forward(self, input: torch.Tensor):
        batch_size, groups = input.shape
        assert groups == self.groups
        weights = []
        for i, embedding in enumerate(self.embeddings):
            w = embedding.forward(input[:][:, i].reshape(batch_size, 1))
            weights.append(torch.sum(w, dim=1))
        dnn_out = torch.concat(weights, dim=1)
        for layer in self.layers:
            dnn_out = layer(dnn_out)
        out = self.sigmoid(dnn_out)
        return out
