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
