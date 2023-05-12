#!/usr/bin/python
# -*- coding: UTF-8 -*-
#
# `Damo-Embedding` - 'c++ tool for sparse parameter server'
# Copyright(C) 2019 - present timepi <timepi123@gmail.com>
#
# This file is part of `Damo-Embedding`.
#
# `Damo-Embedding` is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# `Damo-Embedding` is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY
# without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with `Damo-Embedding`.  If not, see < http: # www.gnu.org/licenses/>.
#

# dataset: https://www.kaggle.com/datasets/mrkmakr/criteo-dataset

import mmh3
import numpy as np
import pandas as pd
import torch
import torch.utils.data as Data
from typing import Tuple
from sklearn.model_selection import train_test_split


def dense_value_func(name, value):
    if np.isnan(value):
        return mmh3.hash64(f"{name}-NAN", signed=False)[0]
    return mmh3.hash64(f"{name}-{int(value*1e7)}", signed=False)[0]


def sparse_value_func(name: str, value):
    return mmh3.hash64(f"{name}-{value}", signed=False)[0]


def process(train_path: str) -> Tuple[Data.DataLoader, Data.DataLoader]:
    """hash all the features to uint64

    Args:
        train_path (str): train data path

    Returns:
        Tuple[Data.DataLoader, Data.DataLoader]: train and test loaders
    """
    sparse_cols = [f"C-{i}" for i in range(1, 27)]
    dense_cols = [f"I-{i}" for i in range(1, 14)]
    train_cols = ["label"] + dense_cols + sparse_cols
    feature_cols = dense_cols + sparse_cols
    train_data = pd.read_csv(train_path, names=train_cols, sep="\t")
    for col in sparse_cols:
        train_data[col] = train_data[col].apply(
            lambda x: sparse_value_func(col, x))
    for col in dense_cols:
        train_data[col] = train_data[col].apply(
            lambda x: dense_value_func(col, x))

    train, test = train_test_split(train_data, test_size=0.2, random_state=42)
    train_dataset = Data.TensorDataset(
        torch.from_numpy(train[feature_cols].values.astype(np.int64)),
        torch.FloatTensor(train["label"].values),
    )

    train_loader = Data.DataLoader(
        dataset=train_dataset, batch_size=2048, shuffle=True)

    test_dataset = Data.TensorDataset(
        torch.from_numpy(test[feature_cols].values.astype(np.int64)),
        torch.FloatTensor(test["label"].values),
    )
    test_loader = Data.DataLoader(
        dataset=test_dataset, batch_size=4096, shuffle=False)

    return (
        train_loader,
        test_loader,
    )
