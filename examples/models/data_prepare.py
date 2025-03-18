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

# dataset: https://www.kaggle.com/datasets/mrkmakr/criteo-dataset

import json
import os
from typing import List, Tuple
import minia
import numpy as np
import pandas as pd
import torch
import torch.utils.data as Data
from sklearn.model_selection import train_test_split


def to_json(train_path: str, json_path: str):
    """to_json

    Args:
        train_path (str): train path
        json_path (str): json path
    """
    sparse_cols = [f"C_{i}" for i in range(26)]
    dense_cols = [f"I_{i}" for i in range(13)]
    train_cols = ["label"] + dense_cols + sparse_cols
    train_data = pd.read_csv(train_path, names=train_cols, sep=",", header=1)

    with open(json_path, "w") as writer:
        for _, row in train_data.iterrows():
            example = transform(row, sparse_cols, dense_cols)
            writer.write(json.dumps(example) + "\n")


def transform(row: pd.Series, sparse_cols: List[str], dense_cols: List[str]) -> dict:
    """transform row to json

    Args:
        row (pd.Series): row to transform
        sparse_cols (List[str]): sparse columns to transform
        dense_cols (List[str]): dense columns to transform

    Returns:
        dict: result of transform
    """
    dic = {
        "label": {
            "type": 0,
            "value": int(row["label"]),
        }
    }
    for col in sparse_cols:
        dic[col] = {
            "type": 2,
            "value": str(row[col]),
        }
    for col in dense_cols:
        if not np.isnan(row[col]):
            dic[col] = {
                "type": 1,
                "value": float(row[col]),
            }
        else:
            dic[col] = {
                "type": 1,
                "value": 0.0,
            }
    return dic


def minia_process(config_path: str, data_path: str) -> list:
    """use luban to generate training data

    Args:
        config_path (str): configure path
        data_path (str): train data path

    Returns:
        list: training data records
    """
    toolkit = minia.Minia("config.toml")
    json_path = data_path + ".json"
    to_json(data_path, json_path)
    ret = []
    with open(json_path, "r") as reader:
        line = reader.readline()
        while line:
            line = line.strip()
            if len(line) > 0:
                m = toolkit(line)
                label = json.loads(line)["label"]["value"]
                ret.append((label, list(m.values())))
            line = reader.readline()
    os.remove(json_path)
    return ret


def process(
    config_path: str, train_path: str
) -> Tuple[Data.DataLoader, Data.DataLoader]:
    """hash all the features to uint64

    Args:
        config_path (str): configure path
        train_path (str): train data path

    Returns:
        Tuple[Data.DataLoader, Data.DataLoader]: train and test loaders
    """
    train_data = minia_process(config_path, train_path)
    train, test = train_test_split(train_data, test_size=0.2, random_state=42)
    train_dataset = Data.TensorDataset(
        torch.from_numpy(
            np.array(list(map(lambda _: _[1], train)), dtype=np.int64)
        ),
        torch.from_numpy(np.array(list(map(lambda _: _[0], train)), dtype=np.float32)),
    )

    train_loader = Data.DataLoader(dataset=train_dataset, batch_size=2048, shuffle=True)

    test_dataset = Data.TensorDataset(
        torch.from_numpy(
            np.array(list(map(lambda _: _[1], test)), dtype=np.int64)
        ),
        torch.from_numpy(np.array(list(map(lambda _: _[0], test)), dtype=np.float32)),
    )
    test_loader = Data.DataLoader(dataset=test_dataset, batch_size=4096, shuffle=False)

    return (
        train_loader,
        test_loader,
    )


if __name__ == "__main__":
    train_loader, valid_loader = process("config.toml", "criteo_sample.txt")
    print(train_loader, valid_loader)
