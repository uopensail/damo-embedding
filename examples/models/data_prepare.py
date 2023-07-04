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

import luban
import luban_parser
import numpy as np
import pandas as pd
import tensorflow as tf
import torch
import torch.utils.data as Data
from sklearn.model_selection import train_test_split

config = {
    "transforms": [
        'C_0_T = concat("C-0", C_0)',
        'C_1_T = concat("C-0", C_1)',
        'C_2_T = concat("C-0", C_2)',
        'C_3_T = concat("C-0", C_3)',
        'C_4_T = concat("C-0", C_4)',
        'C_5_T = concat("C-0", C_5)',
        'C_6_T = concat("C-0", C_6)',
        'C_7_T = concat("C-0", C_7)',
        'C_8_T = concat("C-0", C_8)',
        'C_9_T = concat("C-0", C_9)',
        'C_10_T = concat("C-0", C_10)',
        'C_11_T = concat("C-0", C_11)',
        'C_12_T = concat("C-0", C_12)',
        'C_13_T = concat("C-0", C_13)',
        'C_14_T = concat("C-0", C_14)',
        'C_15_T = concat("C-0", C_15)',
        'C_16_T = concat("C-0", C_16)',
        'C_17_T = concat("C-0", C_17)',
        'C_18_T = concat("C-0", C_18)',
        'C_19_T = concat("C-0", C_19)',
        'C_20_T = concat("C-0", C_20)',
        'C_21_T = concat("C-0", C_21)',
        'C_22_T = concat("C-0", C_22)',
        'C_23_T = concat("C-0", C_23)',
        'C_24_T = concat("C-0", C_24)',
        'C_25_T = concat("C-0", C_25)',
    ],
    "outputs": [
        {"name": "C_0_T", "slot": 0},
        {"name": "C_1_T", "slot": 1},
        {"name": "C_2_T", "slot": 2},
        {"name": "C_3_T", "slot": 3},
        {"name": "C_4_T", "slot": 4},
        {"name": "C_5_T", "slot": 5},
        {"name": "C_6_T", "slot": 6},
        {"name": "C_7_T", "slot": 7},
        {"name": "C_8_T", "slot": 8},
        {"name": "C_9_T", "slot": 9},
        {"name": "C_10_T", "slot": 10},
        {"name": "C_11_T", "slot": 11},
        {"name": "C_12_T", "slot": 12},
        {"name": "C_13_T", "slot": 13},
        {"name": "C_14_T", "slot": 14},
        {"name": "C_15_T", "slot": 15},
        {"name": "C_16_T", "slot": 16},
        {"name": "C_17_T", "slot": 17},
        {"name": "C_18_T", "slot": 18},
        {"name": "C_19_T", "slot": 19},
        {"name": "C_20_T", "slot": 20},
        {"name": "C_21_T", "slot": 21},
        {"name": "C_22_T", "slot": 22},
        {"name": "C_23_T", "slot": 23},
        {"name": "C_24_T", "slot": 24},
        {"name": "C_25_T", "slot": 25},
        {"name": "I_0", "slot": 26},
        {"name": "I_1", "slot": 27},
        {"name": "I_3", "slot": 28},
        {"name": "I_4", "slot": 29},
        {"name": "I_5", "slot": 30},
        {"name": "I_6", "slot": 31},
        {"name": "I_7", "slot": 32},
        {"name": "I_8", "slot": 33},
        {"name": "I_9", "slot": 34},
        {"name": "I_10", "slot": 35},
        {"name": "I_11", "slot": 36},
        {"name": "I_12", "slot": 37},
        {"name": "I_2", "slot": 38},
    ],
}


def to_tfrecord(train_path: str, tfrecord_path: str):
    """to_tfrecord

    Args:
        train_path (str): train path
        tfrecord_path (str): tf record path
    """
    sparse_cols = [f"C_{i}" for i in range(26)]
    dense_cols = [f"I_{i}" for i in range(13)]
    train_cols = ["label"] + dense_cols + sparse_cols
    train_data = pd.read_csv(train_path, names=train_cols, sep="\t")

    with tf.io.TFRecordWriter(tfrecord_path) as writer:
        for _, row in train_data.iterrows():
            example = transform(row, sparse_cols, dense_cols)
            writer.write(example.SerializeToString())


def transform(
    row: pd.Series, sparse_cols: List[str], dense_cols: List[str]
) -> tf.train.Example:
    """transform row to example

    Args:
        row (pd.Series): row to transform
        sparse_cols (List[str]): sparse columns
        dense_cols (List[str]): dense columns

    Returns:
        tf.train.Example: tfrecord's example
    """
    dic = {}
    dic["label"] = tf.train.Feature(
        int64_list=tf.train.Int64List(value=[int(row["label"])])
    )
    for col in sparse_cols:
        value = str(row[col])
        dic[col] = tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[value.encode("utf8")])
        )
    for col in dense_cols:
        if not np.isnan(row[col]):
            dic[col] = tf.train.Feature(
                float_list=tf.train.FloatList(value=[float(row[col])])
            )

    return tf.train.Example(features=tf.train.Features(feature=dic))


def luban_process(train_path: str, outpath: str):
    """hash keys

    Args:
        train_path (str): tfrecord path
        outpath (str): hashed path
    """
    json.dump(config, open("config.json", "w"))
    luban_parser.parse("config.json", "config.toml")
    toolkit = luban.PyToolKit("config.toml")
    toolkit.process_file(train_path, outpath)


def process(train_path: str) -> Tuple[Data.DataLoader, Data.DataLoader]:
    """hash all the features to uint64

    Args:
        train_path (str): train data path

    Returns:
        Tuple[Data.DataLoader, Data.DataLoader]: train and test loaders
    """
    to_tfrecord(train_path, "train.tfrecord")
    luban_process("train.tfrecord", "train.txt")
    os.remove("train.tfrecord")
    cols = [f"col-{i}" for i in range(39)]
    train_cols = ["label"] + cols
    train_data = pd.read_csv("train.txt", names=train_cols, sep=" ")
    os.remove("train.txt")
    train, test = train_test_split(train_data, test_size=0.2, random_state=42)
    train_dataset = Data.TensorDataset(
        torch.from_numpy(train[cols].values.astype(np.int64)),
        torch.FloatTensor(train["label"].values),
    )

    train_loader = Data.DataLoader(dataset=train_dataset, batch_size=2048, shuffle=True)

    test_dataset = Data.TensorDataset(
        torch.from_numpy(test[cols].values.astype(np.int64)),
        torch.FloatTensor(test["label"].values),
    )
    test_loader = Data.DataLoader(dataset=test_dataset, batch_size=4096, shuffle=False)

    return (
        train_loader,
        test_loader,
    )


if __name__ == "__main__":
    train_loader, valid_loader = process("sample.txt")
    print(train_loader, valid_loader)
