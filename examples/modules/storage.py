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
import numpy as np
import struct
import damo

# first param: data dir
# second param: ttl second
storage = damo.PyStorage("/tmp/data_dir", 86400 * 100)


cond = damo.Parameters()
cond.insert("expire_days", 100)
cond.insert("min_count", 3)
cond.insert("group", 0)


# dump weights
path = "/tmp/weight.dat"
storage.dump(path, cond)

# extract weights
weight_dict = [{} for _ in range(256)]
with open(path, "rb") as f:
    data_for_dim = f.read(128 * 4)
    group_dims = struct.unpack("@128i", data_for_dim)
    print(group_dims)
    data_for_count = f.read(128 * 8)
    group_counts = struct.unpack("@128q", data_for_count)
    print(group_counts)

    def get_weight(key):
        data_for_group = f.read(4)
        group = struct.unpack("@I", data_for_group)[0]
        key_dim = group_dims[group]
        data_for_weight = f.read(4 * key_dim)
        weight = struct.unpack(f"@{key_dim}f", data_for_weight)
        weight = np.array(weight, dtype=np.float32)
        weight_dict[group][key] = weight
        print(group, key, weight)

    data_for_key = f.read(8)
    while data_for_key:
        key = struct.unpack("@q", data_for_key)[0]
        get_weight(key)
        data_for_key = f.read(8)
