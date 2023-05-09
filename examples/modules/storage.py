#
# `Damo-Embedding` - 'c++ tool for sparse parameter server'
# Copyright(C) 2019 - present timepi < timepi123@gmail.com >
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

import numpy as np
import struct
import damo

# first param: data dir
# second param: ttl second
storage = damo.PyStorage("/tmp/data_dir", 86400*100)


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
    data_for_dim = f.read(256 * 4)
    group_dims = struct.unpack("@256i", data_for_dim)
    print(group_dims)
    data_for_count = f.read(256 * 8)
    group_counts = struct.unpack("@256Q", data_for_count)
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
        key = struct.unpack("@Q", data_for_key)[0]
        get_weight(key)
        data_for_key = f.read(8)
