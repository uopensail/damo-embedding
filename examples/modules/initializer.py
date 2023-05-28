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

import damo
import numpy as np

# zero
param = damo.Parameters()
param.insert("name", "zeros")
# 必须是float32类型
value = np.random.random(10).astype(np.float32)
obj = damo.PyInitializer(param)
obj.call(value)
print("zeros: ", value)

# ones
param = damo.Parameters()
param.insert("name", "ones")
# 必须是float32类型
value = np.random.random(10).astype(np.float32)
obj = damo.PyInitializer(param)
obj.call(value)
print("ones: ", value)

# random_uniform
param = damo.Parameters()
param.insert("name", "random_uniform")
param.insert("min", -1.0)
param.insert("max", 1.0)
# 必须是float32类型
value = np.random.random(10).astype(np.float32)
obj = damo.PyInitializer(param)
obj.call(value)
print("random_uniform: ", value)

# random_normal
param = damo.Parameters()
param.insert("name", "random_normal")
param.insert("mean", 0.0)
param.insert("stddev", 1.0)
# 必须是float32类型
value = np.random.random(10).astype(np.float32)
obj = damo.PyInitializer(param)
obj.call(value)
print("random_normal: ", value)

# truncate_normal
param = damo.Parameters()
param.insert("name", "truncate_normal")
param.insert("mean", 0.0)
param.insert("stddev", 1.0)
# 必须是float32类型
value = np.random.random(10).astype(np.float32)
obj = damo.PyInitializer(param)
obj.call(value)
print("truncate_normal: ", value)
