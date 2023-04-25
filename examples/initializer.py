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
