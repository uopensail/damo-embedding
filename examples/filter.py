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

# 设置参数
param = damo.Parameters()
param.insert("capacity", 1 << 28)
param.insert("count", 15)
param.insert("path", "/tmp/cbf")
param.insert("reload", True)
param.insert("ffp", 0.001)
print(param.to_json())

filter = damo.PyFilter(param)

key = 123456
for i in range(16):
    filter.add(key, 1)
    print(filter.check(key))
