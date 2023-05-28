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
import json

# 设置参数
param = damo.Parameters()

# 这里要注意，如果是浮点数，必须明确类型
param.insert("float_type", 3.0)  # yes
param.insert("float_type", float(3))  # yes
# param.insert("float_type", 3)              # error, 3 is int type

param.insert("string_type", "value")
param.insert("int_type", 3)
param.insert("bool_type", True)

value = param.to_json()
print(json.loads(value))
