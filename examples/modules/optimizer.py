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

# configure learning rate scheduler
schedluer_params = damo.Parameters()
schedluer_params.insert("name", "")

# configure optimizer
optimizer_params = damo.Parameters()
optimizer_params.insert("name", "sgd")
optimizer_params.insert("gamma", 0.001)
optimizer_params.insert("lambda", 0.0)

# no scheduler
opt1 = damo.PyOptimizer(optimizer_params)

# specific scheduler
# opt1 = damo.PyOptimizer(optimizer_params, schedluer_params)

w = np.zeros(10, dtype=np.float32)
gs = np.random.random(10).astype(np.float32)
step = 0
print("w: ", w)
print("gs: ", gs)


opt1.call(w, gs, step)
print("w: ", w)

print("w: ", w)
print("gs: ", gs)
# default step is 0
opt1.call(w, gs)
print("w: ", w)
