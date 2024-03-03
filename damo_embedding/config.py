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
import os


# damo instance
DAMO_INSTANCE = None

# damo embedding http address
DAMO_SERVICE_ADDRESS = "http://localhost:9275"

# damo server binaray file path
DAMO_SERVICE_BINARY = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "damo-server"
)

global_step_control = 0
def set_global_step_control(train_id: int, step: int):
    global global_step_control
    global_step_control = ((train_id << 32&0xffffffff00000000) | (step&0x00000000ffffffff))
    