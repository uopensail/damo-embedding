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
import unittest
import damo


class CountBloomFilterTestCase(unittest.TestCase):
    def setUp(self):
        params = damo.Parameters()
        self.filter = damo.PyFilter(params)

    def tearDown(self):
        pass

    def testAdd(self):
        key = 10000000

        for i in range(16):
            self.filter.add(1, key, 1)
            print(self.filter.check(1, key))
            print(self.filter.check(2, key))


if __name__ == "__main__":
    unittest.main()
