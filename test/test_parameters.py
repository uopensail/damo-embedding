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
import unittest
from test import support
import damo
import random


class ParametersTestCase(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test(self):
        params = damo.Parameters()
        params.insert("str", "str")
        params.insert("int", 5)
        params.insert("float", float(5.0))
        params.insert("float2", 3.0)
        params.insert("float", 4.0)
        params.insert("bool", True)
        print(params.to_json())


if __name__ == "__main__":
    unittest.main()
