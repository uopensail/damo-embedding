#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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

import platform
import sys

from setuptools import Extension, setup, find_packages

COMPILE_OPTIONS = []
LINK_OPTIONS = []


def is_new_osx():
    """Check whether we're on OSX >= 10.7"""
    if sys.platform != "darwin":
        return False
    mac_ver = platform.mac_ver()[0]
    ver_ss = mac_ver.split(".")
    version = int(ver_ss[0])

    if version > 10:
        return True
    elif version == 10:
        minor_version = int(ver_ss[1])
        return minor_version >= 7
    else:
        return False


if is_new_osx():
    COMPILE_OPTIONS.append("-stdlib=libc++")
    COMPILE_OPTIONS.append("-std=c++17")
    LINK_OPTIONS.append("-lc++")
    LINK_OPTIONS.append("-lrocksdb")
    LINK_OPTIONS.append("-nodefaultlibs")
else:
    COMPILE_OPTIONS.append("-std=c++17")
    LINK_OPTIONS.append("-lrocksdb")
    LINK_OPTIONS.append("-lpthread")
    LINK_OPTIONS.append("-Wl,-rpath=/usr/local/lib")


class get_numpy_include(object):
    """Defer numpy.get_include() until after numpy is installed."""

    def __str__(self):
        import numpy

        return numpy.get_include()


damoModule = Extension(
    name="_damo",
    include_dirs=["include", get_numpy_include()],
    sources=[
        "src/pyembedding.cpp",
        "damo_wrap.cxx",
        "src/learning_rate_scheduler.cpp",
        "src/initializer.cpp",
        "src/embedding.cpp",
        "src/optimizer.cpp",
        "src/counting_bloom_filter.cpp",
        "src/common.cpp",
    ],
    extra_compile_args=COMPILE_OPTIONS,
    extra_link_args=LINK_OPTIONS,
)

with open("README.md", "r", encoding="utf-8") as fd:
    long_description = fd.read()

setup(
    name="damo-embedding",
    version="1.0.6",
    description="Python wrapper for damo, a set of fast and robust hash functions.",
    license="License :: AGLP3",
    author="timepi",
    author_email="",
    url="https://github.com/uopensail/damo-embedding",
    packages=find_packages(),
    py_modules=["damo", "damo_embedding"],
    ext_modules=[damoModule],
    keywords=[
        "sparse embedding using rocksdb",
        "parameter server",
        "ftrl",
        "adam",
        "adagrad",
    ],
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=["numpy>=1.19.0"],
    setup_requires=["numpy>=1.19.0"],
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Software Development :: Libraries",
        "Topic :: Utilities",
    ],
)
