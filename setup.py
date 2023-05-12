#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 3 17:58:27 2022

@author: timepi

@description: this is setup script for pydamo
"""
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
    if mac_ver.startswith("10"):
        minor_version = int(mac_ver.split(".")[1])
        return minor_version >= 7
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
    include_dirs=[
        "include",
        get_numpy_include()
    ],
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
    version="1.0.1",
    description="Python wrapper for damo, a set of fast and robust hash functions.",
    license="License :: GLP3",
    author="timepi",
    author_email="",
    url="https://github.com/uopensail/damo-embedding",
    packages=find_packages(),
    py_modules=["damo"],
    ext_modules=[damoModule],
    keywords="sparse embedding using rocksdb",
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
