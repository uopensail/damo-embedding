cmake_minimum_required(VERSION 3.16)
project(damo)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 REQUIRED)
find_package(RocksDB)

set(DAMO_SOURCES
    src/common.cc
    src/embedding.cc
    src/initializer.cc
    src/optimizer.cc
    src/damo.cc
    src/pydamo.cc
)
pybind11_add_module(damo ${DAMO_SOURCES})

target_include_directories(damo PRIVATE
    include
    ${Python3_INCLUDE_DIRS}
)

if(ROCKSDB_FOUND)
    target_include_directories(damo PRIVATE ${ROCKSDB_INCLUDE_DIRS})
    target_link_libraries(damo PRIVATE ${ROCKSDB_LIBRARIES})
endif()

target_link_libraries(damo PRIVATE
    rocksdb
    Python3::Python
    pybind11::module
    dl
    pthread
)
