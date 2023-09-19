cmake_minimum_required(VERSION 3.16)
project(damo)

set(CMAKE_CXX_STANDARD 17)

include_directories(include)
include_directories(/usr/local/include)
include_directories(/usr/include)

link_directories(/usr/local/lib)


# Find and link to Python
find_package(Python3 COMPONENTS Interpreter Development)
include_directories(${Python3_INCLUDE_DIRS})
message(STATUS "PYTHON3_LIBRARIES = ${Python3_LIBRARIES}")
message(STATUS "PYTHON3_EXECUTABLE = ${PYTHON_EXECUTABLE}")
message(STATUS "PYTHON3_INCLUDE_DIRS = ${Python3_INCLUDE_DIRS}")

find_package(RocksDB)
if(ROCKSDB_FOUND)
    include_directories(${ROCKSDB_INCLUDE_DIRS})
    link_directories(${ROCKSDB_LIBRARY_DIRS})
endif()


SET(DAMO_INCLUDE include/common.h include/counting_bloom_filter.h include/cpptoml.h
    include/embedding.h include/initializer.h include/learning_rate_scheduler.h
    include/optimizer.h include/pyembedding.h)

SET(DAMO_SOURCE src/common.cpp src/counting_bloom_filter.cpp src/embedding.cpp
    src/initializer.cpp src/learning_rate_scheduler.cpp src/optimizer.cpp
    src/pydamo.cpp src/pyembedding.cpp)

# Include pybind11
find_package(pybind11 REQUIRED)
# Define the Python module
pybind11_add_module(damo ${DAMO_INCLUDE} ${DAMO_SOURCE})

# target_link_libraries(toolkit pthread)
target_link_libraries(damo PRIVATE -lrocksdb -lpthread)