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


SET(DAMO_INCLUDE include/common.h include/json.hpp include/embedding.h include/initializer.h include/optimizer.h)
SET(DAMO_SOURCE src/common.cpp src/embedding.cpp src/initializer.cpp src/optimizer.cpp)

# Include pybind11
find_package(pybind11 REQUIRED)
# Define the Python module
pybind11_add_module(damo ${DAMO_INCLUDE} ${DAMO_SOURCE} include/damo.h src/damo.cpp src/pydamo.cpp)

add_executable(damo-server ${DAMO_INCLUDE} ${DAMO_SOURCE} src/main.cpp include/httplib.h include/argparse.hpp)

target_link_libraries(damo PRIVATE -ldl -lrocksdb -lpthread)
target_link_libraries(damo-server -ldl -lrocksdb -lpthread)