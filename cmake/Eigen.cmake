#===============================================================================
# Copyright 2018 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#===============================================================================

# Manage Eigen-related compiler flags
#===============================================================================

if(Eigen_cmake_included)
    return()
endif()
set(Eigen_cmake_included true)

if(NOT MKLDNN_THREADING STREQUAL "EIGEN")
    return()
endif()

if (NOT EIGENROOT)
    if(DEFINED ENV{EIGENROOT})
        set (EIGENROOT $ENV{EIGENROOT})
    else()
        message("FATAL_ERROR" "EIGENROOT is unset")
    endif()
endif()

find_package(Threads REQUIRED)
if(CMAKE_THREAD_LIBS_INIT)
    list(APPEND mkldnn_LINKER_LIBS ${CMAKE_THREAD_LIBS_INIT})
endif()

set_threading("EIGEN")
include_directories(${EIGENROOT})

message(STATUS "Eigen: ${EIGENROOT}")
