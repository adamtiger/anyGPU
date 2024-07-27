cmake_minimum_required(VERSION 3.22)

project(AnyGPU)
message("Start generating!")

option(BUILD_WITH_CUDA "Whether NVIDIA CUDA based modules built" ON)
option(BUILD_WITH_SYCL "Whether SYCL based modules built" ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(CMAKE_CXX_EXTENSIONS OFF)

if (BUILD_WITH_CUDA)
    find_package(CUDA 12.5 REQUIRED)
	add_compile_definitions(XG_WITH_CUDA)
    message("use cuda operators")
endif()


# TODO: sycl related ...

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

add_subdirectory(transformers )
#add_subdirectory(tests )

message("Finished")
