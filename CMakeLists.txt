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
    message("Compile with cuda")
endif()


set(CMAKE_POSITION_INDEPENDENT_CODE ON)


set(ANYGPU_ROOT ${CMAKE_CURRENT_LIST_DIR})

# core

set(CORE_DIR ${ANYGPU_ROOT}/core)

set (CORE_SRCS
    ${CORE_DIR}/core.cpp
	${CORE_DIR}/tensor.cpp
)

set (CORE_HEADERS
    ${CORE_DIR}/core.hpp
	${CORE_DIR}/core_concepts.hpp
	${CORE_DIR}/tensor.hpp
)

source_group("core" FILES ${CORE_HEADERS} ${CORE_SRCS})

# tests

set(TEST_DIR ${ANYGPU_ROOT}/tests)

set (TEST_SRCS
    ${TEST_DIR}/tests.cpp
    ${TEST_DIR}/ext_torch_tests.cpp
)

set (TEST_HEADERS
    ${TEST_DIR}/tests.hpp
    ${TEST_DIR}/ext_torch_tests.hpp
)

source_group("tests" FILES ${TEST_HEADERS} ${TEST_SRCS})

# ops

set(OPS_DIR ${ANYGPU_ROOT}/ops)

set (OPS_SRCS
    ${OPS_DIR}/binary_ops.cu
	${OPS_DIR}/mm_ops.cu
	${OPS_DIR}/transp_ops.cu
    ${OPS_DIR}/softmax_ops.cu
)

set (OPS_HEADERS
    ${OPS_DIR}/binary_ops.cuh
    ${OPS_DIR}/binary_ops.hpp
    ${OPS_DIR}/mm_ops.cuh
    ${OPS_DIR}/mm_ops.hpp
	${OPS_DIR}/transp_ops.cuh
	${OPS_DIR}/transp_ops.hpp
    ${OPS_DIR}/softmax_ops.cuh
    ${OPS_DIR}/softmax_ops.hpp
    ${OPS_DIR}/ops.hpp
)

source_group("ops" FILES ${OPS_HEADERS} ${OPS_SRCS})

# transformers

set(TRF_DIR ${ANYGPU_ROOT}/transformers)

set (TRF_SRCS
    
)

set (TRF_HEADERS
    ${TRF_DIR}/attention.hpp
    ${TRF_DIR}/sdp.hpp
)

source_group("transformers" FILES ${TRF_HEADERS} ${TRF_SRCS})


# filter for main and similar files in visual studio
source_group("apps" FILES ${ANYGPU_ROOT}/main.cpp)

# set all sources
set (SOURCES
    ${ANYGPU_ROOT}/main.cpp
    ${TRF_SRCS}
	${CORE_SRCS}
	${TEST_SRCS}
	${OPS_SRCS}
)

set (HEADERS
    ${TRF_HEADERS}
	${CORE_HEADERS}
	${TEST_HEADERS}
	${OPS_HEADERS}
)

set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /MD")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /MDd")

cuda_add_executable(anygpu ${SOURCES} ${HEADERS})
target_include_directories(anygpu PUBLIC ${TRF_DIR} ${CORE_DIR} ${TEST_DIR} ${OPS_DIR})
target_link_options(anygpu PUBLIC "/NODEFAULTLIB:libcmt.lib") 

message("Finished")
