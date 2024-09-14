cmake_minimum_required(VERSION 3.22)

set(CMAKE_CUDA_ARCHITECTURES 86)

project(AnyGPU CXX CUDA)
message("Start generating!")

option(BUILD_WITH_CUDA "Whether NVIDIA CUDA based modules built" ON)
option(BUILD_WITH_SYCL "Whether SYCL based modules built" ON)
option(BUILD_WITH_ASSERT "Whether to assert when needed" ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(CMAKE_CXX_EXTENSIONS OFF)

IF (BUILD_WITH_CUDA)
    #find_package(CUDA 12.5 REQUIRED)
	add_compile_definitions(XG_WITH_CUDA)
    message("Compile with cuda")
ENDIF()

IF (BUILD_WITH_ASSERT)
    add_compile_definitions(AC_WITH_ASSERT)
	message("Compile with asserts")
ENDIF(BUILD_WITH_ASSERT)

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

# io

set(IO_DIR ${ANYGPU_ROOT}/io)

set (IO_SRCS
    ${IO_DIR}/safetensors_file.cpp
)

set (IO_HEADERS
    ${IO_DIR}/dat_file.hpp
	${IO_DIR}/safetensors_file.hpp
)

source_group("io" FILES ${IO_HEADERS} ${IO_SRCS})

# tests

set(TEST_DIR ${ANYGPU_ROOT}/tests)

set (TEST_SRCS
    ${TEST_DIR}/tests.cpp
    ${TEST_DIR}/ext_torch_tests.cpp
	${TEST_DIR}/performance.cpp
)

set (TEST_HEADERS
    ${TEST_DIR}/test_tools.hpp
    ${TEST_DIR}/tests.hpp
    ${TEST_DIR}/ext_torch_tests.hpp
	${TEST_DIR}/performance.hpp
)

source_group("tests" FILES ${TEST_HEADERS} ${TEST_SRCS})

# ops

set(OPS_DIR ${ANYGPU_ROOT}/ops)

set (OPS_SRCS
    ${OPS_DIR}/binary_ops.cu
	${OPS_DIR}/mm_ops.cu
	${OPS_DIR}/transp_ops.cu
    ${OPS_DIR}/softmax_ops.cu
	${OPS_DIR}/quantize_ops.cu
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
	${OPS_DIR}/quantize_ops.cuh
    ${OPS_DIR}/quantize_ops.hpp
    ${OPS_DIR}/ops.hpp
)

source_group("ops" FILES ${OPS_HEADERS} ${OPS_SRCS})

# transformers

set(TRF_DIR ${ANYGPU_ROOT}/transformers)

set (TRF_SRCS
    
)

set (TRF_HEADERS
    ${TRF_DIR}/attention_tools.hpp
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
	${IO_SRCS}
	${TEST_SRCS}
	${OPS_SRCS}
)

set (HEADERS
    ${TRF_HEADERS}
	${CORE_HEADERS}
	${IO_HEADERS}
	${TEST_HEADERS}
	${OPS_HEADERS}
)

set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /MD")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /MDd")

add_executable(anygpu ${SOURCES} ${HEADERS})
target_include_directories(anygpu PUBLIC ${TRF_DIR} ${CORE_DIR} ${IO_DIR} ${TEST_DIR} ${OPS_DIR})
#target_compile_options(anygpu PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:--keep>)
target_link_options(anygpu PUBLIC "/NODEFAULTLIB:libcmt.lib") 

message("Finished")
