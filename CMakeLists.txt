cmake_minimum_required(VERSION 3.22)

set(CMAKE_CUDA_ARCHITECTURES 86)

project(AnyGPU CXX CUDA)
message("Start generating!")

option(BUILD_WITH_CUDA "Whether NVIDIA CUDA based modules built" ON)
option(BUILD_WITH_VULKAN "Whether Vulkan based modules built" ON)
option(BUILD_WITH_SYCL "Whether SYCL based modules built" ON)
option(BUILD_WITH_ASSERT "Whether to assert when needed" ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(CMAKE_CXX_EXTENSIONS OFF)

IF (BUILD_WITH_CUDA)
	add_compile_definitions(AC_WITH_CUDA)
    message("Compile with cuda")
ENDIF()

IF (BUILD_WITH_VULKAN)
    find_package(Vulkan REQUIRED)
	add_compile_definitions(AC_WITH_VULKAN)
    message("Compile with vulkan")
ENDIF()

IF (BUILD_WITH_ASSERT)
    add_compile_definitions(AC_WITH_ASSERT)
	message("Compile with asserts")
ENDIF(BUILD_WITH_ASSERT)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)


set(ANYGPU_ROOT ${CMAKE_CURRENT_LIST_DIR})

# core

set(CORE_DIR ${ANYGPU_ROOT}/srcs/core)

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

set(IO_DIR ${ANYGPU_ROOT}/srcs/io)

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
    ${TEST_DIR}/ext_zamba2_tests.cpp
    ${TEST_DIR}/ext_torch_tests.cpp
	${TEST_DIR}/performance.cpp
	${TEST_DIR}/vk_relu_skeleton.cpp
)

set (TEST_HEADERS
    ${TEST_DIR}/test_tools.hpp
    ${TEST_DIR}/tests.hpp
    ${TEST_DIR}/ext_zamba2_tests.hpp
    ${TEST_DIR}/ext_torch_tests.hpp
	${TEST_DIR}/performance.hpp
	${TEST_DIR}/vk_relu_skeleton.hpp
)

source_group("tests" FILES ${TEST_HEADERS} ${TEST_SRCS})

# ops

set(OPS_DIR ${ANYGPU_ROOT}/srcs/ops)

set (OPS_SRCS
    ${OPS_DIR}/binary_ops.cu
    ${OPS_DIR}/math_ops.cu
    ${OPS_DIR}/mm_ops.cu
    ${OPS_DIR}/norm_ops.cu
    ${OPS_DIR}/transp_ops.cu
    ${OPS_DIR}/softmax_ops.cu
    ${OPS_DIR}/quantize_ops.cu
    ${OPS_DIR}/embedding_ops.cu
)

set (OPS_HEADERS
    ${OPS_DIR}/binary_ops.cuh
    ${OPS_DIR}/binary_ops.hpp
    ${OPS_DIR}/math_ops.cuh
    ${OPS_DIR}/math_ops.hpp
    ${OPS_DIR}/mm_ops.cuh
    ${OPS_DIR}/mm_ops.hpp
    ${OPS_DIR}/norm_ops.cuh
    ${OPS_DIR}/norm_ops.hpp
    ${OPS_DIR}/transp_ops.cuh
    ${OPS_DIR}/transp_ops.hpp
    ${OPS_DIR}/softmax_ops.cuh
    ${OPS_DIR}/softmax_ops.hpp
    ${OPS_DIR}/quantize_ops.cuh
    ${OPS_DIR}/quantize_ops.hpp
    ${OPS_DIR}/embedding_ops.cuh
    ${OPS_DIR}/embedding_ops.hpp
    ${OPS_DIR}/shape_ops.hpp
    ${OPS_DIR}/ops.hpp
)

source_group("ops" FILES ${OPS_HEADERS} ${OPS_SRCS})

# transformers

set(TRF_DIR ${ANYGPU_ROOT}/srcs/transformers)

set (TRF_SRCS
    
)

set (TRF_HEADERS
    ${TRF_DIR}/attention_tools.hpp
    ${TRF_DIR}/attention.hpp
    ${TRF_DIR}/sdp.hpp
)

source_group("transformers" FILES ${TRF_HEADERS} ${TRF_SRCS})

# model related (zamba2)

set(ZAM2_DIR ${ANYGPU_ROOT}/srcs/zamba2)

set (ZAM2_SRCS
    
)

set (ZAM2_HEADERS
    
)

source_group("zamba2" FILES ${ZAM2_HEADERS} ${ZAM2_SRCS})

# filter for main and similar files in visual studio
source_group("apps" FILES ${ANYGPU_ROOT}/main.cpp)

# set all sources
set (SOURCES
    ${ANYGPU_ROOT}/main.cpp
	${ZAM2_SRCS}
    ${TRF_SRCS}
	${CORE_SRCS}
	${IO_SRCS}
	${TEST_SRCS}
	${OPS_SRCS}
)

set (HEADERS
    ${ZAM2_HEADERS}
    ${TRF_HEADERS}
	${CORE_HEADERS}
	${IO_HEADERS}
	${TEST_HEADERS}
	${OPS_HEADERS}
)

set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /MD")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /MDd")

add_executable(anygpu ${SOURCES} ${HEADERS})
target_include_directories(anygpu PUBLIC ${Vulkan_INCLUDE_DIR} ${TRF_DIR} ${CORE_DIR} ${IO_DIR} ${TEST_DIR} ${OPS_DIR})
#target_compile_options(anygpu PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:--keep>)
target_link_options(anygpu PUBLIC "/NODEFAULTLIB:libcmt.lib") 
target_link_libraries(anygpu PUBLIC ${Vulkan_LIBRARIES})

message("Finished")
