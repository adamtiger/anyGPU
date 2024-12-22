cmake_minimum_required(VERSION 3.26)

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

# fused

set(FUSED_DIR ${ANYGPU_ROOT}/srcs/fused)

set (FUSED_SRCS
    
)

set (FUSED_HEADERS
	${FUSED_DIR}/causal_conv1d.hpp
)

source_group("fused" FILES ${FUSED_HEADERS} ${FUSED_SRCS})

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
    ${TEST_DIR}/gemma2_tests.cpp
    ${TEST_DIR}/zamba2_tests.cpp
    ${TEST_DIR}/ext_torch_tests.cpp
	${TEST_DIR}/performance.cpp
)

set (TEST_HEADERS
    ${TEST_DIR}/test_tools.hpp
    ${TEST_DIR}/tests.hpp
    ${TEST_DIR}/gemma2_tests.hpp
    ${TEST_DIR}/zamba2_tests.hpp
    ${TEST_DIR}/ext_torch_tests.hpp
	${TEST_DIR}/performance.hpp
)

source_group("tests" FILES ${TEST_HEADERS} ${TEST_SRCS})

# ops

set(OPS_DIR ${ANYGPU_ROOT}/srcs/ops)

set (OPS_SRCS

)

set (OPS_HEADERS
    ${OPS_DIR}/binary_ops.hpp
    ${OPS_DIR}/math_ops.hpp
    ${OPS_DIR}/mm_ops.hpp
    ${OPS_DIR}/norm_ops.hpp
    ${OPS_DIR}/transp_ops.hpp
    ${OPS_DIR}/softmax_ops.hpp
    ${OPS_DIR}/quantize_ops.hpp
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

# kernels (cuda)

set(KERNEL_CU_DIR ${ANYGPU_ROOT}/srcs/kernels/cuda)

set (KRN_CU_SRCS
    ${KERNEL_CU_DIR}/binary_ops.cu
    ${KERNEL_CU_DIR}/math_ops.cu
    ${KERNEL_CU_DIR}/mm_ops.cu
    ${KERNEL_CU_DIR}/fast_mm_f32_v1.cu
    ${KERNEL_CU_DIR}/fast_mm_f32_v3.cu
    ${KERNEL_CU_DIR}/norm_ops.cu
    ${KERNEL_CU_DIR}/transp_ops.cu
    ${KERNEL_CU_DIR}/softmax_ops.cu
    ${KERNEL_CU_DIR}/quantize_ops.cu
    ${KERNEL_CU_DIR}/embedding_ops.cu
    ${KERNEL_CU_DIR}/causal_conv1d.cu
    ${KERNEL_CU_DIR}/mlp_gemma2_fused_upproj_f32_v1.cu
    ${KERNEL_CU_DIR}/mlp_gemma2_fused_upproj_f32_v2.cu
    ${KERNEL_CU_DIR}/mlp_gemma2_dp_linear_f32_v1.cu
    ${KERNEL_CU_DIR}/mlp_gemma2_dp_linear_f32_v4.cu
    ${KERNEL_CU_DIR}/sdpa_gemma2_linear_f32_v1.cu
)

set (KRN_CU_HEADERS
    ${KERNEL_CU_DIR}/binary_ops.cuh
    ${KERNEL_CU_DIR}/math_ops.cuh
    ${KERNEL_CU_DIR}/mm_ops.cuh
    ${KERNEL_CU_DIR}/fast_mm.cuh
    ${KERNEL_CU_DIR}/norm_ops.cuh
    ${KERNEL_CU_DIR}/transp_ops.cuh
    ${KERNEL_CU_DIR}/softmax_ops.cuh
    ${KERNEL_CU_DIR}/quantize_ops.cuh
    ${KERNEL_CU_DIR}/embedding_ops.cuh
    ${KERNEL_CU_DIR}/causal_conv1d.cuh
    ${KERNEL_CU_DIR}/mlp_gemma2_fused_upproj.cuh
    ${KERNEL_CU_DIR}/mlp_gemma2_dp_linear.cuh
    ${KERNEL_CU_DIR}/sdpa_gemma2_linear.cuh
)

source_group("kcuda" FILES ${KRN_CU_HEADERS} ${KRN_CU_SRCS})


# model related (zamba2)

set(ZAM2_DIR ${ANYGPU_ROOT}/srcs/models/zamba2)

set (ZAM2_SRCS
    ${ZAM2_DIR}/zamba_glu.cu
    ${ZAM2_DIR}/zamba_rotary.cu
)

set (ZAM2_HEADERS
    ${ZAM2_DIR}/zamba_glu.cuh
    ${ZAM2_DIR}/zamba_glu.hpp
    ${ZAM2_DIR}/zamba_mlp.hpp
    ${ZAM2_DIR}/zamba_rotary.cuh
    ${ZAM2_DIR}/zamba_rotary.hpp
    ${ZAM2_DIR}/zamba_sdpa.hpp
    ${ZAM2_DIR}/zamba_attn_decoder.hpp
    ${ZAM2_DIR}/zamba_gated_rmsnorm.hpp
)

source_group("mzamba2" FILES ${ZAM2_HEADERS} ${ZAM2_SRCS})

# model related (gemma2)

set(GEM2_DIR ${ANYGPU_ROOT}/srcs/models/gemma2)

set (GEM2_SRCS
    ${GEM2_DIR}/gemma_linsoftcap.cu
    ${GEM2_DIR}/gemma_update_mask.cu
    ${GEM2_DIR}/gemma_slide_mask.cu
)

set (GEM2_HEADERS
    ${GEM2_DIR}/gemma_config.hpp
    ${GEM2_DIR}/gemma_kv_cache.hpp
    ${GEM2_DIR}/gemma_mlp.hpp
    ${GEM2_DIR}/gemma_decoder.hpp
    ${GEM2_DIR}/gemma_sdpa.hpp
    ${GEM2_DIR}/gemma_model.hpp
    ${GEM2_DIR}/gemma_causallm.hpp
    ${GEM2_DIR}/gemma_linsoftcap.cuh
    ${GEM2_DIR}/gemma_linsoftcap.hpp
    ${GEM2_DIR}/gemma_update_mask.cuh
    ${GEM2_DIR}/gemma_update_mask.hpp
    ${GEM2_DIR}/gemma_slide_mask.cuh
    ${GEM2_DIR}/gemma_slide_mask.hpp
)

source_group("mgemma2" FILES ${GEM2_HEADERS} ${GEM2_SRCS})

# filter for main and similar files in visual studio
source_group("apps" FILES ${ANYGPU_ROOT}/main.cpp)

# set all sources
set (SOURCES
    ${ANYGPU_ROOT}/main.cpp
    ${ZAM2_SRCS}
    ${GEM2_SRCS}
    ${TRF_SRCS}
    ${CORE_SRCS}
    ${FUSED_SRCS}
    ${IO_SRCS}
    ${TEST_SRCS}
    ${OPS_SRCS}
    ${KRN_CU_SRCS}
)

set (HEADERS
    ${ZAM2_HEADERS}
    ${GEM2_HEADERS}
    ${TRF_HEADERS}
    ${CORE_HEADERS}
    ${FUSED_HEADERS}
    ${IO_HEADERS}
    ${TEST_HEADERS}
    ${OPS_HEADERS}
    ${KRN_CU_HEADERS}
)

if (MSVC OR CMAKE_CXX_SIMULATE_ID STREQUAL "MSVC")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /MD")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /MDd")
endif()

if (UNIX)
    set(CUDA_HEADERS "/usr/local/cuda-12.6/include")  # temporary!
endif()

add_executable(anygpu ${SOURCES} ${HEADERS})
target_include_directories(
    anygpu PUBLIC ${CUDA_HEADERS}
	${TRF_DIR} ${CORE_DIR} ${IO_DIR} ${KERNEL_CU_DIR}
	${TEST_DIR} ${OPS_DIR} ${FUSED_DIR}
	${ZAM2_DIR} ${GEM2_DIR} 
)
#target_compile_options(anygpu PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:--keep>)
if (MSVC)
    set_target_properties(anygpu PROPERTIES LINK_FLAGS "/PROFILE")
    target_link_options(anygpu PUBLIC "/NODEFAULTLIB:libcmt.lib")
endif()

message("Finished")
