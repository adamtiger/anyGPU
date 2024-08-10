set(TRF_ROOT ${CMAKE_CURRENT_LIST_DIR})
set(TRF_INCLUDE_DIR ${CMAKE_CURRENT_LIST_DIR})


file(GLOB TRF_SRCS 
    ${TRF_ROOT}/*.cu
	${TRF_ROOT}/*.cpp
    ${TRF_ROOT}/*.cc
)
file(GLOB TRF_HEADERS
    ${TRF_ROOT}/*.cuh
	${TRF_ROOT}/*.hpp 
	${TRF_ROOT}/*.h
)
source_group("transformers" FILES ${TRF_HEADERS} ${TRF_SRCS})

set (SOURCES
    ${TRF_SRCS}
)

set (HEADERS
    ${TRF_HEADERS}
)

set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /MD")

cuda_add_executable(trf ${SOURCES} ${HEADERS})
target_include_directories(trf PUBLIC ${TRF_INCLUDE_DIR})
target_link_options(trf PUBLIC "$<$<CONFIG:Release>:/NODEFAULTLIB:libcmt.lib>") 
