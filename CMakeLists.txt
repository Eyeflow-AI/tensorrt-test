set(CMAKE_POLICY_DEFAULT_CMP0135 NEW)
cmake_minimum_required(VERSION 3.20 FATAL_ERROR)

project(tensorrt_test
        VERSION 1.0
        DESCRIPTION "TensorRT Test"
        LANGUAGES CXX
)

include(CTest)
enable_testing()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_THREAD_LIBS_INIT "-lpthread")
SET(CMAKE_CXX_FLAGS "-Wno-deprecated-declarations ${CMAKE_CXX_FLAGS} -pthread -DBUILD_SYSTEM=cmake_oss")
set(CMAKE_HAVE_THREADS_LIBRARY 1)
set(CMAKE_USE_WIN32_THREADS_INIT 0)
set(CMAKE_USE_PTHREADS_INIT 1)
set(THREADS_PREFER_PTHREAD_FLAG ON)

add_definitions(-DCMAKE_EXPORT_COMPILE_COMMANDS=ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})

include(CPack)

# Cross-compilation settings
if (${CMAKE_SYSTEM_PROCESSOR} STREQUAL "x86_64")
    set(EFL_PLATFORM_ID "x86_64")
elseif(${CMAKE_SYSTEM_PROCESSOR} STREQUAL "aarch64")
    set(EFL_PLATFORM_ID "aarch64")
endif()

message(STATUS "Targeting Edge Platform: ${EFL_PLATFORM_ID}")

add_executable(${PROJECT_NAME}
    inference.cpp
    new_model.cpp
    old_model.cpp
    trt_utils.cpp
    trt_utils_old.cpp
)

target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_17)
target_compile_definitions(${PROJECT_NAME} PRIVATE JSON_DIAGNOSTICS=1)
target_link_libraries(${PROJECT_NAME} PRIVATE ${CMAKE_DL_LIBS})


# Edge SDK
include_directories( /opt/edge_sdk/include )
link_directories( /opt/edge_sdk/lib )
target_link_libraries(${PROJECT_NAME} PRIVATE /opt/edge_sdk/lib/libedge_sdk.so.2)


# JSON
include(FetchContent)
FetchContent_Declare(json URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz)
FetchContent_MakeAvailable(json)
target_link_libraries(${PROJECT_NAME} PRIVATE nlohmann_json::nlohmann_json)


# Google LOG
find_package(glog 0.7.0 REQUIRED)
include_directories(${glog_INCLUDE_DIRS})
link_directories(${glog_LIBRARY_DIRS})
target_link_libraries(${PROJECT_NAME} PRIVATE glog::glog)


# OpenCV
find_package(OpenCV  REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} PRIVATE ${OpenCV_LIBS})
target_link_libraries(${PROJECT_NAME} PRIVATE X11)


# TensorRT
if(POLICY CMP0148)
    cmake_policy(SET CMP0146 OLD)
endif()
set(CUDA_VERSION 12.4)
find_package(CUDA ${CUDA_VERSION} REQUIRED)
include_directories(
    ${CUDA_INCLUDE_DIRS}
    ${CUDNN_ROOT_DIR}/include
)
message(STATUS "FIND library = [" ${CUDA_LIBRARIES} "]")
target_link_libraries(${PROJECT_NAME} PRIVATE ${CUDA_LIBRARIES})
include_directories( /opt/TensorRT-10.0.1.6/include )
link_directories( /opt/TensorRT-10.0.1.6/lib )
find_library(NVINFER_LIB
    NAMES nvinfer
    HINTS /opt/TensorRT-10.0.1.6/lib
    REQUIRED
)
message(STATUS "FIND library = [" ${NVINFER_LIB} "]")
target_link_libraries(${PROJECT_NAME} PRIVATE ${NVINFER_LIB})
find_library(NVINFER_PLUGIN_LIB
    NAMES nvinfer_plugin
    HINTS /opt/TensorRT-10.0.1.6/lib
    REQUIRED
)
message(STATUS "FIND library = [" ${NVINFER_PLUGIN_LIB} "]")
target_link_libraries(${PROJECT_NAME} PRIVATE ${NVINFER_PLUGIN_LIB})
