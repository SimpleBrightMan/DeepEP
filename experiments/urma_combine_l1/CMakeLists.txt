cmake_minimum_required(VERSION 3.20)
project(urma_combine_l1 LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CATLASS_ROOT "$ENV{CATLASS_ROOT}" CACHE PATH "Production CATLASS source tree")
if(NOT CATLASS_ROOT)
    get_filename_component(CATLASS_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../third-party/catlass" ABSOLUTE)
endif()
if(NOT EXISTS "${CATLASS_ROOT}/include/catlass/catlass.hpp")
    message(FATAL_ERROR "Set CATLASS_ROOT to the same CATLASS source used to build MegaMoE")
endif()
include("${CMAKE_CURRENT_SOURCE_DIR}/../megamoe/cmake/Headers.cmake")
ascend_deepep_find_mega_moe_headers()
find_program(BISHENG bisheng PATHS "${ASCEND_HOME_PATH}/tools/bisheng_compiler/bin" NO_DEFAULT_PATH REQUIRED)
set(inc "${ASCEND_HOME_PATH}/include" "${ASCEND_HOME_PATH}/pkg_inc"
    "${ASCEND_HOME_PATH}/include/experiment/runtime" "${ASCEND_HOME_PATH}/pkg_inc/runtime"
    "${SHMEM_ROOT}/include" "${SHMEM_ROOT}/src" "${SHMEM_ROOT}/src/device" "${SHMEM_ROOT}/src/device_simt"
    "${CATLASS_ROOT}/include" "${CATLASS_ROOT}/examples/common"
    "${CMAKE_CURRENT_SOURCE_DIR}/../megamoe/csrc" "${CMAKE_CURRENT_SOURCE_DIR}/../megamoe/csrc/kernels/mega_moe"
    "${ASCEND_HOME_PATH}/include/op_common" "${ASCEND_HOME_PATH}/pkg_inc/op_common"
    "${ASCEND_HOME_PATH}/compiler/ascendc/act" "${ASCEND_HOME_PATH}/aarch64-linux/asc/atcos"
    "${ASCEND_DEEPEP_MEGA_MOE_COMMON_INCLUDE_DIR}"
    "${ASCEND_DEEPEP_MEGA_MOE_COMMON_INCLUDE_DIR}/tensor_api"
    "${ASCEND_DEEPEP_MEGA_MOE_COMMON_INCLUDE_DIR}/tensor_api/include")
set(flags -O2 -std=c++17 -xasc --npu-arch=dav-3510 --cce-auto-infer-kernel-type=false
    --cce-auto-sync=off --cce-fatobj-link -shared -fPIC -DL2_CACHE_HINT
    -DCATLASS_ARCH=3510 -DASCEND_DEEPEP_DIRECT_TENSOR_INPUT=1
    -DASCEND_DEEPEP_DISABLE_MEGA_MOE_EXCEPTION_DUMP=1 -DENABLE_TENSOR_API
    -DNOT_DYNAMIC_COMPILE -DHIGH_PERFORMANCE=1 -DDETERMINISTIC_MODE=0)
foreach(p IN LISTS inc)
    list(APPEND flags "-I${p}")
endforeach()
file(GLOB_RECURSE protocol_headers CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/../megamoe/csrc/kernels/*.hpp")
add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/libcombine_l1.so"
    COMMAND "${BISHENG}" ${flags} "${CMAKE_CURRENT_SOURCE_DIR}/kernel.cpp"
    "-L${ASCEND_HOME_PATH}/lib64" "-L${SHMEM_ROOT}/lib" -lruntime -lascendcl -lshmem -lgraph_base -ltiling_api -lplatform -lregister -lstdc++ -ldl
    -o "${CMAKE_CURRENT_BINARY_DIR}/libcombine_l1.so"
    DEPENDS kernel.cpp layout.hpp compute.hpp ${protocol_headers} VERBATIM)
add_custom_target(device_build DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/libcombine_l1.so")
add_library(device SHARED IMPORTED)
set_target_properties(device PROPERTIES IMPORTED_LOCATION "${CMAKE_CURRENT_BINARY_DIR}/libcombine_l1.so")
add_dependencies(device device_build)
add_executable(combine_l1 main.cpp)
target_compile_options(combine_l1 PRIVATE -O2 -Wall -Wextra)
target_include_directories(combine_l1 PRIVATE ${inc})
target_link_directories(combine_l1 PRIVATE "${ASCEND_HOME_PATH}/lib64" "${SHMEM_ROOT}/lib")
# S920 may ship tiling_api as a static archive. Its platform references must
# appear BEFORE libplatform on the link line (especially with --as-needed).
target_link_libraries(combine_l1 PRIVATE device ascendcl runtime shmem tiling_api platform opapi nnopbase unified_dlog dl m)
set_target_properties(combine_l1 PROPERTIES BUILD_RPATH "$ORIGIN;${ASCEND_HOME_PATH}/lib64;${SHMEM_ROOT}/lib")
