# Copyright (c) 2026, Lu Lu
# Modified by SimpleBright_Man 2026

find_program(DISPATCH_BISHENG bisheng PATHS "${ASCEND_HOME_PATH}/tools/bisheng_compiler/bin"
    NO_DEFAULT_PATH REQUIRED)
set(dispatch_include
    "${CMAKE_CURRENT_SOURCE_DIR}/csrc"
    "${SHMEM_ROOT}/include" "${SHMEM_ROOT}/src" "${SHMEM_ROOT}/src/device"
    "${ASCEND_HOME_PATH}/include" "${ASCEND_HOME_PATH}/pkg_inc"
    "${ASCEND_HOME_PATH}/asc/include"
    "${ASCEND_HOME_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/asc/include"
    "${ASCEND_HOME_PATH}/include/experiment/runtime" "${ASCEND_HOME_PATH}/pkg_inc/runtime")
set(dispatch_flags
    -O2 -std=c++17 -xasc --npu-arch=dav-3510 --cce-auto-infer-kernel-type=false
    --cce-auto-sync=off --cce-fatobj-link -shared -fPIC
    -Wl,-soname,libdispatch_bandwidth_kernel.so -DL2_CACHE_HINT)
foreach(path IN LISTS dispatch_include)
    list(APPEND dispatch_flags "-I${path}")
endforeach()
set(dispatch_kernel "${CMAKE_CURRENT_BINARY_DIR}/libdispatch_bandwidth_kernel.so")
add_custom_command(OUTPUT "${dispatch_kernel}"
    COMMAND "${DISPATCH_BISHENG}" ${dispatch_flags} csrc/dispatch_bandwidth.cpp
        "-L${ASCEND_HOME_PATH}/lib64" "-L${SHMEM_ROOT}/lib"
        -lruntime -lascendcl -lshmem -lunified_dlog -ldl -lstdc++ -lm -o "${dispatch_kernel}"
    DEPENDS csrc/dispatch_bandwidth.cpp csrc/dispatch_types.hpp csrc/kernel_launch.hpp
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}" VERBATIM)
add_custom_target(dispatch_bandwidth_device DEPENDS "${dispatch_kernel}")
add_library(dispatch_bandwidth_kernel SHARED IMPORTED)
set_target_properties(dispatch_bandwidth_kernel PROPERTIES IMPORTED_LOCATION "${dispatch_kernel}")
add_executable(dispatch_bandwidth_bench csrc/main.cpp)
add_dependencies(dispatch_bandwidth_bench dispatch_bandwidth_device)
target_include_directories(dispatch_bandwidth_bench PRIVATE ${dispatch_include})
target_link_directories(dispatch_bandwidth_bench PRIVATE "${ASCEND_HOME_PATH}/lib64")
target_link_libraries(dispatch_bandwidth_bench PRIVATE dispatch_bandwidth_kernel shmem ascendcl ascend_runtime
    tiling_api platform unified_dlog ${CMAKE_DL_LIBS})
target_compile_options(dispatch_bandwidth_bench PRIVATE -O2 -Wall -Wextra -Werror -fstack-protector-strong)
target_link_options(dispatch_bandwidth_bench PRIVATE -Wl,-z,relro,-z,now,-z,noexecstack)
set_target_properties(dispatch_bandwidth_bench PROPERTIES BUILD_WITH_INSTALL_RPATH TRUE INSTALL_RPATH "$ORIGIN")
