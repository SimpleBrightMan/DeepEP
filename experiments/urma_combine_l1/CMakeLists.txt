cmake_minimum_required(VERSION 3.20)
project(urma_combine_l1 LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
find_program(BISHENG bisheng PATHS "${ASCEND_HOME_PATH}/tools/bisheng_compiler/bin" NO_DEFAULT_PATH REQUIRED)
set(inc "${ASCEND_HOME_PATH}/include" "${ASCEND_HOME_PATH}/pkg_inc"
    "${ASCEND_HOME_PATH}/include/experiment/runtime" "${ASCEND_HOME_PATH}/pkg_inc/runtime"
    "${SHMEM_ROOT}/include" "${SHMEM_ROOT}/src" "${SHMEM_ROOT}/src/device" "${SHMEM_ROOT}/src/device_simt")
set(flags -O2 -std=c++17 -xasc --npu-arch=dav-3510 --cce-auto-infer-kernel-type=false
    --cce-auto-sync=off --cce-fatobj-link -shared -fPIC -DL2_CACHE_HINT)
foreach(p IN LISTS inc)
    list(APPEND flags "-I${p}")
endforeach()
file(GLOB protocol_headers "${CMAKE_CURRENT_SOURCE_DIR}/../megamoe/csrc/kernels/mega_moe/urma/*.hpp")
add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/libcombine_l1.so"
    COMMAND "${BISHENG}" ${flags} "${CMAKE_CURRENT_SOURCE_DIR}/kernel.cpp"
    "-L${ASCEND_HOME_PATH}/lib64" "-L${SHMEM_ROOT}/lib" -lruntime -lascendcl -lshmem -lstdc++ -ldl
    -o "${CMAKE_CURRENT_BINARY_DIR}/libcombine_l1.so"
    DEPENDS kernel.cpp layout.hpp ${protocol_headers} VERBATIM)
add_custom_target(device_build DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/libcombine_l1.so")
add_library(device SHARED IMPORTED)
set_target_properties(device PROPERTIES IMPORTED_LOCATION "${CMAKE_CURRENT_BINARY_DIR}/libcombine_l1.so")
add_dependencies(device device_build)
add_executable(combine_l1 main.cpp)
target_compile_options(combine_l1 PRIVATE -O2 -Wall -Wextra)
target_include_directories(combine_l1 PRIVATE ${inc})
target_link_directories(combine_l1 PRIVATE "${ASCEND_HOME_PATH}/lib64" "${SHMEM_ROOT}/lib")
target_link_libraries(combine_l1 PRIVATE device ascendcl runtime shmem platform tiling_api opapi nnopbase unified_dlog dl m)
set_target_properties(combine_l1 PROPERTIES BUILD_RPATH "$ORIGIN;${ASCEND_HOME_PATH}/lib64;${SHMEM_ROOT}/lib")
