// Test-only host state double; production compiles against the pinned SDK header.
#pragma once
#include <cstdint>
struct aclshmem_device_host_state_t {
    int mype = 0;
    int npes = 0;
    uint8_t topo_list[512]{};
};
