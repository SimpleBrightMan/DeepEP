// Test-only topology constants; not a substitute for SDK ABI verification.
#pragma once
#include <cstdint>
enum aclshmem_transport_t : uint8_t {
    ACLSHMEM_TRANSPORT_MTE = 1 << 0,
    ACLSHMEM_TRANSPORT_UDMA = 1 << 3,
};
