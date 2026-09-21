#pragma once
#include <cstddef>
#include <cstdint>
using aclrtContext = void*;
int aclrtGetCurrentContext(aclrtContext*);
const char* aclrtGetSocName();
int aclrtMemset(void*, size_t, int, size_t);
