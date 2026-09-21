#pragma once
#include <acl/acl.h>
#include <cstddef>
#include <cstdint>
#include <string>

namespace deepep {
// The framework owns ACL initialization, device and context. Only this workspace
// and the SHMEM instance are owned here. Destruction is an explicit collective.
class Session {
public:
    Session(int rank, int size, uint64_t bytes, const std::string& endpoint, unsigned timeout);
    Session(const Session&) = delete;
    Session& operator=(const Session&) = delete;
    ~Session() = default;  // Never start collectives from Python GC/interpreter exit.
    void barrier();
    void close();
    bool closed() const
    {
        return closed_;
    }
    uint64_t capacity() const
    {
        return bytes_;
    }
    static bool busy();

private:
    void check_context() const;
    aclrtContext context_ = nullptr;
    void* workspace_ = nullptr;
    uint64_t bytes_ = 0;
    bool closed_ = true;
    bool poisoned_ = false;
};
}  // namespace deepep
