// Copyright (c) 2026, SimpleBrightMan
//
// Host wrapper for the MTE All-to-All microbenchmark. The kernel only moves
// payload bytes; routing, timing and correctness checks stay in Python.

#include "ops/alltoall.hpp"
#include <pybind11/stl.h>
#include <acl/acl.h>
#include <stdexcept>
#include <string>
#include <vector>
#include "host/data_plane/shmem_host_rma.h"
#include "kernels/alltoall/launch.hpp"
#include "runtime/session.hpp"

namespace py = pybind11;
namespace deepep {
namespace {

constexpr uint32_t kMinTokenBytes = 32U;
constexpr uint32_t kMaxTokenBytes = 64U * 1024U;
constexpr uint32_t kMaxTokens = 16384U;
constexpr uint32_t kMaxCores = 64U;

uint8_t* tensor_pointer(py::handle tensor, const std::vector<int64_t>& shape, int device)
{
    auto torch = py::module_::import("torch");
    if (!torch.attr("is_tensor")(tensor).cast<bool>()) {
        throw std::invalid_argument("All-to-All requires framework tensors");
    }
    auto obj = py::reinterpret_borrow<py::object>(tensor);
    if (obj.attr("device").attr("type").cast<std::string>() != "npu" ||
        obj.attr("device").attr("index").cast<int>() != device || !obj.attr("is_contiguous")().cast<bool>() ||
        obj.attr("shape").cast<std::vector<int64_t>>() != shape ||
        !obj.attr("dtype").is(torch.attr("uint8"))) {
        throw std::invalid_argument("All-to-All tensor shape, dtype, device or contiguity mismatch");
    }
    return reinterpret_cast<uint8_t*>(obj.attr("data_ptr")().cast<uintptr_t>());
}

uint64_t required_bytes(uint32_t world, uint32_t tokens, uint32_t token_bytes)
{
    if (world < 2U || world > 128U || tokens < 1U || tokens > kMaxTokens || token_bytes < kMinTokenBytes ||
        token_bytes > kMaxTokenBytes || token_bytes % 32U != 0U) {
        throw std::invalid_argument("All-to-All requires EP2..128, T1..16384, B32..65536 aligned to 32 bytes");
    }
    return uint64_t(world) * uint64_t(tokens) * uint64_t(token_bytes);
}

int current_device()
{
    int device = -1;
    if (aclrtGetDevice(&device) != 0) {
        throw std::runtime_error("Cannot obtain current ACL device");
    }
    return device;
}

}  // namespace

void bind_alltoall(py::module_& module)
{
    module.def("alltoall_workspace_bytes",
               [](uint32_t world, uint32_t tokens, uint32_t token_bytes) {
                   return required_bytes(world, tokens, token_bytes);
               },
               py::arg("world"), py::arg("tokens"), py::arg("token_bytes"));
    module.def("alltoall_peers_ready", [](Session& session) {
        const int world = session.world();
        auto* workspace = static_cast<uint8_t*>(session.workspace());
        for (int peer = 0; peer < world; ++peer) {
            if (peer != session.rank() && aclshmem_ptr(workspace, peer) == nullptr) {
                return false;
            }
        }
        return true;
    });
    module.def(
        "alltoall",
        [](Session& session, py::object send, uint32_t tokens, uint32_t token_bytes, uint32_t group, uint32_t mode,
           uint32_t cores, uintptr_t stream_address) {
            const uint32_t world = static_cast<uint32_t>(session.world());
            if (session.closed()) throw std::invalid_argument("All-to-All requires a live runtime");
            if (group == 0U || world % group != 0U || mode > 3U || cores == 0U || cores > kMaxCores) {
                throw std::invalid_argument("All-to-All requires group|world, mode 0..3 and cores 1..64");
            }
            const uint64_t needed = required_bytes(world, tokens, token_bytes);
            if (needed > session.capacity()) {
                throw std::invalid_argument("All-to-All payload exceeds the symmetric workspace");
            }
            if (stream_address == 0) throw std::invalid_argument("All-to-All requires a valid NPU stream");
            auto* input = tensor_pointer(send, {world, tokens, token_bytes}, current_device());
            auto* workspace = static_cast<uint8_t*>(session.workspace());
            alltoall_mte_kernel_do(reinterpret_cast<void*>(stream_address), input, workspace, session.rank(), world,
                                   group, token_bytes, tokens, mode, cores);
        },
        py::arg("session"), py::arg("send"), py::arg("tokens"), py::arg("token_bytes"), py::arg("group"),
        py::arg("mode"), py::arg("cores"), py::arg("stream"));
    module.def(
        "alltoall_drain",
        [](Session& session, py::object out, uint32_t tokens, uint32_t token_bytes, uint32_t cores,
           uintptr_t stream_address) {
            const uint32_t world = static_cast<uint32_t>(session.world());
            if (session.closed()) throw std::invalid_argument("All-to-All drain requires a live runtime");
            if (cores == 0U || cores > kMaxCores) throw std::invalid_argument("All-to-All drain cores 1..64");
            const uint64_t needed = required_bytes(world, tokens, token_bytes);
            if (needed > session.capacity()) {
                throw std::invalid_argument("All-to-All payload exceeds the symmetric workspace");
            }
            if (stream_address == 0) throw std::invalid_argument("All-to-All drain requires a valid NPU stream");
            auto* output = tensor_pointer(out, {world, tokens, token_bytes}, current_device());
            alltoall_mte_drain_kernel_do(reinterpret_cast<void*>(stream_address),
                                         static_cast<uint8_t*>(session.workspace()), output, needed, cores);
        },
        py::arg("session"), py::arg("out"), py::arg("tokens"), py::arg("token_bytes"), py::arg("cores"),
        py::arg("stream"));
}

}  // namespace deepep
