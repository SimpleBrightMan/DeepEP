#include "ops/dispatch.hpp"
#include <pybind11/stl.h>
#include <stdexcept>
#include <vector>
#include "kernels/dispatch/tiling.hpp"
#include "kernels/dispatch/launch.hpp"
#include "runtime/session.hpp"

namespace py = pybind11;
namespace deepep {
namespace {
ascend_deepep::DispatchTiling layout(int rank, int world, int tokens, int topk, int experts, int rows, bool fp8)
{
    if (world < 2 || world > 128 || (world & (world - 1)) || rank < 0 || rank >= world || tokens < 1 ||
        tokens > 10240 || topk < 1 || topk > 16 || experts < world || experts > 1024 || experts % world || rows < 1 ||
        uint64_t(rows) > uint64_t(world) * tokens * topk || uint64_t(world) * world * tokens * topk >= (1ULL << 31)) {
        throw std::invalid_argument("Invalid dispatch shape (EP2..128, H7168, T1..10240, K1..16, E<=1024)");
    }
    auto t = ascend_deepep::MakeDispatchTiling(rank, world, tokens, topk, experts, rows, fp8);
    if (uint64_t(world) * t.address_stride >= (1ULL << 31) || t.workspace_bytes > (32ULL << 30)) {
        throw std::invalid_argument("Dispatch address encoding or workspace capacity overflow");
    }
    return t;
}

using Shape = std::vector<int64_t>;
uint8_t* pointer(py::handle tensor, const Shape& shape, const char* dtype, int device)
{
    auto torch = py::module_::import("torch");
    if (!torch.attr("is_tensor")(tensor).cast<bool>()) {
        throw std::invalid_argument("Dispatch requires framework tensors");
    }
    auto obj = py::reinterpret_borrow<py::object>(tensor);
    if (obj.attr("device").attr("type").cast<std::string>() != "npu" ||
        obj.attr("device").attr("index").cast<int>() != device || !obj.attr("is_contiguous")().cast<bool>() ||
        obj.attr("shape").cast<Shape>() != shape || !obj.attr("dtype").is(torch.attr(dtype))) {
        throw std::invalid_argument("Dispatch tensor shape, dtype, device or contiguity mismatch");
    }
    return reinterpret_cast<uint8_t*>(obj.attr("data_ptr")().cast<uintptr_t>());
}
void copy(void* dst, const void* src, uint64_t bytes, aclrtStream stream)
{
    if (bytes && aclrtMemcpyAsync(dst, bytes, src, bytes, ACL_MEMCPY_DEVICE_TO_DEVICE, stream) != 0) {
        throw std::runtime_error("Dispatch workspace copy failed; restart the distributed job");
    }
}
}  // namespace

void bind_dispatch(py::module_& module)
{
    module.def("dispatch_layout", [](int rank, int world, int tokens, int topk, int experts, int rows, bool fp8) {
        const auto t = layout(rank, world, tokens, topk, experts, rows, fp8);
        py::dict result;
        result["workspace_bytes"] = t.workspace_bytes;
        result["address_stride"] = t.address_stride;
        result["sync"] = t.sync;
        result["book"] = t.book;
        result["input"] = t.input;
        result["output"] = t.output;
        result["weights"] = t.weights;
        result["scales"] = t.scales;
        result["inbox"] = t.inbox;
        result["inbox_stride"] = t.inbox_stride;
        return result;
    });
    module.def("dispatch", [](Session& session, py::dict tensors, int tokens, int topk, int experts, int rows,
                              int local_rows, bool fp8, uintptr_t stream_address) {
        const auto t = layout(session.rank(), session.world(), tokens, topk, experts, rows, fp8);
        auto* workspace = static_cast<uint8_t*>(session.workspace());
        if (!session.udma() || t.workspace_bytes > session.capacity() / 2U || stream_address == 0 || local_rows < 0 ||
            local_rows > rows) {
            throw std::invalid_argument(
                "Dispatch needs a MTE+URMA session, sufficient workspace and valid stream/rows");
        }
        int32_t device = -1;
        if (aclrtGetDevice(&device) != 0) {
            throw std::runtime_error("Cannot obtain current ACL device");
        }
        const char* dtype = fp8 ? "float8_e4m3fn" : "bfloat16";
        auto* input = pointer(tensors["input"], {tokens, 7168}, dtype, device);
        auto* output = pointer(tensors["output"], {local_rows, 7168}, dtype, device);
        auto* destinations = pointer(tensors["dst"], {tokens, topk}, "int32", device);
        auto* forward = pointer(tensors["forward_list"], {session.world(), int64_t(tokens) * topk, 6}, "int32", device);
        auto* counts = pointer(tensors["forward_counts"], {session.world()}, "int32", device);
        uint8_t *weights = nullptr, *out_weights = nullptr, *scales = nullptr, *out_scales = nullptr;
        if (!tensors["weights"].is_none()) {
            weights = pointer(tensors["weights"], {tokens, topk}, "float32", device);
            out_weights = pointer(tensors["output_weights"], {local_rows}, "float32", device);
        }
        if (fp8) {
            scales = pointer(tensors["scales"], {tokens, 56}, "float32", device);
            out_scales = pointer(tensors["output_scales"], {local_rows, 56}, "float32", device);
        }
        auto stream = reinterpret_cast<aclrtStream>(stream_address);
        const uint64_t row_bytes = fp8 ? 7168 : 14336;
        py::gil_scoped_release release;
        // URMA always reads registered SHMEM, never an unregistered framework allocation.
        copy(workspace + t.input, input, uint64_t(tokens) * row_bytes, stream);
        auto config = t;
        dispatch_kernel_do(stream, workspace, destinations, forward, counts, weights, scales,
                           reinterpret_cast<uint8_t*>(&config));
        // Own framework storage survives reuse/destroy of the symmetric workspace.
        copy(output, workspace + t.output, uint64_t(local_rows) * row_bytes, stream);
        if (weights) {
            copy(out_weights, workspace + t.weights, uint64_t(local_rows) * 4, stream);
        }
        if (fp8) {
            copy(out_scales, workspace + t.scales, uint64_t(local_rows) * 224, stream);
        }
    });
}
}  // namespace deepep
