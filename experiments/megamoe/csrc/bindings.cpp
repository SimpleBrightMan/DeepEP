// Copyright (c) 2026, Lu Lu
// Modified by SimpleBright_Man 2026

#include "forward.hpp"
#include "shmem.h"
#include "kernels/mega_moe/common/ascend_timer_v2.hpp"
#include "kernels/mega_moe/common/mega_moe_constants.hpp"

namespace py = pybind11;

PYBIND11_MODULE(_C, module)
{
    using deepep::megamoe::Plan;
    module.attr("api_protocol") = 5;
    module.attr("timer_schema_version") = AscendTimer::TIMER_SCHEMA_VERSION;
    module.attr("small_batch_max_tokens") = MegaMoeImpl::W4A8_SMALL_BATCH_MAX_TOKENS;
    module.def("buffer_size", &deepep::megamoe::buffer_size);
    py::class_<Plan>(module, "Plan")
        .def(py::init<torch::Tensor, int64_t, int64_t, int64_t, int64_t, int64_t, bool, int64_t, int64_t, uint64_t,
                      uint64_t, int64_t, bool>())
        .def("forward", &Plan::forward)
        .def_property_readonly("workspace_bytes", &Plan::workspace_bytes)
        .def_property_readonly("prefill_optimization_config", [](const Plan& plan) {
            py::dict result;
            const bool enabled = plan.prefill_optimized();
            result["enabled"] = enabled;
            result["requested"] = plan.prefill_requested();
            result["dispatch_intra_cores"] = enabled ? 4 : 0;
            result["combine_intra_cores"] = enabled ? 4 : 0;
            result["dispatch_buffer_count"] = plan.dispatch_buffer_count();
            // W8A8 has different steady/final depths; it has no fixed profile.
            result["combine_buffer_count"] = plan.is_a8w4() ? py::cast(plan.combine_buffer_count()) : py::none();
            result["dispatch_metadata_mode"] = enabled ? "dma" : "scalar";
            result["group_order"] = enabled ? "paired-xor" : "original";
            result["metadata_bytes"] = plan.prefill_metadata_bytes();
            result["timer_schema_version"] = AscendTimer::TIMER_SCHEMA_VERSION;
            return result;
        });
    module.def("ffts_config", [] { return shmemx_get_ffts_config(); });
    module.attr("timer_numel") = AscendTimer::TOTAL_BUFFER_SIZE;
#ifdef DEEPEP_MEGAMOE_TIMER
    module.attr("timer_enabled") = true;
#else
    module.attr("timer_enabled") = false;
#endif
}
