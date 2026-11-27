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
    using deepep::megamoe::TpPlan;
    module.attr("api_protocol") = 13;
    module.attr("small_batch_max_tokens") = MegaMoeImpl::W4A8_SMALL_BATCH_MAX_TOKENS;
    module.attr("paired_gate_up_max_tokens") = MegaMoeImpl::W4A8_PAIRED_GATE_UP_MAX_TOKENS;
    module.def("buffer_size", &deepep::megamoe::buffer_size);
    module.def("tp_buffer_size", &deepep::megamoe::tp_buffer_size, py::arg("tp_world"), py::arg("tokens"),
               py::arg("topk"), py::arg("hidden"),
               py::arg("rs_tokens_per_chunk") = MegaMoeImpl::TP_DEFAULT_RS_TOKENS_PER_CHUNK);
    py::class_<TpPlan>(module, "TpPlan")
        .def(py::init<torch::Tensor, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, uint64_t,
                      uint64_t, int64_t, bool, int64_t, int64_t, int64_t, c10::optional<torch::Tensor>, bool,
                      int64_t>(),
             py::arg("example"), py::arg("global_intermediate"), py::arg("shared_intermediate"), py::arg("topk"),
             py::arg("num_experts"), py::arg("tp_world"), py::arg("max_tokens"), py::arg("shared_experts"),
             py::arg("ffts"), py::arg("symmetric_address"), py::arg("symmetric_bytes"),
             py::arg("route_weight_before_quant") = false,
             py::arg("m_groups_per_wave") = MegaMoeImpl::TP_DEFAULT_M_GROUPS_PER_WAVE,
             py::arg("ag_cores_per_peer") = MegaMoeImpl::TP_DEFAULT_AG_CORES_PER_PEER,
             py::arg("rs_tokens_per_chunk") = MegaMoeImpl::TP_DEFAULT_RS_TOKENS_PER_CHUNK,
             py::arg("workspace_hint") = c10::nullopt, py::arg("use_urma_allgather") = true,
             py::arg("shmem_instance_id") = 0)
        .def("forward", &TpPlan::forward, py::arg("x"), py::arg("x_scales"), py::arg("topk_ids"),
             py::arg("topk_weights"), py::arg("weight1"), py::arg("weight2"), py::arg("weight_scales1"),
             py::arg("weight_scales2"), py::arg("timer"), py::arg("shared_weight1"), py::arg("shared_weight2"),
             py::arg("shared_weight_scales1"), py::arg("shared_weight_scales2"), py::arg("activation_clamp"),
             py::arg("out"), py::arg("route_weight_before_quant") = c10::nullopt)
        .def_property_readonly("workspace_bytes", &TpPlan::workspace_bytes)
        .def_property_readonly("workspace_tensor", &TpPlan::workspace_tensor);
    py::class_<Plan>(module, "Plan")
        .def(py::init<torch::Tensor, int64_t, int64_t, int64_t, int64_t, int64_t, bool, int64_t, int64_t, uint64_t,
                      uint64_t, int64_t>())
        .def(py::init<torch::Tensor, int64_t, int64_t, int64_t, int64_t, int64_t, bool, int64_t, int64_t, uint64_t,
                      uint64_t, int64_t, int64_t>())
        .def(py::init<torch::Tensor, int64_t, int64_t, int64_t, int64_t, int64_t, bool, int64_t, int64_t, uint64_t,
                      uint64_t, int64_t, int64_t, bool>())
        .def(py::init<torch::Tensor, int64_t, int64_t, int64_t, int64_t, int64_t, bool, int64_t, int64_t, uint64_t,
                      uint64_t, int64_t, int64_t, bool, bool>())
        .def(py::init<torch::Tensor, int64_t, int64_t, int64_t, int64_t, int64_t, bool, int64_t, int64_t, uint64_t,
                      uint64_t, int64_t, int64_t, bool, bool, c10::optional<torch::Tensor>>())
        .def("forward", &Plan::forward)
        .def("enable_compute_replay", &Plan::enable_compute_replay)
        .def("prepare_local_compute", &Plan::prepare_local_compute)
        .def("local_compute_output", &Plan::local_compute_output)
        .def_property_readonly("workspace_bytes", &Plan::workspace_bytes)
        .def_property_readonly("workspace_tensor", &Plan::workspace_tensor);
    module.def("ffts_config", [] { return shmemx_get_ffts_config(); });
    module.attr("timer_numel") = AscendTimer::TOTAL_BUFFER_SIZE;
#ifdef DEEPEP_MEGAMOE_TIMER
    module.attr("timer_enabled") = true;
#else
    module.attr("timer_enabled") = false;
#endif
}
