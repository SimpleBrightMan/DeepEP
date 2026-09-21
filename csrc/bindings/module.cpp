#include <pybind11/pybind11.h>
#include <dlfcn.h>
#include "host/init/shmem_host_init.h"
#include "runtime/session.hpp"

namespace py = pybind11;
PYBIND11_MODULE(_C, module)
{
    module.attr("runtime_protocol") = 1;
    module.def("shmem_library_path", []() {
        Dl_info info{};
        if (dladdr(reinterpret_cast<void*>(&aclshmemx_init_attr), &info) == 0 || info.dli_fname == nullptr) {
            throw std::runtime_error("Cannot identify the loaded SHMEM library");
        }
        return std::string(info.dli_fname);
    });
    module.def("runtime_busy", &deepep::Session::busy, py::call_guard<py::gil_scoped_release>());
    py::class_<deepep::Session>(module, "Runtime")
        .def(py::init<int, int, uint64_t, const std::string&, unsigned>(), py::call_guard<py::gil_scoped_release>())
        .def("barrier", &deepep::Session::barrier, py::call_guard<py::gil_scoped_release>())
        .def("close", &deepep::Session::close, py::call_guard<py::gil_scoped_release>())
        .def_property_readonly("closed", &deepep::Session::closed)
        .def_property_readonly("capacity", &deepep::Session::capacity);
}
