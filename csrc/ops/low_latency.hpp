#pragma once
#include <pybind11/pybind11.h>
namespace deepep {
void bind_low_latency(pybind11::module_& module);
}
