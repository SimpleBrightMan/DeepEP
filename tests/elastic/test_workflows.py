"""Pending V2 semantic and generalized BF16/FP8 device qualification."""

import pytest

from tests.utils.cases import ROUTES, make_case
from tests.utils.device import buffer_for, run_cached, run_operation
from tests.utils.matrix import DEVICE_MATRIX, validate_device_spec

pytestmark = pytest.mark.device


@pytest.mark.parametrize("name", ROUTES)
@pytest.mark.parametrize("expanded", [False, True], ids=["ordinary", "expanded"])
@pytest.mark.parametrize("with_weights", [False, True])
@pytest.mark.parametrize(
    "dtype", ["bfloat16", "float8_e4m3fn"], ids=["bf16", "fp8-e4m3fn"]
)
@pytest.mark.parametrize("operation", ["dispatch", "combine", "roundtrip"])
def test_elastic(device_session, name, expanded, with_weights, dtype, operation):
    if device_session[2].world_size != 2:
        pytest.skip(
            "two-rank semantic scenarios are covered separately from the EP matrix"
        )
    case = make_case(name, hidden=256, with_weights=with_weights, dtype=dtype)
    with buffer_for(device_session, case, "elastic") as buffer:
        run_operation(device_session, buffer, case, "elastic", expanded, operation)


@pytest.mark.parametrize("spec", DEVICE_MATRIX, ids=lambda spec: spec["name"])
def test_elastic_qualification_matrix(device_session, spec):
    if spec["world_size"] != device_session[2].world_size:
        pytest.skip("case belongs to a different torchrun world size")
    case = validate_device_spec(spec)
    with buffer_for(device_session, case, "elastic") as buffer:
        run_operation(
            device_session,
            buffer,
            case,
            "elastic",
            spec["expanded"],
            spec["operation"],
        )


@pytest.mark.parametrize("expanded", [False, True], ids=["ordinary", "expanded"])
def test_cached_dispatch_inherits_layout(device_session, expanded):
    if device_session[2].world_size != 2:
        pytest.skip("cached semantic scenario uses the two-rank hand-checkable routes")
    case = make_case("normal", hidden=256, with_weights=False)
    with buffer_for(device_session, case, "elastic") as buffer:
        run_cached(device_session, buffer, case, "elastic", expanded)
