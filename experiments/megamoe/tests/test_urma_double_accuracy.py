# Copyright (c) 2026, Lu Lu
# Modified by SimpleBright_Man 2026

"""CPU-only acceptance policy tests for the URMA three-way accuracy check."""

import json

import numpy as np
import pytest

from deep_ep_experimental.megamoe.accuracy import TRIANGLE_RATIOS, compare_triangle, triangle_metrics
from deep_ep_experimental.megamoe.urma_accuracy import compare_outputs


def rows_by_name(rows):
    return {row["metric"]: row for row in rows}


def test_triangle_ratios_match_reference_contract():
    assert TRIANGLE_RATIOS == {
        "max_relative": 5.0,
        "mean_relative": 1.5,
        "sum_relative": 2.0,
        "rmse": 1.5,
        "small_bad_elements": 2.0,
    }


def test_relative_threshold_is_inclusive_for_both_signs():
    below = np.nextafter(0.001, 0.0)
    expected = np.array([0.001, -0.001, below, -below, 0.0])
    actual = expected + np.array([0.0001, 0.0002, 0.1, 0.1, 0.1])
    metrics = triangle_metrics(actual, expected)
    assert metrics["small_elements"] == 3
    assert metrics["small_bad_elements"] == 3
    assert metrics["max_relative"] == pytest.approx(0.2)
    assert metrics["mean_relative"] == pytest.approx(0.15)
    assert metrics["sum_relative"] == pytest.approx(0.3)
    assert metrics["rmse"] == pytest.approx(np.sqrt(np.mean((actual - expected) ** 2)))


def test_small_bad_absolute_threshold_is_strict():
    actual = np.array([1e-5, -1e-5, np.nextafter(1e-5, np.inf), -np.nextafter(1e-5, np.inf)])
    metrics = triangle_metrics(actual, np.zeros(4))
    assert metrics["small_bad_elements"] == 2
    assert metrics["small_elements"] == 4
    assert metrics["max_relative"] == metrics["mean_relative"] == metrics["sum_relative"] == 0.0


def test_exact_cascade_has_exact_zero_budgets():
    expected = np.ones((2, 3), dtype=np.float64)
    actual = expected.copy()
    # This is smaller than FP32 rounding: a fixed FP32 diagnostic can pass,
    # but a nonzero FP64 triangle error must still fail the zero budget.
    actual[0, 0] += 2**-40
    result, rows = compare_triangle(actual, expected.copy(), expected)
    assert result["cpu_mixed_passed"]
    assert not result["passed"]
    assert all(row["limit"] == 0 for row in rows)
    assert not rows_by_name(rows)["rmse"]["passed"]


def test_triangle_includes_budget_boundary():
    expected = np.ones((2, 3))
    cascade = expected + 0.125
    actual = expected + 0.1875
    result, rows = compare_triangle(actual, cascade, expected)
    by_name = rows_by_name(rows)
    assert result["passed"]
    for name in ("mean_relative", "rmse"):
        assert by_name[name]["mega_vs_cpu"] == by_name[name]["limit"]
        assert by_name[name]["passed"]


def test_fixed_cpu_threshold_is_diagnostic_only():
    expected = np.ones((2, 3))
    cascade = expected + 0.5
    result, rows = compare_triangle(cascade.copy(), cascade, expected)
    assert not result["cpu_mixed_passed"]
    assert result["passed"]
    assert all(row["passed"] for row in rows if row["enabled"])


def test_fixed_cpu_pass_does_not_override_triangle_failure():
    expected = np.ones((2, 3))
    result, rows = compare_triangle(expected + 0.002, expected + 0.001, expected)
    assert result["cpu_mixed_passed"]
    assert not result["passed"]
    assert not rows_by_name(rows)["mean_relative"]["passed"]


@pytest.mark.parametrize("count, enabled, passed", [(99999, False, True), (100000, True, False)])
def test_small_element_budget_starts_at_100000(count, enabled, passed):
    expected = np.zeros(count)
    cascade, actual = expected.copy(), expected.copy()
    cascade[:100] = 2e-5
    actual[:201] = 2e-5
    result, rows = compare_triangle(actual, cascade, expected)
    row = rows_by_name(rows)["small_bad_elements"]
    assert row["enabled"] is enabled
    assert row["passed"] is (False if enabled else None)
    assert row["limit"] == 200
    assert row["mega_vs_cpu"] == 201
    assert result["passed"] is passed
    assert all(item["passed"] for item in rows if item["enabled"] and item["metric"] != "small_bad_elements")


def test_small_element_budget_includes_boundary():
    expected = np.zeros(100000)
    cascade, actual = expected.copy(), expected.copy()
    cascade[:100] = 2e-5
    actual[:200] = 2e-5
    result, rows = compare_triangle(actual, cascade, expected)
    assert result["passed"]
    row = rows_by_name(rows)["small_bad_elements"]
    assert row["passed"]
    assert row["mega_vs_cpu"] == row["limit"] == 200


@pytest.mark.parametrize("which", [0, 1, 2])
@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_triangle_rejects_every_nonfinite_operand(which, value):
    values = [np.ones((2, 3)) for _ in range(3)]
    values[which][0, 0] = value
    result, rows = compare_triangle(*values)
    assert not result["passed"]
    assert not rows
    assert "non-finite" in result["reason"]
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("which", [0, 1, 2])
def test_triangle_rejects_shapes_that_numpy_would_broadcast(which):
    values = [np.ones((2, 3)) for _ in range(3)]
    values[which] = np.ones((2, 1))
    result, rows = compare_triangle(*values)
    assert not result["passed"]
    assert not rows
    assert "shape" in result["reason"]


@pytest.fixture
def outputs():
    expected = np.arange(1, 13, dtype=np.float32).reshape(4, 3)
    return dict(
        actual=expected.copy(),
        cascade=expected.copy(),
        expected=expected.copy(),
        actual_counts=np.array([3, 5], dtype=np.int64),
        cascade_counts=np.array([3, 5], dtype=np.int64),
        expected_counts=np.array([3, 5], dtype=np.int64),
    )


def test_full_output_report_can_be_sent_through_json_store(outputs):
    report = compare_outputs(**outputs)
    assert report["passed"]
    assert report["finite"]
    assert report["counts_equal"] and report["cascade_counts_equal"]
    assert report["triangle"]["passed"]
    assert len(report["triangle_metrics"]) == 5
    json.dumps(report, allow_nan=False)


def test_output_helper_uses_triangle_verdict_not_fixed_threshold(outputs):
    outputs["cascade"] += 0.5
    outputs["actual"] += 0.5
    report = compare_outputs(**outputs)
    assert report["passed"]
    assert not report["triangle"]["cpu_mixed_passed"]


def test_selected_rows_match_the_cached_golden_order(outputs):
    selected = np.array([3, 1], dtype=np.int64)
    outputs["expected"] = outputs["expected"][selected]
    # Unselected finite rows may differ during an explicitly sampled check.
    outputs["actual"][0] += 3
    report = compare_outputs(**outputs, selected=selected)
    assert report["passed"]
    assert report["finite"]


@pytest.mark.parametrize("which", ["actual", "cascade"])
@pytest.mark.parametrize("value", [np.nan, np.inf])
def test_full_output_finiteness_is_checked_outside_selected_rows(outputs, which, value):
    outputs["expected"] = outputs["expected"][[1, 3]]
    outputs[which][0, 0] = value
    report = compare_outputs(**outputs, selected=np.array([1, 3], dtype=np.int64))
    assert not report["passed"]
    assert not report["finite"]
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("which", ["actual_counts", "cascade_counts"])
def test_each_count_vector_must_match_cpu_golden(outputs, which):
    outputs[which][0] += 1
    report = compare_outputs(**outputs)
    assert not report["passed"]
    assert report["triangle"]["passed"]
    key = "counts_equal" if which == "actual_counts" else "cascade_counts_equal"
    assert not report[key]


@pytest.mark.parametrize("which", ["actual_counts", "cascade_counts", "expected_counts"])
@pytest.mark.parametrize("invalid", [np.array([3.0, 5.0]), np.array([3, -5]), np.array([[3, 5]])])
def test_malformed_count_vectors_cannot_pass(outputs, which, invalid):
    outputs[which] = invalid
    report = compare_outputs(**outputs)
    assert not report["passed"]
    json.dumps(report, allow_nan=False)


def test_identical_negative_counts_still_fail(outputs):
    for name in ("actual_counts", "cascade_counts", "expected_counts"):
        outputs[name] = np.array([3, -1], dtype=np.int64)
    assert not compare_outputs(**outputs)["passed"]


@pytest.mark.parametrize("which", ["actual", "cascade"])
def test_full_output_shape_mismatch_cannot_be_hidden_by_selection(outputs, which):
    outputs["expected"] = outputs["expected"][[0, 1]]
    outputs[which] = outputs[which][:3]
    report = compare_outputs(**outputs, selected=np.array([0, 1], dtype=np.int64))
    assert not report["passed"]


@pytest.mark.parametrize(
    "selected",
    [np.array([-1]), np.array([4]), np.array([0.0]), np.array([[0]]), np.array([], dtype=int), np.array([0, 0])],
)
def test_invalid_selected_indices_fail_as_report(outputs, selected):
    outputs["expected"] = outputs["expected"][:1]
    report = compare_outputs(**outputs, selected=selected)
    assert not report["passed"]
    json.dumps(report, allow_nan=False)
