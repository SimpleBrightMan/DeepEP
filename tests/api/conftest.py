"""Options for explicitly requiring the optional framework compatibility check."""


def pytest_addoption(parser):
    parser.addoption(
        "--require-framework",
        action="store_true",
        help="Fail rather than skip when Torch/torch_npu are unavailable.",
    )
