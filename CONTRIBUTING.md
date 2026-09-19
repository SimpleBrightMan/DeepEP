# Contributing

Contributions to the Ascend implementation, tests, benchmarks, and documentation are welcome. Start with the [repository layout](docs/repository-layout.md) and [development guide](docs/development.md).

## Scope and ownership

- Preserve the public V1/V2 contract recorded against a pinned upstream revision.
- Place Python wrappers in `deep_ep/`, native responsibilities in `csrc/`, and the complete MegaMoE experiment in `experiments/megamoe/`.
- Keep single-kernel helpers with their implementation. Extract shared device components when concrete consumers and tests justify them.
- Resolve dependency versions in the root build; record the actual selected revisions and environment.

## Changes and evidence

Describe the affected user workflow, supported inputs, build targets, and observable behavior. Include relevant correctness tests, installation checks, and measurements when performance is affected. Record hardware, software versions, topology, command, and timing boundary for device results. Clearly identify checks that were not run.

Documentation changes should preserve working relative links and keep English and Chinese homepages aligned. First-party C++/AscendC files use `.cpp`/`.hpp`. Text uses UTF-8 and LF; see `.editorconfig`.

## Development commands

API contract tests are available; see [tests/README.md](tests/README.md). Native build and device test entry points remain under development; see [docs/build.md](docs/build.md).

## Licensing

A project license has not yet been selected. Preserve copyright and license information when introducing dependency sources. Attribution for adopted interface declarations is recorded in [third-party/README.md](third-party/README.md).
