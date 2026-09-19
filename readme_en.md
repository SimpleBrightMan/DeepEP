<h1 align="center">DeepEP</h1>

<div align="center">

<a href="./README.md">简体中文</a> | English

</div>

<div align="center">

[![Document](https://img.shields.io/badge/Document-DeepEP-blue)](docs/README.md)
[![Platform](https://img.shields.io/badge/Platform-Ascend%20NPU-red)](https://www.hiascend.com/)
[![Kernel](https://img.shields.io/badge/Kernel-AscendC-brightgreen)](csrc/kernels/README.md)
[![Communication](https://img.shields.io/badge/Communication-SHMEM-lightgrey)](https://gitcode.com/cann/shmem)

</div>

An expert-parallel communication library for Ascend NPUs, designed for token dispatch, result combine, and routing layouts in mixture-of-experts workloads.

The planned `deep_ep` package follows DeepEP V1/V2 user workflows. Device kernels use AscendC with SHMEM as the initial communication foundation. MegaMoE Beta is a separately organized experiment for communication and expert-compute fusion.

> **Early development.** Importable API declarations and contract tests are available. All runtime constructors, communication operations, and event operations explicitly raise `NotImplementedError`. Native implementations, installable packages, and runnable communication examples are not yet available. See the [roadmap](docs/roadmap.md).

## Architecture

The target architecture below connects MoE frameworks to Ascend devices through distinct responsibilities: public interfaces, operation semantics and execution preparation, device kernels, and platform dependencies. These modules await implementation and device validation.

<p align="center">
  <a href="figures/architecture_en.svg">
    <img src="figures/architecture_en.svg" alt="DeepEP software architecture" width="800">
  </a>
</p>

Core EP is planned to expose V1/V2 interfaces for token dispatch and combine, which can integrate layout processing, data reordering and quantization. The calling framework handles expert computation. MegaMoE Beta is planned to provide an independent experimental API for a large fused MoE kernel spanning communication, expert computation and activation.

In this design, both paths can share runtime contexts, resources and completion objects, with implementation choices informed by topology and execution policies. SHMEM provides the communication foundation; CATLASS and CatCCOS are selected by the kernels that use them. Solid arrows indicate calls or launches; dashed arrows indicate support or dependencies.

## Planned features

| Area | Target entry | Current status |
| --- | --- | --- |
| V1 EP | `Buffer`; layout, dispatch, combine | Planned |
| V1 low latency | Dedicated dispatch/combine workflows | Planned; separate qualification |
| V2 EP | `ElasticBuffer`, `EPHandle` | Planned |
| Communication/computation overlap | Events and asynchronous options | Planned |
| MegaMoE Beta | `deep_ep_experimental.megamoe` | Planned; optional build |

Compatibility will be checked against a pinned upstream API revision, including imports, arguments, return layouts, handle reuse, and completion behavior. See the [API plan](docs/api.md) and [roadmap](docs/roadmap.md).

## Repository layout

```text
DeepEP/
├── deep_ep/              # Python API families and completion helpers
├── csrc/                 # Bindings, ops, runtime, execution, kernels, device
├── cmake/                # Host/AscendC build and install rules
├── third-party/          # Pinned dependency sources or SDK integration
├── configs/              # Topologies, execution policies, release presets
├── tests/                # API, V1/V2 correctness, integration
├── benchmarks/           # Communication measurements and result records
├── examples/             # Complete public API workflows
├── experiments/megamoe/  # Self-contained optional Beta implementation
├── docs/                 # Architecture, API, build, performance, development
└── figures/              # Documentation diagrams
```

Directory READMEs describe module responsibilities. See the [repository layout](docs/repository-layout.md) for the planned source organization and file responsibilities.

## API workflows

The target V1 workflow is layout preparation → dispatch → expert computation → combine. V2 uses `ElasticBuffer` and `EPHandle`, with expert computation consuming the selected returned layout. Asynchronous consumers must establish the required completion dependency before reading communication results.

The framework or a selected fusion implementation performs expert computation. The signature baseline and target semantics are documented in [docs/api.md](docs/api.md). Data transport and Ascend tuning remain unimplemented.

## Performance

No device measurements are available yet. EP results will report dispatch/combine latency and a stated bandwidth definition. MegaMoE results will report full-workflow latency against a semantically equivalent baseline. Every result must identify the revision, environment, topology, shapes, precision, timing boundary, and reproduction command. See the [measurement protocol](docs/performance.md) and [result template](benchmarks/results/template.csv).

## Requirements and installation

The target environment includes Ascend NPUs, a matched driver/CANN/AscendC stack, SHMEM, Python, PyTorch, and `torch_npu`. Exact versions and device support remain to be validated. CATLASS and CatCCOS are optional dependencies selected by the implementation.

Clone the repository:

```bash
git clone https://gitcode.com/Ascend/DeepEP.git
cd DeepEP
```

Installation is not available yet. See [build and packaging](docs/build.md) for the planned build pipeline, dependencies, and artifacts.

## MegaMoE Beta

[experiments/megamoe/](experiments/megamoe/README.md) owns the planned fused workflow, native extension, Python namespace, configuration, tests, and benchmark entry. The target composition is dispatch–GMM–activation–GMM–combine. Kernel boundaries and optional quantization depend on the selected implementation.

`DEEPEP_BUILD_MEGAMOE` is a planned build option, defaulting to `OFF`. Beta API and performance qualification will be versioned independently of the core API.

## Development and testing

Start with the [architecture](docs/architecture.md), [development guide](docs/development.md), and [contribution guide](CONTRIBUTING.md). The [test plan](tests/README.md) covers API compatibility, numerical correctness, resource ownership, and installation. [Examples](examples/README.md) will demonstrate complete V1/V2 workflows; [benchmarks](benchmarks/README.md) will measure device performance.

## Roadmap

1. Implement the root build, dependency resolution, package layout, and artifact collection.
2. Implement and qualify V1/V2 workflows against a pinned API baseline.
3. Deliver and qualify the optional MegaMoE Beta with reproducible measurements.
4. Extract shared device components where concrete consumers establish a stable contract.

See the [roadmap and qualification matrix](docs/roadmap.md) for completion criteria.

## Contributing and licensing

Contributions should include the affected workflow, build scope, and correctness/performance evidence; see [CONTRIBUTING.md](CONTRIBUTING.md). A project license is still to be selected. Attribution and licensing for the adopted interface declarations are recorded in [third-party/](third-party/README.md).
