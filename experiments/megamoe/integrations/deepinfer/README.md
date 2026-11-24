# DeepInfer empty decode experiment (2026-11-23)

The MegaMoE source in this revision is **R13z1**, built from R13a on top of
`c5d68f0d174be6a0fb81676de560e2193dd11e35`. Rebuild the matching Python and native components together
(API protocol 13). This benchmark used `DEEPEP_MEGAMOE_TIMER=ON`.
The SHA-256 of the private package used for timing was
`b827f64727d1106d452da9082a97935d654ffc990e97399fa9752540df09ad3b`.

[empty-decode-driver.patch](empty-decode-driver.patch) is a separate change for
our DeepInfer **benchmark driver**, not a patch against upstream DeepInfer.
Apply it with `git apply` from the directory containing the original `run_system.py`.
Its LF-normalized SHA-256 must be `52a3a97c628839b729b18d2139e5b24ad948bc65975a489cb4e713ff6dbef78e`.
The resulting benchmark driver has SHA-256 `79d3671f869c3c4fb50d8f13d1d4098cceb4a0e28d87936364dd6abaf40eca35`.
An existing DeepInfer benchmark environment and the R13a backend adapter are
prerequisites; this directory is not a standalone launcher.

Use `empty_decode_policy=baseline` for the control or `skip-global-empty` for
the candidate. The candidate needs CP1, all DP ranks in the same EP group,
ZOS, and the same disabled-for-the-test expert rearrangement setting as the run.
It bypasses the decode graph only when the full group has no eligible model or
cache work, and still advances pending output bookkeeping. It does not change
the default MegaMoE entry point or the engine's default scheduler.

On 16 Ascend NPUs, with concurrency 16 and 256 fixed requests of 128 output
tokens each, the control averaged 768.19 token/s and the candidate averaged
922.92 token/s. These are closed loop results with the same R13z1 package.
Two independent control instances and one candidate instance completed; the
second candidate stopped in process group initialization. Output IDs were
not identical across runs, including the controls. This is performance
exploration, not precision or production validation. The 806.74 token/s
upstream baseline was a separate run on the same nodes.
