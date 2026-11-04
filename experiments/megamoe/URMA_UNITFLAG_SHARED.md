# Unitflag dimensions and shared migration

Base: origin/megamoe-urma-unitflag-so b904c14. Commit one generalizes H (256..8192,
multiple256), routed I (256..16384, multiple256), and TopK1..8, E384, EP4/8/16.
CLI: --hidden / --intermediate / --topk. Shared experts remain disabled in this commit.

Keep dense unitflag GMM2 output, peer-owned span reorder, 128-row Combine PUTs,
CV SO epoch flags and self-reorder readiness. The generic reducer MUST read these
per-CV epochs, not the comm-first branch's bitmap flags. Original H3584/K8 reducer
remains selected. Other K uses exact-byte metadata DMA; H is sliced into 2048 columns.
Input-scale gather uses H/32 bytes with aligned UB strides. Reorder already uses
full H rows in two bounded 64KiB slots. No performance claim without device testing.
