# URMA dimension contract

H (hidden) accepts multiples of 256 in [256,8192]; routed I (intermediate_hidden)
accepts multiples of 256 in [256,16384]. EP4/8/16, E384 and TopK8 remain unchanged.
These limits describe supported layouts, not a guarantee that every maximum-size
combination fits the available HBM. Allocation failures remain explicit.

The benchmark accepts --hidden and --intermediate. Native/Python/host-plan
validation agree; weights retain their real H/I shapes. Pack has its existing
8192-byte capacity. Input-scale gather uses H/32 bytes and an aligned UB stride.
H3584 selects the original fixed-width scale-gather specialization and original
Reduce implementation. Other widths reduce in <=2048-column slices with the same
six-route/two-output pipeline. No GMM tile, routing order or communication batching
change is introduced by the dimension commit. Rebuild the full wheel.
