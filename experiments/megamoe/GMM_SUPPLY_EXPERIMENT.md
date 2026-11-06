# Combined consumer experiment v2

This branch now combines A-window prefetch, B4 L1 supply, independent AIV0 UB input4/output3, and paired AIV1 FIX completion.

GMM1 uses physical AIC 27-(global logical tile%28), including shared GMM1. GMM2 keeps global tile%28. AIV0 and the routed completion cursor use exactly the same assignment as AIC. Routed activation follows the reversed GMM1 owner; shared activation retains its original distributed M128 task scheduler.

Every AIV1 reorders its paired AIC's GMM2 N256 tile after real FIX completion. Two M128 UB slots copy across source/CV segments; remote output goes to source-major gmm2Output and self output directly to returnY. Only after the MTE3 stores drain does it increment gmm2Done. Communication owners no longer copy Reorder payloads: they inspect the completed-group prefix, PUT128 rows when ready and publish SO per CV. Self CV publication and final incoming-finished accounting retain their existing owners.

Cooperative priority: bounded communication -> FIX completion consumption -> input-scale gather -> routed activation -> paired Reorder -> Pack -> shared activation. Pack is closed before low-UB/event reuse; counters in high UB become live only after this communication owner's Pack is complete. Reorder may run while other Pack quotas remain.

L1: A2x64KiB + B4x64KiB + scaleA2x32KiB + scaleB2x32KiB =512KiB. AIV0 UB: input4x16KiB + output3x32KiB =160KiB. Cross-core B full4..7, empty8..11; context retirement2; routed AIV1 full12/13, empty14/15. A prefetch retains the new64KiB A-window capacity bound.

Fine tile bars remain submission scopes. AIV1 completion bars are actual FIX consumption; Reorder bars include its final GM store drain. Use formal round-max-mean for performance, coarse GMM for the compute envelope. Reverse ownership balances the remainder in opposite directions, not an assurance that runtimes become identical.
