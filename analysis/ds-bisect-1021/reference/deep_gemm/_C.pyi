# Stubs for module: _C

from typing import Any, Optional
import torch


def aclnn_fp8_fp4_gemm_nt(
    a: tuple[torch.Tensor, torch.Tensor],
    b: tuple[torch.Tensor, torch.Tensor],
    d: torch.Tensor,
    c: Optional[torch.Tensor] = None,
    recipe: Optional[tuple[int, int, int]] = None,
    recipe_a: Optional[tuple[int, int]] = None,
    recipe_b: Optional[tuple[int, int]] = None,
    disable_ue8m0_cast: bool = False
) -> None:
    """
    ACLNN-backed dense low-precision GEMM: D = A @ B^T (+ C).

    Computes the product through aclnnQuantMatmulV5 using MX (ue8m0) scale factors;
    supports recipe [m, n, k] = [1, 1, 32] only, gran_mn != 1 will be rejected by aclnn.
    supports the **fp8xfp8**, **fp4xfp4** and **fp8xfp4** operand dtype combinations
    (**fp4xfp8** is not supported, see the table below).
    Operands are passed in their natural orientation (aclnn honours their real
    strides), while each scale factor follows its operand's physical major. The
    `nt` / `nn` / `tn` / `tt` binding suffix selects the **logical shape** of the two
    operands (see the per-argument shapes below). Requires K to be a multiple of 64.

    | A dtype | A major | B dtype | B major | ok?     | reason                                            |
    |---------|---------|---------|---------|---------|---------------------------------------------------|
    | fp8     | *       | fp8     | *       | **yes** | W8A8: any major, bf16 or fp32 output              |
    | fp4     | *       | fp4     | *       | **yes** | W4A4: any major, bf16 or fp32 output              |
    | fp8     | K       | fp4     | K       | **yes** | A8W4 (NT); bf16/fp16 output only (fp32 rejected)  |
    | fp8     | K       | fp4     | MN      | **no**  | A8W4 ND only support transposeX2 is true          |
    | fp8     | MN      | fp4     | *       | **no**  | Only support transposeX1 is false                 |
    | fp4     | *       | fp8     | *       | **no**  | Invalid x1 or x2 dtype (fp4 must be x2, not x1)    |

    Args:
        a: (A, sfA) tuple.
            * **A**: [M, K] (nt/nn) or [K, M] (tn/tt), fp8_e4m3 or fp4_e2m1x2
            * **sfA**: [ceil(M / gran_m), ceil(K / gran_k)], fp32 or short (packed ue8m0), in any major
            * **NOTE**: when sfA is fp32, only 8-bit exponential is used, 23-bit mantissa must be 0, otherwise the kernel will panic
        b: (B, sfB) tuple.
            * **B**: [N, K] (nt/tt) or [K, N] (tn/nn), fp8_e4m3 or fp4_e2m1x2
            * **sfB**: [ceil(N / gran_n), ceil(K / gran_k)], fp32 or short (packed ue8m0), in any major
            * **NOTE**: when sfB is fp32, only 8-bit exponential is used, 23-bit mantissa must be 0, otherwise the kernel will panic
        d: shape [M, N], bf16 or fp32.
        c: if present, the kernel computes C += A @ B, otherwise computes D = A @ B
        recipe: optional combined quantization granularity (gran_m, gran_n, gran_k);
            mutually exclusive with recipe_a / recipe_b.
        recipe_a: optional per-operand granularity (gran_mn, gran_k) for A.
        recipe_b: optional per-operand granularity (gran_mn, gran_k) for B.
        disable_ue8m0_cast: when true, forbid casting float32 scale factors into ue8m0.
    """
    ...


def aclnn_fp8_fp4_gemm_nn(
    a: tuple[torch.Tensor, torch.Tensor],
    b: tuple[torch.Tensor, torch.Tensor],
    d: torch.Tensor,
    c: Optional[torch.Tensor] = None,
    recipe: Optional[tuple[int, int, int]] = None,
    recipe_a: Optional[tuple[int, int]] = None,
    recipe_b: Optional[tuple[int, int]] = None,
    disable_ue8m0_cast: bool = False
) -> None:
    """
    ACLNN-backed dense low-precision GEMM: D = A @ B^T (+ C).

    Computes the product through aclnnQuantMatmulV5 using MX (ue8m0) scale factors;
    supports recipe [m, n, k] = [1, 1, 32] only, gran_mn != 1 will be rejected by aclnn.
    supports the **fp8xfp8**, **fp4xfp4** and **fp8xfp4** operand dtype combinations
    (**fp4xfp8** is not supported, see the table below).
    Operands are passed in their natural orientation (aclnn honours their real
    strides), while each scale factor follows its operand's physical major. The
    `nt` / `nn` / `tn` / `tt` binding suffix selects the **logical shape** of the two
    operands (see the per-argument shapes below). Requires K to be a multiple of 64.

    | A dtype | A major | B dtype | B major | ok?     | reason                                            |
    |---------|---------|---------|---------|---------|---------------------------------------------------|
    | fp8     | *       | fp8     | *       | **yes** | W8A8: any major, bf16 or fp32 output              |
    | fp4     | *       | fp4     | *       | **yes** | W4A4: any major, bf16 or fp32 output              |
    | fp8     | K       | fp4     | K       | **yes** | A8W4 (NT); bf16/fp16 output only (fp32 rejected)  |
    | fp8     | K       | fp4     | MN      | **no**  | A8W4 ND only support transposeX2 is true          |
    | fp8     | MN      | fp4     | *       | **no**  | Only support transposeX1 is false                 |
    | fp4     | *       | fp8     | *       | **no**  | Invalid x1 or x2 dtype (fp4 must be x2, not x1)    |

    Args:
        a: (A, sfA) tuple.
            * **A**: [M, K] (nt/nn) or [K, M] (tn/tt), fp8_e4m3 or fp4_e2m1x2
            * **sfA**: [ceil(M / gran_m), ceil(K / gran_k)], fp32 or short (packed ue8m0), in any major
            * **NOTE**: when sfA is fp32, only 8-bit exponential is used, 23-bit mantissa must be 0, otherwise the kernel will panic
        b: (B, sfB) tuple.
            * **B**: [N, K] (nt/tt) or [K, N] (tn/nn), fp8_e4m3 or fp4_e2m1x2
            * **sfB**: [ceil(N / gran_n), ceil(K / gran_k)], fp32 or short (packed ue8m0), in any major
            * **NOTE**: when sfB is fp32, only 8-bit exponential is used, 23-bit mantissa must be 0, otherwise the kernel will panic
        d: shape [M, N], bf16 or fp32.
        c: if present, the kernel computes C += A @ B, otherwise computes D = A @ B
        recipe: optional combined quantization granularity (gran_m, gran_n, gran_k);
            mutually exclusive with recipe_a / recipe_b.
        recipe_a: optional per-operand granularity (gran_mn, gran_k) for A.
        recipe_b: optional per-operand granularity (gran_mn, gran_k) for B.
        disable_ue8m0_cast: when true, forbid casting float32 scale factors into ue8m0.
    """
    ...


def aclnn_fp8_fp4_gemm_tn(
    a: tuple[torch.Tensor, torch.Tensor],
    b: tuple[torch.Tensor, torch.Tensor],
    d: torch.Tensor,
    c: Optional[torch.Tensor] = None,
    recipe: Optional[tuple[int, int, int]] = None,
    recipe_a: Optional[tuple[int, int]] = None,
    recipe_b: Optional[tuple[int, int]] = None,
    disable_ue8m0_cast: bool = False
) -> None:
    """
    ACLNN-backed dense low-precision GEMM: D = A @ B^T (+ C).

    Computes the product through aclnnQuantMatmulV5 using MX (ue8m0) scale factors;
    supports recipe [m, n, k] = [1, 1, 32] only, gran_mn != 1 will be rejected by aclnn.
    supports the **fp8xfp8**, **fp4xfp4** and **fp8xfp4** operand dtype combinations
    (**fp4xfp8** is not supported, see the table below).
    Operands are passed in their natural orientation (aclnn honours their real
    strides), while each scale factor follows its operand's physical major. The
    `nt` / `nn` / `tn` / `tt` binding suffix selects the **logical shape** of the two
    operands (see the per-argument shapes below). Requires K to be a multiple of 64.

    | A dtype | A major | B dtype | B major | ok?     | reason                                            |
    |---------|---------|---------|---------|---------|---------------------------------------------------|
    | fp8     | *       | fp8     | *       | **yes** | W8A8: any major, bf16 or fp32 output              |
    | fp4     | *       | fp4     | *       | **yes** | W4A4: any major, bf16 or fp32 output              |
    | fp8     | K       | fp4     | K       | **yes** | A8W4 (NT); bf16/fp16 output only (fp32 rejected)  |
    | fp8     | K       | fp4     | MN      | **no**  | A8W4 ND only support transposeX2 is true          |
    | fp8     | MN      | fp4     | *       | **no**  | Only support transposeX1 is false                 |
    | fp4     | *       | fp8     | *       | **no**  | Invalid x1 or x2 dtype (fp4 must be x2, not x1)    |

    Args:
        a: (A, sfA) tuple.
            * **A**: [M, K] (nt/nn) or [K, M] (tn/tt), fp8_e4m3 or fp4_e2m1x2
            * **sfA**: [ceil(M / gran_m), ceil(K / gran_k)], fp32 or short (packed ue8m0), in any major
            * **NOTE**: when sfA is fp32, only 8-bit exponential is used, 23-bit mantissa must be 0, otherwise the kernel will panic
        b: (B, sfB) tuple.
            * **B**: [N, K] (nt/tt) or [K, N] (tn/nn), fp8_e4m3 or fp4_e2m1x2
            * **sfB**: [ceil(N / gran_n), ceil(K / gran_k)], fp32 or short (packed ue8m0), in any major
            * **NOTE**: when sfB is fp32, only 8-bit exponential is used, 23-bit mantissa must be 0, otherwise the kernel will panic
        d: shape [M, N], bf16 or fp32.
        c: if present, the kernel computes C += A @ B, otherwise computes D = A @ B
        recipe: optional combined quantization granularity (gran_m, gran_n, gran_k);
            mutually exclusive with recipe_a / recipe_b.
        recipe_a: optional per-operand granularity (gran_mn, gran_k) for A.
        recipe_b: optional per-operand granularity (gran_mn, gran_k) for B.
        disable_ue8m0_cast: when true, forbid casting float32 scale factors into ue8m0.
    """
    ...


def aclnn_fp8_fp4_gemm_tt(
    a: tuple[torch.Tensor, torch.Tensor],
    b: tuple[torch.Tensor, torch.Tensor],
    d: torch.Tensor,
    c: Optional[torch.Tensor] = None,
    recipe: Optional[tuple[int, int, int]] = None,
    recipe_a: Optional[tuple[int, int]] = None,
    recipe_b: Optional[tuple[int, int]] = None,
    disable_ue8m0_cast: bool = False
) -> None:
    """
    ACLNN-backed dense low-precision GEMM: D = A @ B^T (+ C).

    Computes the product through aclnnQuantMatmulV5 using MX (ue8m0) scale factors;
    supports recipe [m, n, k] = [1, 1, 32] only, gran_mn != 1 will be rejected by aclnn.
    supports the **fp8xfp8**, **fp4xfp4** and **fp8xfp4** operand dtype combinations
    (**fp4xfp8** is not supported, see the table below).
    Operands are passed in their natural orientation (aclnn honours their real
    strides), while each scale factor follows its operand's physical major. The
    `nt` / `nn` / `tn` / `tt` binding suffix selects the **logical shape** of the two
    operands (see the per-argument shapes below). Requires K to be a multiple of 64.

    | A dtype | A major | B dtype | B major | ok?     | reason                                            |
    |---------|---------|---------|---------|---------|---------------------------------------------------|
    | fp8     | *       | fp8     | *       | **yes** | W8A8: any major, bf16 or fp32 output              |
    | fp4     | *       | fp4     | *       | **yes** | W4A4: any major, bf16 or fp32 output              |
    | fp8     | K       | fp4     | K       | **yes** | A8W4 (NT); bf16/fp16 output only (fp32 rejected)  |
    | fp8     | K       | fp4     | MN      | **no**  | A8W4 ND only support transposeX2 is true          |
    | fp8     | MN      | fp4     | *       | **no**  | Only support transposeX1 is false                 |
    | fp4     | *       | fp8     | *       | **no**  | Invalid x1 or x2 dtype (fp4 must be x2, not x1)    |

    Args:
        a: (A, sfA) tuple.
            * **A**: [M, K] (nt/nn) or [K, M] (tn/tt), fp8_e4m3 or fp4_e2m1x2
            * **sfA**: [ceil(M / gran_m), ceil(K / gran_k)], fp32 or short (packed ue8m0), in any major
            * **NOTE**: when sfA is fp32, only 8-bit exponential is used, 23-bit mantissa must be 0, otherwise the kernel will panic
        b: (B, sfB) tuple.
            * **B**: [N, K] (nt/tt) or [K, N] (tn/nn), fp8_e4m3 or fp4_e2m1x2
            * **sfB**: [ceil(N / gran_n), ceil(K / gran_k)], fp32 or short (packed ue8m0), in any major
            * **NOTE**: when sfB is fp32, only 8-bit exponential is used, 23-bit mantissa must be 0, otherwise the kernel will panic
        d: shape [M, N], bf16 or fp32.
        c: if present, the kernel computes C += A @ B, otherwise computes D = A @ B
        recipe: optional combined quantization granularity (gran_m, gran_n, gran_k);
            mutually exclusive with recipe_a / recipe_b.
        recipe_a: optional per-operand granularity (gran_mn, gran_k) for A.
        recipe_b: optional per-operand granularity (gran_mn, gran_k) for B.
        disable_ue8m0_cast: when true, forbid casting float32 scale factors into ue8m0.
    """
    ...


def aclnn_bf16_gemm_nt(
    a: torch.Tensor,
    b: torch.Tensor,
    d: torch.Tensor,
    c: Optional[torch.Tensor] = None
) -> None:
    """
    ACLNN-backed dense bfloat16 GEMM: D = A @ B^T (+ C).

    Uses aclnnMatmul when `c` is absent, or aclnnInplaceAddmm to compute
    D = C + A @ B^T when `c` is provided. The `nt` / `nn` / `tn` / `tt` binding
    suffix selects the logical shape of the two operands (see the shapes below).

    Args:
        a: [M, K] (nt/nn) or [K, M] (tn/tt), bf16
        b: [N, K] (nt/tt) or [K, N] (tn/nn), bf16
        d: [M, N], bf16.
        c: if present, the kernel computes C += A @ B, otherwise computes D = A @ B
    """
    ...


def aclnn_bf16_gemm_nn(
    a: torch.Tensor,
    b: torch.Tensor,
    d: torch.Tensor,
    c: Optional[torch.Tensor] = None
) -> None:
    """
    ACLNN-backed dense bfloat16 GEMM: D = A @ B^T (+ C).

    Uses aclnnMatmul when `c` is absent, or aclnnInplaceAddmm to compute
    D = C + A @ B^T when `c` is provided. The `nt` / `nn` / `tn` / `tt` binding
    suffix selects the logical shape of the two operands (see the shapes below).

    Args:
        a: [M, K] (nt/nn) or [K, M] (tn/tt), bf16
        b: [N, K] (nt/tt) or [K, N] (tn/nn), bf16
        d: [M, N], bf16.
        c: if present, the kernel computes C += A @ B, otherwise computes D = A @ B
    """
    ...


def aclnn_bf16_gemm_tn(
    a: torch.Tensor,
    b: torch.Tensor,
    d: torch.Tensor,
    c: Optional[torch.Tensor] = None
) -> None:
    """
    ACLNN-backed dense bfloat16 GEMM: D = A @ B^T (+ C).

    Uses aclnnMatmul when `c` is absent, or aclnnInplaceAddmm to compute
    D = C + A @ B^T when `c` is provided. The `nt` / `nn` / `tn` / `tt` binding
    suffix selects the logical shape of the two operands (see the shapes below).

    Args:
        a: [M, K] (nt/nn) or [K, M] (tn/tt), bf16
        b: [N, K] (nt/tt) or [K, N] (tn/nn), bf16
        d: [M, N], bf16.
        c: if present, the kernel computes C += A @ B, otherwise computes D = A @ B
    """
    ...


def aclnn_bf16_gemm_tt(
    a: torch.Tensor,
    b: torch.Tensor,
    d: torch.Tensor,
    c: Optional[torch.Tensor] = None
) -> None:
    """
    ACLNN-backed dense bfloat16 GEMM: D = A @ B^T (+ C).

    Uses aclnnMatmul when `c` is absent, or aclnnInplaceAddmm to compute
    D = C + A @ B^T when `c` is provided. The `nt` / `nn` / `tn` / `tt` binding
    suffix selects the logical shape of the two operands (see the shapes below).

    Args:
        a: [M, K] (nt/nn) or [K, M] (tn/tt), bf16
        b: [N, K] (nt/tt) or [K, N] (tn/nn), bf16
        d: [M, N], bf16.
        c: if present, the kernel computes C += A @ B, otherwise computes D = A @ B
    """
    ...


def fp8_fp4_mqa_logits(
    q: tuple[torch.Tensor, Optional[torch.Tensor]],
    kv: tuple[torch.Tensor, torch.Tensor],
    weights: torch.Tensor,
    cu_seq_len_k_start: torch.Tensor,
    cu_seq_len_k_end: torch.Tensor,
    max_seqlen_k: int
) -> torch.Tensor: ...


def get_paged_mqa_logits_metadata(
    context_lens: torch.Tensor,
    num_heads: int,
    indices: torch.Tensor
) -> torch.Tensor: ...


def fp8_fp4_paged_mqa_logits(
    q: tuple[torch.Tensor, Optional[torch.Tensor]],
    kv_cache: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    block_table: torch.Tensor,
    schedule_meta: torch.Tensor,
    max_context_len: int,
    indices: torch.Tensor,
    use_ascend_kv_layout: bool = False
) -> torch.Tensor: ...


def einsum(
    expr: str,
    a: torch.Tensor,
    b: torch.Tensor,
    d: torch.Tensor,
    c: Optional[torch.Tensor] = None,
    use_cublaslt: bool = False,
    epilogue: Any = None
) -> None:
    """
    Batched bfloat16 einsum over a fixed set of contraction expressions.

    Permutes the operands and dispatches to a batched bf16 GEMM. Dimensions:
    b = batch, h = head, d / r = feature dims.

    Args:
        expr: the einsum contraction string. Supported values (any other raises):
            "bhr,hdr->bhd":  a [b, h, r], b [h, d, r] -> d [b, h, d]
            "bhd,hdr->bhr":  a [b, h, d], b [h, d, r] -> d [b, h, r]
            "bhd,bhr->hdr":  a [b, h, d], b [b, h, r] -> d [h, d, r]
        a: bf16. Shape matches the first term of `expr`.
        b: bf16. Shape matches the second term of `expr`.
        d: output tensor, bf16 or fp32. Shape matches the result term of `expr`.
        c: if present, the kernel computes C += einsum(expr, a, b), otherwise computes D = einsum(expr, a, b)
        use_cublaslt: if true, route to the Ascend einsum kernel; not implemented yet, so it raises.
    """
    ...


def fp8_bmm(
    a: tuple[torch.Tensor, torch.Tensor],
    b: tuple[torch.Tensor, torch.Tensor],
    d: Any,
    c: Optional[torch.Tensor] = None,
    recipe: Optional[tuple[int, int, int]] = (1, 1, 32),
    compiled_dims: str = "nk",
    epilogue: Any = None
) -> None:
    """
    Batched FP8 GEMM: D[i] = A[i] @ B[i]^T (+ C[i]) for each batch i.

    Args:
        a: (A, sfA) tuple.
            * **A**: [B, M, K], fp8_e4m3
            * **sfA**: [B, ceil(M / gran_m), ceil(K / gran_k)], fp32 or short (packed ue8m0), in any major
            * **NOTE**: when sfA is fp32, only 8-bit exponential is used, 23-bit mantissa must be 0
        b: (B, sfB) tuple.
            * **B**: [B, N, K], fp8_e4m3
            * **sfB**: [B, ceil(N / gran_n), ceil(K / gran_k)], fp32 or short (packed ue8m0), in any major
            * **NOTE**: when sfB is fp32, only 8-bit exponential is used, 23-bit mantissa must be 0
        d: shape [B, M, N], bf16 or fp32, or FP8 with FP8Quantization(sfd) / a (D, SFD) pair.
        c: if present, the kernel computes C += A @ B, otherwise computes D = A @ B
        recipe: optional combined quantization granularity (gran_m, gran_n, gran_k).
        compiled_dims: which dimensions (e.g. "mn" or "nk") are baked in as compile-time constants for the kernel.
    """
    ...


def bf16_bmm(
    a: torch.Tensor,
    b: torch.Tensor,
    d: torch.Tensor,
    c: Optional[torch.Tensor] = None,
    compiled_dims: str = "nk",
    epilogue: Any = None
) -> None:
    """
    Batched bfloat16 GEMM: D[i] = A[i] @ B[i]^T (+ C[i]) for each batch i.

    Args:
        a: A, [B, M, K], bf16
        b: B, [B, N, K], bf16
        d: [B, M, N], bf16 or fp32.
        c: if present, the kernel computes C += A @ B, otherwise computes D = A @ B
        compiled_dims: which dimensions (e.g. "mn" or "nk") are baked in as compile-time constants for the kernel.
    """
    ...


def fp8_einsum(
    expr: str,
    a: tuple[torch.Tensor, torch.Tensor],
    b: tuple[torch.Tensor, torch.Tensor],
    d: Any,
    c: Optional[torch.Tensor] = None,
    recipe: tuple[int, int, int] = (1, 1, 32),
    epilogue: Any = None
) -> None:
    """
    FP8 einsum over a fixed set of contraction expressions.

    Permutes the operands (and their scale factors) and dispatches to a batched fp8
    GEMM. Dimensions: b = batch, h = head, d / r = feature dims.

    Args:
        expr: the einsum contraction string. Supported values (any other raises):
            "bhr,hdr->bhd":  a [b, h, r], b [h, d, r] -> d [b, h, d]
            "bhd,hdr->bhr":  a [b, h, d], b [h, d, r] -> d [b, h, r]
            "bhd,bhr->hdr":  a [b, h, d], b [b, h, r] -> d [h, d, r]
        a: (A, sfA) tuple.
            * **A**: fp8_e4m3, shape matches the first term of `expr`.
            * **sfA**: fp32 or short (packed ue8m0), in any major
            * **NOTE**: when sfA is fp32, only 8-bit exponential is used, 23-bit mantissa must be 0, otherwise the kernel will panic
        b: (B, sfB) tuple.
            * **B**: fp8_e4m3, shape matches the second term of `expr`.
            * **sfB**: fp32 or short (packed ue8m0), in any major
            * **NOTE**: when sfB is fp32, only 8-bit exponential is used, 23-bit mantissa must be 0, otherwise the kernel will panic
        d: bf16/fp32, or FP8 with FP8Quantization(sfd); forward also accepts a (D, SFD) pair.
        c: if present, the kernel computes C += einsum(expr, a, b), otherwise computes D = einsum(expr, a, b)
        recipe: combined quantization granularity (gran_m, gran_n, gran_k).
    """
    ...


def fp8_fp4_gemm_nt(
    a: tuple[torch.Tensor, torch.Tensor],
    b: tuple[torch.Tensor, torch.Tensor],
    d: torch.Tensor,
    c: Optional[torch.Tensor] = None,
    recipe: Optional[tuple[int, int, int]] = None,
    recipe_a: Optional[tuple[int, int]] = None,
    recipe_b: Optional[tuple[int, int]] = None,
    compiled_dims: str = "nk",
    disable_ue8m0_cast: bool = False,
    alpha: Optional[float] = None,
    epilogue: Any = None
) -> None:
    """
    Generic dense low-precision GEMM: D = A @ B^T (+ C).

    Supports the **fp8xfp8**, **fp4xfp4** and **fp8xfp4** operand dtype combinations
    (**fp4xfp8** is no longer supported);
    The `nt` / `nn` / `tn` / `tt` binding suffix selects the **logical shape** of the two operands

    Args:
        a: (A, sfA) tuple.
            * **A**: [M, K] (nt/nn) or [K, M] (tn/tt), fp8_e4m3 or fp4_e2m1x2
            * **sfA**: [ceil(M / gran_m), ceil(K / gran_k)], fp32 or short (packed ue8m0), in any major
            * **NOTE**: when sfA is fp32, only 8-bit exponential is used, 23-bit mantissa must be 0, otherwise the kernel
            will panic
        b: (B, sfB) tuple.
            * **B**: [N, K] (nt/tt) or [K, N] (tn/nn), fp8_e4m3 or fp4_e2m1x2
            * **sfB**: [ceil(N / gran_n), ceil(K / gran_k)], fp32 or short (packed ue8m0), in any major
            * **NOTE**: when sfB is fp32, only 8-bit exponential is used, 23-bit mantissa must be 0, otherwise the kernel
            will panic
        d: shape [M, N], bf16 or fp32.
        c: if present, the kernel computes C += A @ B, otherwise computes D = A @ B
        recipe: optional combined quantization granularity (gran_m, gran_n, gran_k);
            mutually exclusive with recipe_a / recipe_b.
        recipe_a: optional per-operand granularity (gran_mn, gran_k) for A.
        recipe_b: optional per-operand granularity (gran_mn, gran_k) for B.
        compiled_dims: which dimensions (e.g. "mn" or "nk") are baked in as compile-time constants for the kernel.
        disable_ue8m0_cast: when true, forbid casting float32 scale factors into ue8m0.
        alpha: optional multiplier applied to A @ B before adding C.
    """
    ...


def fp8_fp4_gemm_nn(
    a: tuple[torch.Tensor, torch.Tensor],
    b: tuple[torch.Tensor, torch.Tensor],
    d: torch.Tensor,
    c: Optional[torch.Tensor] = None,
    recipe: Optional[tuple[int, int, int]] = None,
    recipe_a: Optional[tuple[int, int]] = None,
    recipe_b: Optional[tuple[int, int]] = None,
    compiled_dims: str = "nk",
    disable_ue8m0_cast: bool = False,
    alpha: Optional[float] = None,
    epilogue: Any = None
) -> None:
    """
    Generic dense low-precision GEMM: D = A @ B^T (+ C).

    Supports the **fp8xfp8**, **fp4xfp4** and **fp8xfp4** operand dtype combinations
    (**fp4xfp8** is no longer supported);
    The `nt` / `nn` / `tn` / `tt` binding suffix selects the **logical shape** of the two operands

    Args:
        a: (A, sfA) tuple.
            * **A**: [M, K] (nt/nn) or [K, M] (tn/tt), fp8_e4m3 or fp4_e2m1x2
            * **sfA**: [ceil(M / gran_m), ceil(K / gran_k)], fp32 or short (packed ue8m0), in any major
            * **NOTE**: when sfA is fp32, only 8-bit exponential is used, 23-bit mantissa must be 0, otherwise the kernel
            will panic
        b: (B, sfB) tuple.
            * **B**: [N, K] (nt/tt) or [K, N] (tn/nn), fp8_e4m3 or fp4_e2m1x2
            * **sfB**: [ceil(N / gran_n), ceil(K / gran_k)], fp32 or short (packed ue8m0), in any major
            * **NOTE**: when sfB is fp32, only 8-bit exponential is used, 23-bit mantissa must be 0, otherwise the kernel
            will panic
        d: shape [M, N], bf16 or fp32.
        c: if present, the kernel computes C += A @ B, otherwise computes D = A @ B
        recipe: optional combined quantization granularity (gran_m, gran_n, gran_k);
            mutually exclusive with recipe_a / recipe_b.
        recipe_a: optional per-operand granularity (gran_mn, gran_k) for A.
        recipe_b: optional per-operand granularity (gran_mn, gran_k) for B.
        compiled_dims: which dimensions (e.g. "mn" or "nk") are baked in as compile-time constants for the kernel.
        disable_ue8m0_cast: when true, forbid casting float32 scale factors into ue8m0.
        alpha: optional multiplier applied to A @ B before adding C.
    """
    ...


def fp8_fp4_gemm_tn(
    a: tuple[torch.Tensor, torch.Tensor],
    b: tuple[torch.Tensor, torch.Tensor],
    d: torch.Tensor,
    c: Optional[torch.Tensor] = None,
    recipe: Optional[tuple[int, int, int]] = None,
    recipe_a: Optional[tuple[int, int]] = None,
    recipe_b: Optional[tuple[int, int]] = None,
    compiled_dims: str = "mn",
    disable_ue8m0_cast: bool = False,
    alpha: Optional[float] = None,
    epilogue: Any = None
) -> None:
    """
    Generic dense low-precision GEMM: D = A @ B^T (+ C).

    Supports the **fp8xfp8**, **fp4xfp4** and **fp8xfp4** operand dtype combinations
    (**fp4xfp8** is no longer supported);
    The `nt` / `nn` / `tn` / `tt` binding suffix selects the **logical shape** of the two operands

    Args:
        a: (A, sfA) tuple.
            * **A**: [M, K] (nt/nn) or [K, M] (tn/tt), fp8_e4m3 or fp4_e2m1x2
            * **sfA**: [ceil(M / gran_m), ceil(K / gran_k)], fp32 or short (packed ue8m0), in any major
            * **NOTE**: when sfA is fp32, only 8-bit exponential is used, 23-bit mantissa must be 0, otherwise the kernel
            will panic
        b: (B, sfB) tuple.
            * **B**: [N, K] (nt/tt) or [K, N] (tn/nn), fp8_e4m3 or fp4_e2m1x2
            * **sfB**: [ceil(N / gran_n), ceil(K / gran_k)], fp32 or short (packed ue8m0), in any major
            * **NOTE**: when sfB is fp32, only 8-bit exponential is used, 23-bit mantissa must be 0, otherwise the kernel
            will panic
        d: shape [M, N], bf16 or fp32.
        c: if present, the kernel computes C += A @ B, otherwise computes D = A @ B
        recipe: optional combined quantization granularity (gran_m, gran_n, gran_k);
            mutually exclusive with recipe_a / recipe_b.
        recipe_a: optional per-operand granularity (gran_mn, gran_k) for A.
        recipe_b: optional per-operand granularity (gran_mn, gran_k) for B.
        compiled_dims: which dimensions (e.g. "mn" or "nk") are baked in as compile-time constants for the kernel.
        disable_ue8m0_cast: when true, forbid casting float32 scale factors into ue8m0.
        alpha: optional multiplier applied to A @ B before adding C.
    """
    ...


def fp8_fp4_gemm_tt(
    a: tuple[torch.Tensor, torch.Tensor],
    b: tuple[torch.Tensor, torch.Tensor],
    d: torch.Tensor,
    c: Optional[torch.Tensor] = None,
    recipe: Optional[tuple[int, int, int]] = None,
    recipe_a: Optional[tuple[int, int]] = None,
    recipe_b: Optional[tuple[int, int]] = None,
    compiled_dims: str = "mn",
    disable_ue8m0_cast: bool = False,
    alpha: Optional[float] = None,
    epilogue: Any = None
) -> None:
    """
    Generic dense low-precision GEMM: D = A @ B^T (+ C).

    Supports the **fp8xfp8**, **fp4xfp4** and **fp8xfp4** operand dtype combinations
    (**fp4xfp8** is no longer supported);
    The `nt` / `nn` / `tn` / `tt` binding suffix selects the **logical shape** of the two operands

    Args:
        a: (A, sfA) tuple.
            * **A**: [M, K] (nt/nn) or [K, M] (tn/tt), fp8_e4m3 or fp4_e2m1x2
            * **sfA**: [ceil(M / gran_m), ceil(K / gran_k)], fp32 or short (packed ue8m0), in any major
            * **NOTE**: when sfA is fp32, only 8-bit exponential is used, 23-bit mantissa must be 0, otherwise the kernel
            will panic
        b: (B, sfB) tuple.
            * **B**: [N, K] (nt/tt) or [K, N] (tn/nn), fp8_e4m3 or fp4_e2m1x2
            * **sfB**: [ceil(N / gran_n), ceil(K / gran_k)], fp32 or short (packed ue8m0), in any major
            * **NOTE**: when sfB is fp32, only 8-bit exponential is used, 23-bit mantissa must be 0, otherwise the kernel
            will panic
        d: shape [M, N], bf16 or fp32.
        c: if present, the kernel computes C += A @ B, otherwise computes D = A @ B
        recipe: optional combined quantization granularity (gran_m, gran_n, gran_k);
            mutually exclusive with recipe_a / recipe_b.
        recipe_a: optional per-operand granularity (gran_mn, gran_k) for A.
        recipe_b: optional per-operand granularity (gran_mn, gran_k) for B.
        compiled_dims: which dimensions (e.g. "mn" or "nk") are baked in as compile-time constants for the kernel.
        disable_ue8m0_cast: when true, forbid casting float32 scale factors into ue8m0.
        alpha: optional multiplier applied to A @ B before adding C.
    """
    ...


def m_grouped_fp8_fp4_gemm_nt_contiguous(
    a: tuple[torch.Tensor, torch.Tensor],
    b: tuple[torch.Tensor, torch.Tensor],
    d: torch.Tensor,
    grouped_layout: torch.Tensor,
    recipe: Optional[tuple[int, int, int]] = None,
    recipe_a: Optional[tuple[int, int]] = None,
    recipe_b: Optional[tuple[int, int]] = None,
    compiled_dims: str = "nk",
    disable_ue8m0_cast: bool = False,
    use_psum_layout: bool = True,
    ensure_zero_padding: bool = True,
    expected_m_for_psum_layout: Optional[int] = None,
    epilogue: Any = None
) -> None:
    """
    M-grouped contiguous low-precision GEMM with D = A @ B^T.

    The rows of A are partitioned into G groups stored contiguously, while B is a
    per-group stack of shape [G, N, K]; group boundaries are described by
    `grouped_layout`. Supports **fp8xfp8**, **fp4xfp4** and **fp8xfp4** operand dtype
    combinations (**fp4xfp8** is no longer supported; the concrete kernel is dispatched
    from the operand dtypes).

    Args:
        a: (A, sfA) tuple.
            * **A**: [M, K] (nt/nn), fp8_e4m3 or fp4_e2m1x2
            * **sfA**: [ceil(M / gran_m), ceil(K / gran_k)], fp32 or short (packed ue8m0), in any major
            * **NOTE**: when sfA is fp32, only 8-bit exponential is used, 23-bit mantissa must be 0, otherwise the kernel
            will panic
        b: (B, sfB) tuple.
            * **B**: [G, N, K] (nt) or [G, K, N] (nn), fp8_e4m3 or fp4_e2m1x2
            * **sfB**: [G, ceil(N / gran_n), ceil(K / gran_k)], fp32 or short (packed ue8m0), in any major
            * **NOTE**: when sfB is fp32, only 8-bit exponential is used, 23-bit mantissa must be 0, otherwise the kernel
            will panic
        d: shape [M, N], bf16 or fp32.
        grouped_layout: [G], int32. Per-group row counts describing the contiguous partition of A.
        recipe: optional combined quantization granularity (gran_m, gran_n, gran_k);
            mutually exclusive with recipe_a / recipe_b.
        recipe_a: optional per-operand granularity (gran_mn, gran_k) for A.
        recipe_b: optional per-operand granularity (gran_mn, gran_k) for B.
        compiled_dims: which dimensions are baked in as compile-time constants.
        disable_ue8m0_cast: when true, forbid casting float32 scale factors into ue8m0.
        use_psum_layout: must be true; selects the partial-sum output layout path.
        ensure_zero_padding: when true, the output is zero-padded only if the input
            is zero-padded (the kernel itself also accepts non-zero padding).
        expected_m_for_psum_layout: optional expected per-group M used to size the layout.
    """
    ...


def m_grouped_fp8_fp4_gemm_nn_contiguous(
    a: tuple[torch.Tensor, torch.Tensor],
    b: tuple[torch.Tensor, torch.Tensor],
    d: torch.Tensor,
    grouped_layout: torch.Tensor,
    recipe: Optional[tuple[int, int, int]] = None,
    recipe_a: Optional[tuple[int, int]] = None,
    recipe_b: Optional[tuple[int, int]] = None,
    compiled_dims: str = "nk",
    disable_ue8m0_cast: bool = False,
    use_psum_layout: bool = True,
    ensure_zero_padding: bool = True,
    expected_m_for_psum_layout: Optional[int] = None,
    epilogue: Any = None
) -> None:
    """
    M-grouped contiguous low-precision GEMM with D = A @ B^T.

    The rows of A are partitioned into G groups stored contiguously, while B is a
    per-group stack of shape [G, N, K]; group boundaries are described by
    `grouped_layout`. Supports **fp8xfp8**, **fp4xfp4** and **fp8xfp4** operand dtype
    combinations (**fp4xfp8** is no longer supported; the concrete kernel is dispatched
    from the operand dtypes).

    Args:
        a: (A, sfA) tuple.
            * **A**: [M, K] (nt/nn), fp8_e4m3 or fp4_e2m1x2
            * **sfA**: [ceil(M / gran_m), ceil(K / gran_k)], fp32 or short (packed ue8m0), in any major
            * **NOTE**: when sfA is fp32, only 8-bit exponential is used, 23-bit mantissa must be 0, otherwise the kernel
            will panic
        b: (B, sfB) tuple.
            * **B**: [G, N, K] (nt) or [G, K, N] (nn), fp8_e4m3 or fp4_e2m1x2
            * **sfB**: [G, ceil(N / gran_n), ceil(K / gran_k)], fp32 or short (packed ue8m0), in any major
            * **NOTE**: when sfB is fp32, only 8-bit exponential is used, 23-bit mantissa must be 0, otherwise the kernel
            will panic
        d: shape [M, N], bf16 or fp32.
        grouped_layout: [G], int32. Per-group row counts describing the contiguous partition of A.
        recipe: optional combined quantization granularity (gran_m, gran_n, gran_k);
            mutually exclusive with recipe_a / recipe_b.
        recipe_a: optional per-operand granularity (gran_mn, gran_k) for A.
        recipe_b: optional per-operand granularity (gran_mn, gran_k) for B.
        compiled_dims: which dimensions are baked in as compile-time constants.
        disable_ue8m0_cast: when true, forbid casting float32 scale factors into ue8m0.
        use_psum_layout: must be true; selects the partial-sum output layout path.
        ensure_zero_padding: when true, the output is zero-padded only if the input
            is zero-padded (the kernel itself also accepts non-zero padding).
        expected_m_for_psum_layout: optional expected per-group M used to size the layout.
    """
    ...


def k_grouped_fp8_gemm_tn_contiguous(
    a: tuple[torch.Tensor, torch.Tensor],
    b: tuple[torch.Tensor, torch.Tensor],
    d: torch.Tensor,
    ks_cpu: Optional[list[int]],
    grouped_layout: torch.Tensor,
    c: Optional[torch.Tensor] = None,
    recipe: tuple[int, int, int] = (1, 1, 32),
    compiled_dims: str = "mn",
    use_psum_layout: bool = True,
    epilogue: Any = None
) -> None:
    """
    K-grouped contiguous low-precision GEMM producing D of shape [G, M, N].

    Contraction along K is split into G groups whose sizes are given by `ks_cpu`;
    operands are passed in the `tn` (transposed) layout. Supports **fp8xfp8**, **fp4xfp4**
    and **fp8xfp4** operand dtype combinations (**fp4xfp8** is no longer supported;
    the concrete kernel is dispatched from the operand dtypes).

    Args:
        a: (A, sfA) tuple.
            * **A**: [K, M] (tn), fp8_e4m3 or fp4_e2m1x2
            * **sfA**: [ceil(M / gran_m), ceil(K / gran_k)], fp32 or short (packed ue8m0), in any major
            * **NOTE**: when sfA is fp32, only 8-bit exponential is used, 23-bit mantissa must be 0, otherwise the kernel
            will panic
        b: (B, sfB) tuple.
            * **B**: [K, N] (tn), fp8_e4m3 or fp4_e2m1x2
            * **sfB**: [ceil(N / gran_n), ceil(K / gran_k)], fp32 or short (packed ue8m0), in any major
            * **NOTE**: when sfB is fp32, only 8-bit exponential is used, 23-bit mantissa must be 0, otherwise the kernel
            will panic
        d: shape [G, M, N], bf16 or fp32.
        ks_cpu: optional [G], int (on CPU). Per-group K sizes.
        grouped_layout: [G], int32. Per-group layout describing the contiguous K partition.
        c: if present, the kernel computes C += A @ B, otherwise computes D = A @ B
        recipe: combined quantization granularity (gran_m, gran_n, gran_k).
        compiled_dims: which dimensions are baked in as compile-time constants.
        use_psum_layout: must be true; selects the partial-sum output layout path.
    """
    ...


def k_grouped_fp8_gemm_nt_contiguous(
    a: tuple[torch.Tensor, torch.Tensor],
    b: tuple[torch.Tensor, torch.Tensor],
    d: torch.Tensor,
    ks_cpu: Optional[list[int]],
    grouped_layout: torch.Tensor,
    c: Optional[torch.Tensor] = None,
    recipe: tuple[int, int, int] = (1, 1, 32),
    compiled_dims: str = "mn",
    use_psum_layout: bool = True,
    epilogue: Any = None
) -> None:
    """
    Deprecated K-grouped low-precision GEMM in the `nt` layout.

    Takes A of shape [M, K] and B of shape [N, K], transposes them into [K, M] and
    [K, N] and forwards to `k_grouped_fp8_gemm_tn_contiguous`. Prefer calling the
    `tn` variant directly with already-transposed operands.
    """
    ...


def bf16_gemm_nt(
    a: torch.Tensor,
    b: torch.Tensor,
    d: torch.Tensor,
    c: Optional[torch.Tensor] = None,
    compiled_dims: str = "nk",
    alpha: Optional[float] = None,
    epilogue: Any = None
) -> None:
    """
    Dense bfloat16 GEMM: D = A @ B^T (+ C).

    The `nt` / `nn` / `tn` / `tt` binding suffix selects the logical shape of the two operands.

    Args:
        a: [M, K] (nt/nn) or [K, M] (tn/tt), bf16
        b: [N, K] (nt/tt) or [K, N] (tn/nn), bf16
        d: [M, N], bf16 or fp32.
        c: if present, the kernel computes C += A @ B, otherwise computes D = A @ B
        compiled_dims: which dimensions (e.g. "mn" or "nk") are baked in as compile-time constants for the kernel.
        alpha: optional multiplier applied to A @ B before adding C.
    """
    ...


def bf16_gemm_nn(
    a: torch.Tensor,
    b: torch.Tensor,
    d: torch.Tensor,
    c: Optional[torch.Tensor] = None,
    compiled_dims: str = "nk",
    alpha: Optional[float] = None,
    epilogue: Any = None
) -> None:
    """
    Dense bfloat16 GEMM: D = A @ B^T (+ C).

    The `nt` / `nn` / `tn` / `tt` binding suffix selects the logical shape of the two operands.

    Args:
        a: [M, K] (nt/nn) or [K, M] (tn/tt), bf16
        b: [N, K] (nt/tt) or [K, N] (tn/nn), bf16
        d: [M, N], bf16 or fp32.
        c: if present, the kernel computes C += A @ B, otherwise computes D = A @ B
        compiled_dims: which dimensions (e.g. "mn" or "nk") are baked in as compile-time constants for the kernel.
        alpha: optional multiplier applied to A @ B before adding C.
    """
    ...


def bf16_gemm_tn(
    a: torch.Tensor,
    b: torch.Tensor,
    d: torch.Tensor,
    c: Optional[torch.Tensor] = None,
    compiled_dims: str = "mn",
    alpha: Optional[float] = None,
    epilogue: Any = None
) -> None:
    """
    Dense bfloat16 GEMM: D = A @ B^T (+ C).

    The `nt` / `nn` / `tn` / `tt` binding suffix selects the logical shape of the two operands.

    Args:
        a: [M, K] (nt/nn) or [K, M] (tn/tt), bf16
        b: [N, K] (nt/tt) or [K, N] (tn/nn), bf16
        d: [M, N], bf16 or fp32.
        c: if present, the kernel computes C += A @ B, otherwise computes D = A @ B
        compiled_dims: which dimensions (e.g. "mn" or "nk") are baked in as compile-time constants for the kernel.
        alpha: optional multiplier applied to A @ B before adding C.
    """
    ...


def bf16_gemm_tt(
    a: torch.Tensor,
    b: torch.Tensor,
    d: torch.Tensor,
    c: Optional[torch.Tensor] = None,
    compiled_dims: str = "mn",
    alpha: Optional[float] = None,
    epilogue: Any = None
) -> None:
    """
    Dense bfloat16 GEMM: D = A @ B^T (+ C).

    The `nt` / `nn` / `tn` / `tt` binding suffix selects the logical shape of the two operands.

    Args:
        a: [M, K] (nt/nn) or [K, M] (tn/tt), bf16
        b: [N, K] (nt/tt) or [K, N] (tn/nn), bf16
        d: [M, N], bf16 or fp32.
        c: if present, the kernel computes C += A @ B, otherwise computes D = A @ B
        compiled_dims: which dimensions (e.g. "mn" or "nk") are baked in as compile-time constants for the kernel.
        alpha: optional multiplier applied to A @ B before adding C.
    """
    ...


def m_grouped_bf16_gemm_nt_contiguous(
    a: torch.Tensor,
    b: torch.Tensor,
    d: torch.Tensor,
    grouped_layout: torch.Tensor,
    compiled_dims: str = "nk",
    use_psum_layout: bool = True,
    ensure_zero_padding: bool = True,
    expected_m_for_psum_layout: Optional[int] = None,
    epilogue: Any = None
) -> None:
    """
    M-grouped contiguous bfloat16 GEMM with D = A @ B^T.

    The rows of A are partitioned into G groups stored contiguously, while B is a
    per-group stack of shape [G, N, K]; group boundaries are described by `grouped_layout`.

    Args:
        a: [M, K] (nt/nn), bf16
        b: [G, N, K] (nt) or [G, K, N] (nn), bf16
        d: [M, N], bf16 or fp32.
        grouped_layout: [G], int32. Per-group row counts describing the contiguous partition of A.
        compiled_dims: which dimensions are baked in as compile-time constants.
        use_psum_layout: must be true; selects the partial-sum output layout path.
        ensure_zero_padding: when true, the output is zero-padded only if the input
            is zero-padded (the kernel itself also accepts non-zero padding).
        expected_m_for_psum_layout: optional expected per-group M used to size the layout.
    """
    ...


def m_grouped_bf16_gemm_nn_contiguous(
    a: torch.Tensor,
    b: torch.Tensor,
    d: torch.Tensor,
    grouped_layout: torch.Tensor,
    compiled_dims: str = "nk",
    use_psum_layout: bool = True,
    ensure_zero_padding: bool = True,
    expected_m_for_psum_layout: Optional[int] = None,
    epilogue: Any = None
) -> None:
    """
    M-grouped contiguous bfloat16 GEMM with D = A @ B^T.

    The rows of A are partitioned into G groups stored contiguously, while B is a
    per-group stack of shape [G, N, K]; group boundaries are described by `grouped_layout`.

    Args:
        a: [M, K] (nt/nn), bf16
        b: [G, N, K] (nt) or [G, K, N] (nn), bf16
        d: [M, N], bf16 or fp32.
        grouped_layout: [G], int32. Per-group row counts describing the contiguous partition of A.
        compiled_dims: which dimensions are baked in as compile-time constants.
        use_psum_layout: must be true; selects the partial-sum output layout path.
        ensure_zero_padding: when true, the output is zero-padded only if the input
            is zero-padded (the kernel itself also accepts non-zero padding).
        expected_m_for_psum_layout: optional expected per-group M used to size the layout.
    """
    ...


def k_grouped_bf16_gemm_tn_contiguous(
    a: torch.Tensor,
    b: torch.Tensor,
    d: torch.Tensor,
    ks_cpu: Optional[list[int]],
    grouped_layout: torch.Tensor,
    c: Optional[torch.Tensor] = None,
    compiled_dims: str = "mn",
    use_psum_layout: bool = True,
    epilogue: Any = None
) -> None:
    """
    K-grouped contiguous bfloat16 GEMM producing D of shape [G, M, N].

    Contraction along K is split into G groups whose sizes are given by `ks_cpu` and grouped_layout;
    operands are passed in the `tn` (transposed) layout.

    Args:
        a: [K, M] (tn), bf16
        b: [K, N] (tn), bf16
        d: [G, M, N], bf16 or fp32.
        ks_cpu: optional [G], int (on CPU). Per-group K sizes.
        grouped_layout: [G], int32. Per-group layout describing the contiguous K partition.
        c: if present, the kernel computes C += A @ B, otherwise computes D = A @ B
        compiled_dims: which dimensions are baked in as compile-time constants.
        use_psum_layout: must be true; selects the partial-sum output layout path.
    """
    ...


def transform_sf_into_required_layout(
    sf: torch.Tensor,
    mn: int,
    k: int,
    recipe: tuple[int, int, int],
    num_groups: Optional[int] = None,
    is_sfa: Optional[bool] = None,
    disable_ue8m0_cast: bool = False,
    psum_layout: Optional[torch.Tensor] = None
) -> torch.Tensor: ...


def transform_k_grouped_sf_into_required_layout(
    sf: torch.Tensor,
    mn: int,
    k: int,
    recipe: tuple[int, int, int],
    is_sfa: Optional[bool],
    psum_layout: torch.Tensor
) -> torch.Tensor: ...


def get_symm_buffer_size_for_mega_moe(
    num_ranks: int,
    num_experts: int,
    num_max_tokens_per_rank: int,
    num_topk: int,
    hidden: int,
    intermediate_hidden: int,
    mma_type: str,
    activation: str,
    num_shared_experts: int = 0
) -> Any: ...


def fp8_fp4_mega_moe(
    y: torch.Tensor,
    l1_weights_tuple: tuple[torch.Tensor, torch.Tensor],
    l2_weights_tuple: tuple[torch.Tensor, torch.Tensor],
    shared_l1_weights_tuple_opt: Optional[tuple[torch.Tensor, torch.Tensor]],
    shared_l2_weights_tuple_opt: Optional[tuple[torch.Tensor, torch.Tensor]],
    cumulative_local_expert_recv_stats: Optional[torch.Tensor],
    sym_buffer: torch.Tensor,
    sym_buffer_ptrs: list[int],
    rank_idx: int,
    num_max_tokens_per_rank: int,
    num_experts: int,
    num_topk: int,
    hidden: int,
    intermediate_hidden: int,
    num_shared_experts: int,
    recipe: tuple[int, int, int],
    activation: str,
    activation_clamp_opt: Optional[float],
    fast_math: bool,
    timer: Optional[torch.Tensor] = None
) -> None: ...


def get_num_sms() -> int: ...


def set_num_sms(
    n: int
) -> None: ...


def set_npu_arch(
    arch: int
) -> None: ...


def get_npu_arch() -> int: ...


def set_mk_alignment_for_contiguous_layout(
    value: int
) -> None: ...


def get_mk_alignment_for_contiguous_layout() -> int: ...


def get_theoretical_mk_alignment_for_contiguous_layout() -> int: ...


def init(
    version: str,
    library_root_path: str
) -> None: ...


def set_dry_run(
    value: bool
) -> None: ...


def get_dry_run() -> bool: ...


def use_deterministic_algorithms(
    enabled: bool
) -> None: ...


def get_deterministic_algorithms() -> bool: ...


def npu_sleep(
    sleep_cycles: int,
    *,
    marker: int = 0
) -> None: ...


def cublaslt_gemm_nt(
    a: torch.Tensor,
    b: torch.Tensor,
    d: torch.Tensor,
    c: Optional[torch.Tensor] = None
) -> None:
    """
    ACLNN-backed dense bfloat16 GEMM: D = A @ B^T (+ C).

    Uses aclnnMatmul when `c` is absent, or aclnnInplaceAddmm to compute
    D = C + A @ B^T when `c` is provided. The `nt` / `nn` / `tn` / `tt` binding
    suffix selects the logical shape of the two operands (see the shapes below).

    Args:
        a: [M, K] (nt/nn) or [K, M] (tn/tt), bf16
        b: [N, K] (nt/tt) or [K, N] (tn/nn), bf16
        d: [M, N], bf16.
        c: if present, the kernel computes C += A @ B, otherwise computes D = A @ B
    """
    ...


def cublaslt_gemm_nn(
    a: torch.Tensor,
    b: torch.Tensor,
    d: torch.Tensor,
    c: Optional[torch.Tensor] = None
) -> None:
    """
    ACLNN-backed dense bfloat16 GEMM: D = A @ B^T (+ C).

    Uses aclnnMatmul when `c` is absent, or aclnnInplaceAddmm to compute
    D = C + A @ B^T when `c` is provided. The `nt` / `nn` / `tn` / `tt` binding
    suffix selects the logical shape of the two operands (see the shapes below).

    Args:
        a: [M, K] (nt/nn) or [K, M] (tn/tt), bf16
        b: [N, K] (nt/tt) or [K, N] (tn/nn), bf16
        d: [M, N], bf16.
        c: if present, the kernel computes C += A @ B, otherwise computes D = A @ B
    """
    ...


def cublaslt_gemm_tn(
    a: torch.Tensor,
    b: torch.Tensor,
    d: torch.Tensor,
    c: Optional[torch.Tensor] = None
) -> None:
    """
    ACLNN-backed dense bfloat16 GEMM: D = A @ B^T (+ C).

    Uses aclnnMatmul when `c` is absent, or aclnnInplaceAddmm to compute
    D = C + A @ B^T when `c` is provided. The `nt` / `nn` / `tn` / `tt` binding
    suffix selects the logical shape of the two operands (see the shapes below).

    Args:
        a: [M, K] (nt/nn) or [K, M] (tn/tt), bf16
        b: [N, K] (nt/tt) or [K, N] (tn/nn), bf16
        d: [M, N], bf16.
        c: if present, the kernel computes C += A @ B, otherwise computes D = A @ B
    """
    ...


def cublaslt_gemm_tt(
    a: torch.Tensor,
    b: torch.Tensor,
    d: torch.Tensor,
    c: Optional[torch.Tensor] = None
) -> None:
    """
    ACLNN-backed dense bfloat16 GEMM: D = A @ B^T (+ C).

    Uses aclnnMatmul when `c` is absent, or aclnnInplaceAddmm to compute
    D = C + A @ B^T when `c` is provided. The `nt` / `nn` / `tn` / `tt` binding
    suffix selects the logical shape of the two operands (see the shapes below).

    Args:
        a: [M, K] (nt/nn) or [K, M] (tn/tt), bf16
        b: [N, K] (nt/tt) or [K, N] (tn/nn), bf16
        d: [M, N], bf16.
        c: if present, the kernel computes C += A @ B, otherwise computes D = A @ B
    """
    ...


def fp8_gemm_nt(
    a: tuple[torch.Tensor, torch.Tensor],
    b: tuple[torch.Tensor, torch.Tensor],
    d: torch.Tensor,
    c: Optional[torch.Tensor] = None,
    recipe: Optional[tuple[int, int, int]] = None,
    recipe_a: Optional[tuple[int, int]] = None,
    recipe_b: Optional[tuple[int, int]] = None,
    compiled_dims: str = "nk",
    disable_ue8m0_cast: bool = False,
    alpha: Optional[float] = None,
    epilogue: Any = None
) -> None:
    """
    Generic dense low-precision GEMM: D = A @ B^T (+ C).

    Supports the **fp8xfp8**, **fp4xfp4** and **fp8xfp4** operand dtype combinations
    (**fp4xfp8** is no longer supported);
    The `nt` / `nn` / `tn` / `tt` binding suffix selects the **logical shape** of the two operands

    Args:
        a: (A, sfA) tuple.
            * **A**: [M, K] (nt/nn) or [K, M] (tn/tt), fp8_e4m3 or fp4_e2m1x2
            * **sfA**: [ceil(M / gran_m), ceil(K / gran_k)], fp32 or short (packed ue8m0), in any major
            * **NOTE**: when sfA is fp32, only 8-bit exponential is used, 23-bit mantissa must be 0, otherwise the kernel
            will panic
        b: (B, sfB) tuple.
            * **B**: [N, K] (nt/tt) or [K, N] (tn/nn), fp8_e4m3 or fp4_e2m1x2
            * **sfB**: [ceil(N / gran_n), ceil(K / gran_k)], fp32 or short (packed ue8m0), in any major
            * **NOTE**: when sfB is fp32, only 8-bit exponential is used, 23-bit mantissa must be 0, otherwise the kernel
            will panic
        d: shape [M, N], bf16 or fp32.
        c: if present, the kernel computes C += A @ B, otherwise computes D = A @ B
        recipe: optional combined quantization granularity (gran_m, gran_n, gran_k);
            mutually exclusive with recipe_a / recipe_b.
        recipe_a: optional per-operand granularity (gran_mn, gran_k) for A.
        recipe_b: optional per-operand granularity (gran_mn, gran_k) for B.
        compiled_dims: which dimensions (e.g. "mn" or "nk") are baked in as compile-time constants for the kernel.
        disable_ue8m0_cast: when true, forbid casting float32 scale factors into ue8m0.
        alpha: optional multiplier applied to A @ B before adding C.
    """
    ...


def fp8_gemm_nn(
    a: tuple[torch.Tensor, torch.Tensor],
    b: tuple[torch.Tensor, torch.Tensor],
    d: torch.Tensor,
    c: Optional[torch.Tensor] = None,
    recipe: Optional[tuple[int, int, int]] = None,
    recipe_a: Optional[tuple[int, int]] = None,
    recipe_b: Optional[tuple[int, int]] = None,
    compiled_dims: str = "nk",
    disable_ue8m0_cast: bool = False,
    alpha: Optional[float] = None,
    epilogue: Any = None
) -> None:
    """
    Generic dense low-precision GEMM: D = A @ B^T (+ C).

    Supports the **fp8xfp8**, **fp4xfp4** and **fp8xfp4** operand dtype combinations
    (**fp4xfp8** is no longer supported);
    The `nt` / `nn` / `tn` / `tt` binding suffix selects the **logical shape** of the two operands

    Args:
        a: (A, sfA) tuple.
            * **A**: [M, K] (nt/nn) or [K, M] (tn/tt), fp8_e4m3 or fp4_e2m1x2
            * **sfA**: [ceil(M / gran_m), ceil(K / gran_k)], fp32 or short (packed ue8m0), in any major
            * **NOTE**: when sfA is fp32, only 8-bit exponential is used, 23-bit mantissa must be 0, otherwise the kernel
            will panic
        b: (B, sfB) tuple.
            * **B**: [N, K] (nt/tt) or [K, N] (tn/nn), fp8_e4m3 or fp4_e2m1x2
            * **sfB**: [ceil(N / gran_n), ceil(K / gran_k)], fp32 or short (packed ue8m0), in any major
            * **NOTE**: when sfB is fp32, only 8-bit exponential is used, 23-bit mantissa must be 0, otherwise the kernel
            will panic
        d: shape [M, N], bf16 or fp32.
        c: if present, the kernel computes C += A @ B, otherwise computes D = A @ B
        recipe: optional combined quantization granularity (gran_m, gran_n, gran_k);
            mutually exclusive with recipe_a / recipe_b.
        recipe_a: optional per-operand granularity (gran_mn, gran_k) for A.
        recipe_b: optional per-operand granularity (gran_mn, gran_k) for B.
        compiled_dims: which dimensions (e.g. "mn" or "nk") are baked in as compile-time constants for the kernel.
        disable_ue8m0_cast: when true, forbid casting float32 scale factors into ue8m0.
        alpha: optional multiplier applied to A @ B before adding C.
    """
    ...


def fp8_gemm_tn(
    a: tuple[torch.Tensor, torch.Tensor],
    b: tuple[torch.Tensor, torch.Tensor],
    d: torch.Tensor,
    c: Optional[torch.Tensor] = None,
    recipe: Optional[tuple[int, int, int]] = None,
    recipe_a: Optional[tuple[int, int]] = None,
    recipe_b: Optional[tuple[int, int]] = None,
    compiled_dims: str = "mn",
    disable_ue8m0_cast: bool = False,
    alpha: Optional[float] = None,
    epilogue: Any = None
) -> None:
    """
    Generic dense low-precision GEMM: D = A @ B^T (+ C).

    Supports the **fp8xfp8**, **fp4xfp4** and **fp8xfp4** operand dtype combinations
    (**fp4xfp8** is no longer supported);
    The `nt` / `nn` / `tn` / `tt` binding suffix selects the **logical shape** of the two operands

    Args:
        a: (A, sfA) tuple.
            * **A**: [M, K] (nt/nn) or [K, M] (tn/tt), fp8_e4m3 or fp4_e2m1x2
            * **sfA**: [ceil(M / gran_m), ceil(K / gran_k)], fp32 or short (packed ue8m0), in any major
            * **NOTE**: when sfA is fp32, only 8-bit exponential is used, 23-bit mantissa must be 0, otherwise the kernel
            will panic
        b: (B, sfB) tuple.
            * **B**: [N, K] (nt/tt) or [K, N] (tn/nn), fp8_e4m3 or fp4_e2m1x2
            * **sfB**: [ceil(N / gran_n), ceil(K / gran_k)], fp32 or short (packed ue8m0), in any major
            * **NOTE**: when sfB is fp32, only 8-bit exponential is used, 23-bit mantissa must be 0, otherwise the kernel
            will panic
        d: shape [M, N], bf16 or fp32.
        c: if present, the kernel computes C += A @ B, otherwise computes D = A @ B
        recipe: optional combined quantization granularity (gran_m, gran_n, gran_k);
            mutually exclusive with recipe_a / recipe_b.
        recipe_a: optional per-operand granularity (gran_mn, gran_k) for A.
        recipe_b: optional per-operand granularity (gran_mn, gran_k) for B.
        compiled_dims: which dimensions (e.g. "mn" or "nk") are baked in as compile-time constants for the kernel.
        disable_ue8m0_cast: when true, forbid casting float32 scale factors into ue8m0.
        alpha: optional multiplier applied to A @ B before adding C.
    """
    ...


def fp8_gemm_tt(
    a: tuple[torch.Tensor, torch.Tensor],
    b: tuple[torch.Tensor, torch.Tensor],
    d: torch.Tensor,
    c: Optional[torch.Tensor] = None,
    recipe: Optional[tuple[int, int, int]] = None,
    recipe_a: Optional[tuple[int, int]] = None,
    recipe_b: Optional[tuple[int, int]] = None,
    compiled_dims: str = "mn",
    disable_ue8m0_cast: bool = False,
    alpha: Optional[float] = None,
    epilogue: Any = None
) -> None:
    """
    Generic dense low-precision GEMM: D = A @ B^T (+ C).

    Supports the **fp8xfp8**, **fp4xfp4** and **fp8xfp4** operand dtype combinations
    (**fp4xfp8** is no longer supported);
    The `nt` / `nn` / `tn` / `tt` binding suffix selects the **logical shape** of the two operands

    Args:
        a: (A, sfA) tuple.
            * **A**: [M, K] (nt/nn) or [K, M] (tn/tt), fp8_e4m3 or fp4_e2m1x2
            * **sfA**: [ceil(M / gran_m), ceil(K / gran_k)], fp32 or short (packed ue8m0), in any major
            * **NOTE**: when sfA is fp32, only 8-bit exponential is used, 23-bit mantissa must be 0, otherwise the kernel
            will panic
        b: (B, sfB) tuple.
            * **B**: [N, K] (nt/tt) or [K, N] (tn/nn), fp8_e4m3 or fp4_e2m1x2
            * **sfB**: [ceil(N / gran_n), ceil(K / gran_k)], fp32 or short (packed ue8m0), in any major
            * **NOTE**: when sfB is fp32, only 8-bit exponential is used, 23-bit mantissa must be 0, otherwise the kernel
            will panic
        d: shape [M, N], bf16 or fp32.
        c: if present, the kernel computes C += A @ B, otherwise computes D = A @ B
        recipe: optional combined quantization granularity (gran_m, gran_n, gran_k);
            mutually exclusive with recipe_a / recipe_b.
        recipe_a: optional per-operand granularity (gran_mn, gran_k) for A.
        recipe_b: optional per-operand granularity (gran_mn, gran_k) for B.
        compiled_dims: which dimensions (e.g. "mn" or "nk") are baked in as compile-time constants for the kernel.
        disable_ue8m0_cast: when true, forbid casting float32 scale factors into ue8m0.
        alpha: optional multiplier applied to A @ B before adding C.
    """
    ...


def m_grouped_fp8_gemm_nt_contiguous(
    a: tuple[torch.Tensor, torch.Tensor],
    b: tuple[torch.Tensor, torch.Tensor],
    d: torch.Tensor,
    grouped_layout: torch.Tensor,
    recipe: Optional[tuple[int, int, int]] = None,
    recipe_a: Optional[tuple[int, int]] = None,
    recipe_b: Optional[tuple[int, int]] = None,
    compiled_dims: str = "nk",
    disable_ue8m0_cast: bool = False,
    use_psum_layout: bool = True,
    ensure_zero_padding: bool = True,
    expected_m_for_psum_layout: Optional[int] = None,
    epilogue: Any = None
) -> None:
    """
    M-grouped contiguous low-precision GEMM with D = A @ B^T.

    The rows of A are partitioned into G groups stored contiguously, while B is a
    per-group stack of shape [G, N, K]; group boundaries are described by
    `grouped_layout`. Supports **fp8xfp8**, **fp4xfp4** and **fp8xfp4** operand dtype
    combinations (**fp4xfp8** is no longer supported; the concrete kernel is dispatched
    from the operand dtypes).

    Args:
        a: (A, sfA) tuple.
            * **A**: [M, K] (nt/nn), fp8_e4m3 or fp4_e2m1x2
            * **sfA**: [ceil(M / gran_m), ceil(K / gran_k)], fp32 or short (packed ue8m0), in any major
            * **NOTE**: when sfA is fp32, only 8-bit exponential is used, 23-bit mantissa must be 0, otherwise the kernel
            will panic
        b: (B, sfB) tuple.
            * **B**: [G, N, K] (nt) or [G, K, N] (nn), fp8_e4m3 or fp4_e2m1x2
            * **sfB**: [G, ceil(N / gran_n), ceil(K / gran_k)], fp32 or short (packed ue8m0), in any major
            * **NOTE**: when sfB is fp32, only 8-bit exponential is used, 23-bit mantissa must be 0, otherwise the kernel
            will panic
        d: shape [M, N], bf16 or fp32.
        grouped_layout: [G], int32. Per-group row counts describing the contiguous partition of A.
        recipe: optional combined quantization granularity (gran_m, gran_n, gran_k);
            mutually exclusive with recipe_a / recipe_b.
        recipe_a: optional per-operand granularity (gran_mn, gran_k) for A.
        recipe_b: optional per-operand granularity (gran_mn, gran_k) for B.
        compiled_dims: which dimensions are baked in as compile-time constants.
        disable_ue8m0_cast: when true, forbid casting float32 scale factors into ue8m0.
        use_psum_layout: must be true; selects the partial-sum output layout path.
        ensure_zero_padding: when true, the output is zero-padded only if the input
            is zero-padded (the kernel itself also accepts non-zero padding).
        expected_m_for_psum_layout: optional expected per-group M used to size the layout.
    """
    ...


def m_grouped_fp8_gemm_nn_contiguous(
    a: tuple[torch.Tensor, torch.Tensor],
    b: tuple[torch.Tensor, torch.Tensor],
    d: torch.Tensor,
    grouped_layout: torch.Tensor,
    recipe: Optional[tuple[int, int, int]] = None,
    recipe_a: Optional[tuple[int, int]] = None,
    recipe_b: Optional[tuple[int, int]] = None,
    compiled_dims: str = "nk",
    disable_ue8m0_cast: bool = False,
    use_psum_layout: bool = True,
    ensure_zero_padding: bool = True,
    expected_m_for_psum_layout: Optional[int] = None,
    epilogue: Any = None
) -> None:
    """
    M-grouped contiguous low-precision GEMM with D = A @ B^T.

    The rows of A are partitioned into G groups stored contiguously, while B is a
    per-group stack of shape [G, N, K]; group boundaries are described by
    `grouped_layout`. Supports **fp8xfp8**, **fp4xfp4** and **fp8xfp4** operand dtype
    combinations (**fp4xfp8** is no longer supported; the concrete kernel is dispatched
    from the operand dtypes).

    Args:
        a: (A, sfA) tuple.
            * **A**: [M, K] (nt/nn), fp8_e4m3 or fp4_e2m1x2
            * **sfA**: [ceil(M / gran_m), ceil(K / gran_k)], fp32 or short (packed ue8m0), in any major
            * **NOTE**: when sfA is fp32, only 8-bit exponential is used, 23-bit mantissa must be 0, otherwise the kernel
            will panic
        b: (B, sfB) tuple.
            * **B**: [G, N, K] (nt) or [G, K, N] (nn), fp8_e4m3 or fp4_e2m1x2
            * **sfB**: [G, ceil(N / gran_n), ceil(K / gran_k)], fp32 or short (packed ue8m0), in any major
            * **NOTE**: when sfB is fp32, only 8-bit exponential is used, 23-bit mantissa must be 0, otherwise the kernel
            will panic
        d: shape [M, N], bf16 or fp32.
        grouped_layout: [G], int32. Per-group row counts describing the contiguous partition of A.
        recipe: optional combined quantization granularity (gran_m, gran_n, gran_k);
            mutually exclusive with recipe_a / recipe_b.
        recipe_a: optional per-operand granularity (gran_mn, gran_k) for A.
        recipe_b: optional per-operand granularity (gran_mn, gran_k) for B.
        compiled_dims: which dimensions are baked in as compile-time constants.
        disable_ue8m0_cast: when true, forbid casting float32 scale factors into ue8m0.
        use_psum_layout: must be true; selects the partial-sum output layout path.
        ensure_zero_padding: when true, the output is zero-padded only if the input
            is zero-padded (the kernel itself also accepts non-zero padding).
        expected_m_for_psum_layout: optional expected per-group M used to size the layout.
    """
    ...
