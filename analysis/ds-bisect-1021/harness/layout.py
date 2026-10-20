"""Diagnostic packed-NZ contract; independent of the torch/NPU runtime."""
SHAPES = (32, 96, 256, 512, 1024, 2048, 4096)


def tile_shape(bs, ep=8, experts=384, topk=6):
    if bs not in SHAPES or (ep, experts, topk) != (8, 384, 6):
        raise ValueError(f'Unsupported layout case EP={ep} BS={bs} E={experts} topk={topk}')
    routes = ep * bs * topk
    if routes % experts:
        raise ValueError('Diagnostic layout requires exact balanced expert counts')
    m = routes // experts
    if m <= 128:
        return 128, 512
    if m % 256:
        raise ValueError('Mixed MAD shapes within one expert need a different weight ABI')
    return 256, 256


def pack_numpy(raw, rows, cols):
    # Physical source [E,N256,256,K/2], destination [E,N256,Ktile,Nsub,K32,N,16B].
    e, nb, n, half_k = raw.shape
    if n != 256 or half_k % (cols // 2) or rows * cols != 65536:
        raise ValueError('Invalid weight shape for diagnostic tile')
    return (raw.reshape(e, nb, 256 // rows, rows, half_k // (cols // 2), cols // 32, 16)
            .transpose(0, 1, 4, 2, 5, 3, 6).copy().reshape(raw.shape))
