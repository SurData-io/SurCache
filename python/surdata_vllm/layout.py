"""KV tensor shapes SurData holds. Geometry comes from the registered tensor."""


class LayoutError(RuntimeError):
    pass


def classify_kv_shape(shape) -> str:
    """MLA is rank 3, or rank 4 with a single latent axis.

    v0.33 DeepSeek stores ``(num_blocks, 1, block_size, latent)``. The second
    axis is that latent, not an MHA K/V axis. Rank 5 with axis 1 equal to 2
    is MHA/GQA. Anything else is refused.
    """
    dims = tuple(int(dim) for dim in shape)
    if len(dims) == 4 and dims[1] == 1 and all(dim > 0 for dim in dims):
        return "mla"
    if len(dims) == 3 and all(dim > 0 for dim in dims):
        return "mla"
    if len(dims) == 5 and dims[1] == 2 and all(dim > 0 for dim in dims):
        return "mha"
    raise LayoutError("unsupported kv shape %s" % (dims,))


def require_deepseek_mla(shape) -> str:
    """DeepSeek stores one MLA latent per block. MHA ranks are refused."""
    kind = classify_kv_shape(shape)
    if kind != "mla":
        raise LayoutError(
            "DeepSeek KV is an MLA block, got %s" % (tuple(shape),)
        )
    return kind


def as_mha_axes(shape):
    """Return an MHA shape. An MLA tensor is not reinterpreted as K/V."""
    dims = tuple(int(dim) for dim in shape)
    kind = classify_kv_shape(dims)
    if kind == "mla":
        raise LayoutError("refusing to treat MLA shape %s as MHA" % (dims,))
    if kind != "mha":
        raise LayoutError("unsupported kv shape %s" % (dims,))
    return dims
