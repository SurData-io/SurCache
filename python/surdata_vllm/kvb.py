"""One vLLM block across every layer.

A KVB is the set of per-layer payloads. It is not one contiguous tensor
spanning layers. A short tail (``valid_tokens < block_size``) is still a
KVB on disk. Its prefix hash covers the tokens in that tail, so a later
lookup can hit through it. The hit length is the stored logical length,
not rounded up to a full block.
"""

from surdata_vllm.layout import LayoutError, classify_kv_shape


class KvGeometry:
    def __init__(self, num_blocks, block_size, latent, itemsize, num_layers):
        self.num_blocks = int(num_blocks)
        self.block_size = int(block_size)
        self.latent = int(latent)
        self.itemsize = int(itemsize)
        self.num_layers = int(num_layers)
        if min(self.num_blocks, self.block_size, self.latent, self.itemsize, self.num_layers) <= 0:
            raise LayoutError("kv geometry must be positive")

    @staticmethod
    def from_shape(shape, itemsize, num_layers):
        dims = tuple(int(dim) for dim in shape)
        kind = classify_kv_shape(dims)
        if kind != "mla":
            raise LayoutError("KVB geometry is MLA, got %s" % (dims,))
        if len(dims) == 4:
            num_blocks, latent_axis, block_size, latent = dims
            if latent_axis != 1:
                raise LayoutError("MLA latent axis must be 1, got %s" % (dims,))
        else:
            num_blocks, block_size, latent = dims
        return KvGeometry(num_blocks, block_size, latent, itemsize, num_layers)

    @staticmethod
    def from_layers(shapes, itemsize):
        shapes = [tuple(int(dim) for dim in shape) for shape in shapes]
        if not shapes:
            raise LayoutError("no kv tensors")
        first = shapes[0]
        for shape in shapes[1:]:
            if shape != first:
                raise LayoutError("kv layer shapes differ: %s vs %s" % (first, shape))
        return KvGeometry.from_shape(first, itemsize, len(shapes))

    def layer_stride(self) -> int:
        """Bytes from one block to the next inside a single layer tensor."""
        return self.block_size * self.latent * self.itemsize

    def layer_payload(self, valid_tokens) -> int:
        tokens = int(valid_tokens)
        if tokens < 0 or tokens > self.block_size:
            raise LayoutError("valid_tokens %s outside 0..%s" % (tokens, self.block_size))
        return tokens * self.latent * self.itemsize

    def dram_layer_offset(self, layer: int, valid_tokens: int) -> int:
        """Byte offset of one layer inside a single-block KVB.

        A short tail uses ``layer_payload(valid_tokens)``, not the full block.
        """
        return int(layer) * self.layer_payload(valid_tokens)

    def dram_prefix_offset(self, layer: int, block_index: int, block_count: int) -> int:
        """Byte offset of one full block inside a layer-major prefix.

        Layer ``L`` of block ``b`` sits at ``(L * block_count + b) * payload``.
        One block matches ``dram_layer_offset``.
        """
        payload = self.layer_payload(self.block_size)
        return (int(layer) * int(block_count) + int(block_index)) * payload

    def kvb_bytes(self, valid_tokens) -> int:
        return self.num_layers * self.layer_payload(valid_tokens)

    def full_prefix_tokens(self, token_count) -> int:
        count = max(0, int(token_count))
        return (count // self.block_size) * self.block_size
