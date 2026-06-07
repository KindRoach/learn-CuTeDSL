import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

import torch


def customized_layout():
    print("customized_layout()")

    shape = (8, 4)
    offset = (1, 0)

    @cute.jit
    def cute_jit():
        def swap_xy(c):
            x, y = c
            return y, x

        outer = cute.make_identity_layout(shape)
        composed_layout = cute.make_composed_layout(swap_xy, offset, outer)

        cute.printf("outer: {}", outer)
        cute.printf("offset: {}", offset)

        x, y = 2, 3
        cute.printf("outer({}, {}) = {}", x, y, outer((x, y)))
        cute.printf("composed_layout({}, {}) = {}", x, y, composed_layout((x, y)))

    cute_jit()
    print()


def gather_layout():
    print("gather_layout()")

    indices = torch.tensor([4, 1, 7, 3, 0, 2, 5, 6], dtype=torch.int32)
    data = torch.arange(100, 108, dtype=torch.int32)

    @cute.jit
    def cute_jit(indices: cute.Tensor, data: cute.Tensor):
        def lookup_index(c):
            return indices[c]

        shape = indices.shape
        outer = cute.make_identity_layout(shape)
        gather_layout = cute.make_composed_layout(lookup_index, 0, outer)

        cute.printf("outer: {}", outer)

        for i in cutlass.range_constexpr(cute.size(shape)):
            src_index = gather_layout(i)
            src_value = data[src_index]
            cute.printf("{} -> {} -> {}", i, src_index, src_value)

    cute_jit(from_dlpack(indices), from_dlpack(data))


if __name__ == "__main__":
    customized_layout()
    gather_layout()
