from typing import Union

import cutlass
import cutlass.cute as cute
import torch

from cutlass.cute.runtime import from_dlpack
from cutlass.utils import SmemAllocator


def make_tiled_copy_for_shape(
    copy_atom: cute.CopyAtom,
    tile_shape: tuple[int, int],
    threads_per_cta: int,
):
    rows, cols = tile_shape
    tile_elems = rows * cols
    assert tile_elems % threads_per_cta == 0

    # This TV layout assumes each thread owns one contiguous column chunk within
    # a single row. Therefore every row needs at least one thread chunk.
    assert threads_per_cta >= rows

    values_per_thread = tile_elems // threads_per_cta
    chunks_per_row = cols // values_per_thread
    assert cols % values_per_thread == 0
    assert rows * chunks_per_row == threads_per_cta

    thread_layout = cute.make_layout(
        (rows, chunks_per_row),
        stride=(chunks_per_row, 1),
    )

    value_layout = cute.make_layout(
        (1, values_per_thread),
        stride=(values_per_thread, 1),
    )

    return cute.make_tiled_copy_tv(copy_atom, thread_layout, value_layout)


@cute.kernel
def transpose_kernel(
    mA: cute.Tensor,
    mB: cute.Tensor,
    cta_tiler_a: cute.Shape,
    cta_tiler_b: cute.Shape,
    smem_layout_a: Union[cute.Layout, cute.ComposedLayout],
    smem_layout_b: Union[cute.Layout, cute.ComposedLayout],
    tiled_copy_a: cute.TiledCopy,
    tiled_copy_b: cute.TiledCopy,
):
    tid, _, _ = cute.arch.thread_idx()
    block_m, block_n, _ = cute.arch.block_idx()

    # Swap block id for transposed tile.
    cta_tile_a = cute.local_tile(mA, cta_tiler_a, (block_m, block_n))
    cta_tile_b = cute.local_tile(mB, cta_tiler_b, (block_n, block_m))

    # Use only single smem tile for both source and transposed tiles.
    # Transposed tile is just a different logical view of the same physical storage.
    smem = SmemAllocator()
    smem_tile_a = smem.allocate_tensor(mA.element_type, smem_layout_a, byte_alignment=16)
    smem_tile_b = cute.make_tensor(smem_tile_a.iterator, smem_layout_b)

    # Get current thread's slice of the tiled copy.
    thr_copy_a = tiled_copy_a.get_slice(tid)
    thr_copy_b = tiled_copy_b.get_slice(tid)

    # GMEM -> SMEM.
    cute.copy(
        tiled_copy_a,
        thr_copy_a.partition_S(cta_tile_a),
        thr_copy_a.partition_D(smem_tile_a),
    )
    cute.arch.sync_threads()

    # SMEM -> GMEM through a transposed logical view:
    # B(n, m) reads the same storage location as A(m, n).
    cute.copy(
        tiled_copy_b,
        thr_copy_b.partition_S(smem_tile_b),
        thr_copy_b.partition_D(cta_tile_b),
    )


def _transpose_impl(mA: cute.Tensor, mB: cute.Tensor, swizzle=None):
    tile_m = 128
    tile_n = 64
    threads_per_cta = 256

    assert mA.element_type is cutlass.Float16
    assert mB.element_type is cutlass.Float16
    assert mA.shape[0] == mB.shape[1]
    assert mA.shape[1] == mB.shape[0]
    assert mA.shape[0] % tile_m == 0
    assert mA.shape[1] % tile_n == 0

    # Source is row-major; the transposed logical view is column-major.
    cta_tiler_a = (tile_m, tile_n)
    cta_tiler_b = (tile_n, tile_m)
    source_layout = cute.make_layout(cta_tiler_a, stride=(tile_n, 1))
    transposed_layout = cute.make_layout(cta_tiler_b, stride=(1, tile_n))

    if swizzle is None:
        smem_source_layout = source_layout
        smem_transposed_layout = transposed_layout
    else:
        # Apply the selected swizzle to both source and transposed layouts.
        smem_source_layout = cute.make_composed_layout(swizzle, 0, source_layout)
        smem_transposed_layout = cute.make_composed_layout(swizzle, 0, transposed_layout)

    # Create TV layouts and tiled copies from tile shape and thread count.
    copy_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), cutlass.Float16)
    tiled_copy_a = make_tiled_copy_for_shape(
        copy_atom,
        cta_tiler_a,
        threads_per_cta,
    )
    tiled_copy_b = make_tiled_copy_for_shape(
        copy_atom,
        cta_tiler_b,
        threads_per_cta,
    )

    # Launch kernel.
    grid_m = cute.ceil_div(mA.shape[0], tile_m)
    grid_n = cute.ceil_div(mA.shape[1], tile_n)
    transpose_kernel(
        mA,
        mB,
        cta_tiler_a,
        cta_tiler_b,
        smem_source_layout,
        smem_transposed_layout,
        tiled_copy_a,
        tiled_copy_b,
    ).launch(
        grid=[grid_m, grid_n, 1],
        block=[threads_per_cta, 1, 1],
    )


@cute.jit
def transpose_no_swizzle(mA: cute.Tensor, mB: cute.Tensor):
    _transpose_impl(mA, mB)


@cute.jit
def transpose_swizzle(mA: cute.Tensor, mB: cute.Tensor):
    # For fp16:
    # 1 bit  [0..1) fixed for 32bit bank.
    # 5 bits [1..6) used for swizzle unit.
    # 1 bit  [6..7) used to banlance store and load conflict.
    _transpose_impl(mA, mB, cute.make_swizzle(5, 1, 6))


def run_transpose_example():
    print("run_transpose_example()")
    shape = (4096, 4096)

    a = torch.randn(shape, device="cuda", dtype=torch.float16)
    expected = a.T.contiguous()

    b_no_swizzle = torch.empty_like(expected)
    transpose_no_swizzle(from_dlpack(a), from_dlpack(b_no_swizzle))
    torch.testing.assert_close(b_no_swizzle, expected)
    print("passed: no-swizzle B == A.T")

    b_swizzle = torch.empty_like(expected)
    transpose_swizzle(from_dlpack(a), from_dlpack(b_swizzle))
    torch.testing.assert_close(b_swizzle, expected)
    print("passed: swizzle B == A.T")


if __name__ == "__main__":
    run_transpose_example()
