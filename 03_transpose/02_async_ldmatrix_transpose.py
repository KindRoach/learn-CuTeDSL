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
    copy_elems: int,
):
    rows, cols = tile_shape
    tile_elems = rows * cols
    assert tile_elems % threads_per_cta == 0

    values_per_thread = tile_elems // threads_per_cta
    assert values_per_thread % copy_elems == 0
    assert cols % values_per_thread == 0
    chunks_per_row = cols // values_per_thread
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
def async_ldmatrix_transpose_staged_kernel(
    mA: cute.Tensor,
    mB: cute.Tensor,
    cta_tiler_a: cute.Shape,
    cta_tiler_b: cute.Shape,
    stage_tiler_a: cute.Shape,
    smem_layout: Union[cute.Layout, cute.ComposedLayout],
    tiled_copy_g2s: cute.TiledCopy,
    tiled_ldmatrix: cute.TiledCopy,
    tiled_copy_r2g: cute.TiledCopy,
):
    tid, _, _ = cute.arch.thread_idx()
    block_m, block_n, _ = cute.arch.block_idx()
    lane_id = tid % 32
    warp_id = tid // 32

    cta_tile_a = cute.local_tile(mA, cta_tiler_a, (block_m, block_n))
    cta_tile_b = cute.local_tile(mB, cta_tiler_b, (block_n, block_m))
    cta_tile_a = cute.make_tensor(cta_tile_a.iterator.align(16), cta_tile_a.layout)

    smem = SmemAllocator()
    smem_tile = smem.allocate_tensor(mA.element_type, smem_layout, byte_alignment=16)
    smem_ldmatrix_tile = cute.make_tensor(
        smem_tile.iterator.align(16),
        smem_tile.layout,
    )

    thr_copy_g2s = tiled_copy_g2s.get_slice(tid)
    thr_ldmatrix = tiled_ldmatrix.get_slice(lane_id)
    thr_copy_r2g = tiled_copy_r2g.get_slice(lane_id)

    # Prime stage 0. Each stage is one 32x64 slab, so every CTA thread issues
    # one 16B cp.async per stage.
    gmem_stage_tile = cute.local_tile(cta_tile_a, stage_tiler_a, (0, 0))
    smem_stage_tile = cute.local_tile(smem_tile, stage_tiler_a, (0, 0))
    cute.copy(
        tiled_copy_g2s,
        thr_copy_g2s.partition_S(gmem_stage_tile),
        thr_copy_g2s.partition_D(smem_stage_tile),
    )
    cute.arch.cp_async_commit_group()

    for warp_m in cutlass.range(4, unroll_full=True):
        cute.arch.cp_async_wait_group(0)
        cute.arch.sync_threads()

        if warp_m < 3:
            next_warp_m = warp_m + 1
            gmem_next_stage_tile = cute.local_tile(
                cta_tile_a,
                stage_tiler_a,
                (next_warp_m, 0),
            )
            smem_next_stage_tile = cute.local_tile(
                smem_tile,
                stage_tiler_a,
                (next_warp_m, 0),
            )
            cute.copy(
                tiled_copy_g2s,
                thr_copy_g2s.partition_S(gmem_next_stage_tile),
                thr_copy_g2s.partition_D(smem_next_stage_tile),
            )
            cute.arch.cp_async_commit_group()

        smem_warp_tile = cute.local_tile(
            smem_ldmatrix_tile,
            (32, 8),
            (warp_m, warp_id),
        )
        smem_warp_tile = cute.make_tensor(
            smem_warp_tile.iterator.align(16),
            smem_warp_tile.layout,
        )
        gmem_warp_tile_b = cute.local_tile(
            cta_tile_b,
            (8, 32),
            (warp_id, warp_m),
        )

        reg_tile = cute.make_rmem_tensor((32, 8), mA.element_type)
        cute.copy(
            tiled_ldmatrix,
            thr_ldmatrix.partition_S(smem_warp_tile),
            thr_ldmatrix.partition_D(reg_tile),
        )

        reg_tile_b = cute.make_tensor(
            reg_tile.iterator,
            cute.make_layout((8, 32), stride=(32, 1)),
        )
        cute.copy(
            tiled_copy_r2g,
            thr_copy_r2g.partition_S(reg_tile_b),
            thr_copy_r2g.partition_D(gmem_warp_tile_b),
        )


@cute.jit
def async_ldmatrix_transpose_staged(mA: cute.Tensor, mB: cute.Tensor):
    assert mA.element_type is cutlass.Float16
    assert mB.element_type is cutlass.Float16
    assert mA.shape[0] == mB.shape[1]
    assert mA.shape[1] == mB.shape[0]

    tile_m = 128
    tile_n = 64
    threads_per_cta = 256
    cta_tiler_a = (tile_m, tile_n)
    cta_tiler_b = (tile_n, tile_m)
    stage_tiler_a = (32, tile_n)

    assert mA.shape[0] % tile_m == 0
    assert mA.shape[1] % tile_n == 0

    copy_bits = 128
    copy_elems = copy_bits // cutlass.Float16.width
    atom_async_copy = cute.make_copy_atom(
        cute.nvgpu.cpasync.CopyG2SOp(
            cache_mode=cute.nvgpu.cpasync.LoadCacheMode.GLOBAL
        ),
        cutlass.Float16,
        num_bits_per_copy=copy_bits,
    )
    tiled_copy_g2s = make_tiled_copy_for_shape(
        atom_async_copy,
        stage_tiler_a,
        threads_per_cta,
        copy_elems,
    )

    atom_ldmatrix = cute.make_copy_atom(
        cute.nvgpu.warp.LdMatrix8x8x16bOp(
            transpose=True,
            num_matrices=4,
        ),
        cutlass.Float16,
    )
    tiled_ldmatrix = cute.make_tiled_copy(
        atom_ldmatrix,
        atom_ldmatrix.layout_src_tv,
        (32, 8),
    )

    atom_store = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), cutlass.Float16)
    tiled_copy_r2g = make_tiled_copy_for_shape(
        atom_store,
        (8, 32),
        32,
        copy_elems,
    )

    smem_layout = cute.make_layout(cta_tiler_a, stride=(tile_n, 1))

    grid_m = cute.ceil_div(mA.shape[0], tile_m)
    grid_n = cute.ceil_div(mA.shape[1], tile_n)
    async_ldmatrix_transpose_staged_kernel(
        mA,
        mB,
        cta_tiler_a,
        cta_tiler_b,
        stage_tiler_a,
        smem_layout,
        tiled_copy_g2s,
        tiled_ldmatrix,
        tiled_copy_r2g,
    ).launch(
        grid=[grid_m, grid_n, 1],
        block=[threads_per_cta, 1, 1],
    )


def run_async_ldmatrix_transpose_staged_example():
    print("run_async_ldmatrix_transpose_staged_example()")
    shape = (4096, 4096)

    a = torch.randn(shape, device="cuda", dtype=torch.float16)
    expected = a.T.contiguous()

    b_staged = torch.empty_like(expected)
    async_ldmatrix_transpose_staged(
        from_dlpack(a, assumed_align=16),
        from_dlpack(b_staged, assumed_align=16),
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(b_staged, expected)
    print("passed: staged async cp + ldmatrix B == A.T")


if __name__ == "__main__":
    run_async_ldmatrix_transpose_staged_example()
