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
    assert threads_per_cta >= rows
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
def async_ldmatrix_transpose_kernel(
    mA: cute.Tensor,
    mB: cute.Tensor,
    cta_tiler_a: cute.Shape,
    cta_tiler_b: cute.Shape,
    stage_tiler_a: cute.Shape,
    warp_tiler_a: cute.Shape,
    warp_tiler_b: cute.Shape,
    smem_layout: Union[cute.Layout, cute.ComposedLayout],
    tiled_copy_g2s: cute.TiledCopy,
    tiled_ldmatrix: cute.TiledCopy,
    tiled_copy_r2g: cute.TiledCopy,
):
    tid, _, _ = cute.arch.thread_idx()
    block_m, block_n, _ = cute.arch.block_idx()
    lane_id = tid % 32
    warp_id = tid // 32

    # Swap block id for transposed tile.
    cta_tile_a = cute.local_tile(mA, cta_tiler_a, (block_m, block_n))
    cta_tile_b = cute.local_tile(mB, cta_tiler_b, (block_n, block_m))

    # Smem tensor for one whole CTA tile.
    smem = SmemAllocator()
    smem_tile = smem.allocate_tensor(mA.element_type, smem_layout, byte_alignment=16)

    # G2S is CTA-wide tile copy, sliced by tid.
    thr_copy_g2s = tiled_copy_g2s.get_slice(tid)

    # Ldmatrix/R2G are warp-wide tile copies, sliced by lane_id.
    thr_ldmatrix = tiled_ldmatrix.get_slice(lane_id)
    thr_copy_r2g = tiled_copy_r2g.get_slice(lane_id)

    # A/B reg tiles use the same physical regs. B is just a logical view of A.
    reg_tile_a = cute.make_rmem_tensor(warp_tiler_a, mA.element_type)
    reg_tile_b = cute.make_tensor(
        reg_tile_a.iterator,
        cute.make_layout(warp_tiler_b, stride=(warp_tiler_b[1], 1)),
    )

    # Start stage 0 copy: GMEM -> SMEM.
    gmem_stage_tile = cute.local_tile(cta_tile_a, stage_tiler_a, (0, 0))
    smem_stage_tile = cute.local_tile(smem_tile, stage_tiler_a, (0, 0))
    cute.copy(
        tiled_copy_g2s,
        thr_copy_g2s.partition_S(gmem_stage_tile),
        thr_copy_g2s.partition_D(smem_stage_tile),
    )
    cute.arch.cp_async_commit_group()

    # Loop over all stages across the CTA tile's M dimension.
    num_warp_m_tiles = cta_tiler_a[0] // warp_tiler_a[0]
    for warp_m in cutlass.range(num_warp_m_tiles, unroll_full=True):
        # Wait for the previously issued copy for this stage.
        cute.arch.cp_async_wait_group(0)
        cute.arch.sync_threads()

        # If not at the last stage, start copying the next stage.
        if warp_m < num_warp_m_tiles - 1:
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

        # Ldmatrix copy: SMEM -> REG.
        smem_warp_tile = cute.local_tile(
            smem_tile,
            warp_tiler_a,
            (warp_m, warp_id),
        )
        cute.copy(
            tiled_ldmatrix,
            thr_ldmatrix.partition_S(smem_warp_tile),
            thr_ldmatrix.partition_D(reg_tile_a),
        )

        # Universal copy: REG -> GMEM.
        gmem_warp_tile_b = cute.local_tile(
            cta_tile_b,
            warp_tiler_b,
            (warp_id, warp_m),
        )
        cute.copy(
            tiled_copy_r2g,
            thr_copy_r2g.partition_S(reg_tile_b),
            thr_copy_r2g.partition_D(gmem_warp_tile_b),
        )


def _async_ldmatrix_transpose_impl(mA: cute.Tensor, mB: cute.Tensor, swizzle=None):
    tile_m = 128
    tile_n = 64
    threads_per_cta = 256
    cta_tiler_a = (tile_m, tile_n)
    cta_tiler_b = (tile_n, tile_m)

    warp_m = 32
    warp_n = 8
    threads_per_warp = 32
    warp_tiler_a = (warp_m, warp_n)
    warp_tiler_b = (warp_n, warp_m)
    stage_tiler_a = (warp_m, tile_n)

    assert mA.element_type is cutlass.Float16
    assert mB.element_type is cutlass.Float16
    assert mA.shape[0] == mB.shape[1]
    assert mA.shape[1] == mB.shape[0]
    assert mA.shape[0] % tile_m == 0
    assert mA.shape[1] % tile_n == 0
    assert threads_per_cta // threads_per_warp == tile_n // warp_tiler_a[1]
    assert stage_tiler_a[0] == warp_tiler_a[0]
    assert stage_tiler_a[1] == cta_tiler_a[1]

    # cp.async copy: GMEM -> SMEM.
    # CopyG2SOp requires an explicit per-instruction copy width.
    copy_bits = 128
    copy_elems = copy_bits // cutlass.Float16.width
    atom_async_copy = cute.make_copy_atom(
        cute.nvgpu.cpasync.CopyG2SOp(
            cache_mode=cute.nvgpu.cpasync.LoadCacheMode.GLOBAL
        ),
        cutlass.Float16,
        num_bits_per_copy=copy_bits,
    )

    # The 128x64 CTA tile is split into four 32x64 stages.
    # 256 CTA threads each issue exactly one 16B cp.async per stage.
    tiled_copy_g2s = make_tiled_copy_for_shape(
        atom_async_copy,
        stage_tiler_a,
        threads_per_cta,
        copy_elems,
    )

    # Transposed ldmatrix copy: SMEM -> REG.
    # One ldmatrix.m8n8.x4.trans reads four stacked 8x8 FP16 matrices,
    # which combine into a 32x8 A subtile.
    # Then transposes it into an 8x32 B subtile.
    # 8 warps distributed along N cover one 32x64 stage tile.
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
        warp_tiler_a,
    )

    # Universal copy: REG -> GMEM.
    # The 16B vector width helps coalescing but is not required for correctness.
    # This is a warp-level tiled copy over an 8x32 B subtile.
    atom_store = cute.make_copy_atom(
        cute.nvgpu.CopyUniversalOp(),
        cutlass.Float16,
        num_bits_per_copy=copy_bits,
    )
    tiled_copy_r2g = make_tiled_copy_for_shape(
        atom_store,
        warp_tiler_b,
        threads_per_warp,
        copy_elems,
    )

    # 128x64 row-major SMEM layout. N is contiguous for ldmatrix.x4.
    # It is split into four 32x64 stage tiles along M.
    source_layout = cute.make_layout(cta_tiler_a, stride=(tile_n, 1))

    if swizzle is None:
        smem_layout = source_layout
    else:
        smem_layout = cute.make_composed_layout(swizzle, 0, source_layout)

    # Kernel launch.
    grid_m = cute.ceil_div(mA.shape[0], tile_m)
    grid_n = cute.ceil_div(mA.shape[1], tile_n)
    async_ldmatrix_transpose_kernel(
        mA,
        mB,
        cta_tiler_a,
        cta_tiler_b,
        stage_tiler_a,
        warp_tiler_a,
        warp_tiler_b,
        smem_layout,
        tiled_copy_g2s,
        tiled_ldmatrix,
        tiled_copy_r2g,
    ).launch(
        grid=[grid_m, grid_n, 1],
        block=[threads_per_cta, 1, 1],
    )


@cute.jit
def async_ldmatrix_transpose_no_swizzle(mA: cute.Tensor, mB: cute.Tensor):
    _async_ldmatrix_transpose_impl(mA, mB)


@cute.jit
def async_ldmatrix_transpose_swizzle(mA: cute.Tensor, mB: cute.Tensor):
    # Same ldmatrix-friendly swizzle used by 02_ldmatrix_transpose.py.
    _async_ldmatrix_transpose_impl(mA, mB, cute.make_swizzle(2, 3, 3))


def run_async_ldmatrix_transpose_example():
    print("run_async_ldmatrix_transpose_example()")
    shape = (4096, 4096)

    a = torch.randn(shape, device="cuda", dtype=torch.float16)
    expected = a.T.contiguous()

    b_no_swizzle = torch.empty_like(expected)
    async_ldmatrix_transpose_no_swizzle(
        from_dlpack(a, assumed_align=16),
        from_dlpack(b_no_swizzle, assumed_align=16),
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(b_no_swizzle, expected)
    print("passed: no-swizzle staged async cp + ldmatrix B == A.T")

    b_swizzle = torch.empty_like(expected)
    async_ldmatrix_transpose_swizzle(
        from_dlpack(a, assumed_align=16),
        from_dlpack(b_swizzle, assumed_align=16),
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(b_swizzle, expected)
    print("passed: swizzle staged async cp + ldmatrix B == A.T")


if __name__ == "__main__":
    run_async_ldmatrix_transpose_example()
