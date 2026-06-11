import cutlass
import cutlass.cute as cute
import torch

from cutlass.cute.runtime import from_dlpack
from cutlass.utils import SmemAllocator


@cute.kernel
def fma_gemm_kernel(
    mA: cute.Tensor,
    mB: cute.Tensor,
    mC: cute.Tensor,
    cta_tiler_a: cute.Shape,
    cta_tiler_b: cute.Shape,
    cta_tiler_c: cute.Shape,
    copy_atom: cute.CopyAtom,
    copy_tv_layout_a: cute.Layout,
    copy_tv_layout_b: cute.Layout,
    mma_atom: cute.MmaAtom,
    mma_tv_layout_a: cute.Layout,
    mma_tv_layout_b: cute.Layout,
    mma_tv_layout_c: cute.Layout,
):
    tid, _, _ = cute.arch.thread_idx()
    block_m, block_n, _ = cute.arch.block_idx()

    # A/B Tiles on Gmem
    cta_tile_a = cute.local_tile(mA, cta_tiler_a, (block_m, None))
    cta_tile_b = cute.local_tile(mB, cta_tiler_b, (block_n, None))

    # A/B Tile on Smem
    smem = SmemAllocator()
    smem_tile_a = smem.allocate_tensor(mA.element_type, cute.make_layout(cta_tiler_a))
    smem_tile_b = smem.allocate_tensor(mB.element_type, cute.make_layout(cta_tiler_b))

    # Accumalator C in registers
    c_value_layout = cute.get(mma_tv_layout_c, mode=[1])
    accum_c = cute.make_rmem_tensor_like(c_value_layout, cutlass.Float16)
    accum_c.fill(0.0)

    mma_thread_coord = (tid, (None, None))
    num_k_tiles = cute.size(cta_tile_a, mode=[2])
    for k_tile in cutlass.range(num_k_tiles):
        # Slice the current CTA's A[M,K] and B[N,K] tiles from global memory.
        k_tile_coord = (None, None, k_tile)
        gmem_tile_a = cta_tile_a[k_tile_coord]
        gmem_tile_b = cta_tile_b[k_tile_coord]

        # Per-thread copy fragments: GMEM -> SMEM.
        copy_thread_coord = (tid, None)
        gmem_a_fragment = cute.composition(gmem_tile_a, copy_tv_layout_a)[copy_thread_coord]
        gmem_b_fragment = cute.composition(gmem_tile_b, copy_tv_layout_b)[copy_thread_coord]
        smem_a_fragment = cute.composition(smem_tile_a, copy_tv_layout_a)[copy_thread_coord]
        smem_b_fragment = cute.composition(smem_tile_b, copy_tv_layout_b)[copy_thread_coord]
        cute.copy(copy_atom, gmem_a_fragment, smem_a_fragment)
        cute.copy(copy_atom, gmem_b_fragment, smem_b_fragment)
        cute.arch.sync_threads()

        # Per-thread MMA
        smem_a_tile = cute.composition(smem_tile_a, mma_tv_layout_a)[mma_thread_coord]
        smem_b_tile = cute.composition(smem_tile_b, mma_tv_layout_b)[mma_thread_coord]
        cute.gemm(mma_atom, accum_c, smem_a_tile, smem_b_tile, accum_c)
        cute.arch.sync_threads()

    # Epilogue store
    cta_tile_c = cute.local_tile(mC, cta_tiler_c, (block_m, block_n))
    gmem_c_tile = cute.composition(cta_tile_c, mma_tv_layout_c)[mma_thread_coord]
    cute.copy(copy_atom, accum_c, gmem_c_tile)


@cute.jit
def fma_gemm(mA: cute.Tensor, mB: cute.Tensor, mC: cute.Tensor):
    assert mA.element_type is cutlass.Float16
    assert mB.element_type is cutlass.Float16
    assert mC.element_type is cutlass.Float16
    assert mA.shape[1] == mB.shape[1]
    assert mA.shape[0] == mC.shape[0]
    assert mB.shape[0] == mC.shape[1]

    # CTA size
    threads_per_cta = 256

    # Copy atom for GMEM -> SMEM
    copy_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), cutlass.Float16)

    # A/B tile in 128x32. Each thread owns an 1x32 fragment.
    cta_tiler_ab, copy_tv_layout_ab = cute.make_layout_tv(
        thr_layout=cute.make_layout((128, 2), stride=(2, 1)),
        val_layout=cute.make_layout((1, 32), stride=(32, 1)),
    )

    # MMA atom via FMA
    mma_atom = cute.make_mma_atom(cute.nvgpu.MmaUniversalOp(cutlass.Float16))

    # A/B tile in 128x64. Each thread owns an 8x64 fragment.
    # Note one thread owns whole 64 k-dim, so 16 threads cover the whole 128x64 tile.
    # tid in [0, ..., 255] maps to [tile_m, tile_n] via [tid % 16, tid // 16]
    # C[tile_m, tile_n] = A[tile_m] * B[tile_n]
    mma_tv_layout_a = cute.make_layout(((16, 16), (8, 64)), stride=((8, 0), (1, 128)))
    mma_tv_layout_b = cute.make_layout(((16, 16), (8, 64)), stride=((0, 8), (1, 128)))

    # C tile in 128x128. Each thread owns an 8x8 fragment.
    cta_tiler_c, mma_tv_layout_c = cute.make_layout_tv(
        thr_layout=cute.make_layout((16, 16), stride=(1, 16)),
        val_layout=cute.make_layout((8, 8), stride=(1, 8)),
    )

    assert mA.shape[0] % cta_tiler_c[0] == 0
    assert mB.shape[0] % cta_tiler_c[1] == 0
    assert mA.shape[1] % cta_tiler_ab[1] == 0

    grid_m, grid_n = cute.ceil_div(mC.shape, cta_tiler_c)

    fma_gemm_kernel(
        mA,
        mB,
        mC,
        cta_tiler_ab,
        cta_tiler_ab,
        cta_tiler_c,
        copy_atom,
        copy_tv_layout_ab,
        copy_tv_layout_ab,
        mma_atom,
        mma_tv_layout_a,
        mma_tv_layout_b,
        mma_tv_layout_c,
    ).launch(
        grid=[grid_m, grid_n, 1],
        block=[threads_per_cta, 1, 1],
    )


def run_fma_gemm_example():
    print("run_fma_gemm_example()")

    # No-bounds path: M/N/K must be divisible by the 128x128x64 CTA tile.
    m, n, k = 4096, 2048, 1024
    a = torch.randn((m, k), device="cuda", dtype=torch.float16)
    b = torch.randn((n, k), device="cuda", dtype=torch.float16)
    c = torch.empty((m, n), device="cuda", dtype=torch.float16)

    fma_gemm(from_dlpack(a), from_dlpack(b), from_dlpack(c))
    torch.cuda.synchronize()

    expected = a @ b.T
    torch.testing.assert_close(c, expected, rtol=5e-2, atol=1.5)
    print("passed: C == A @ B.T")
    print(c[:4, :6])


if __name__ == "__main__":
    run_fma_gemm_example()
