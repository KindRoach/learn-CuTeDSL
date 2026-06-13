import cutlass
import cutlass.cute as cute
import torch

from cutlass.cute.runtime import from_dlpack
from cutlass.utils import SmemAllocator


@cute.kernel
def fma_tile_gemm_kernel(
    mA: cute.Tensor,
    mB: cute.Tensor,
    mC: cute.Tensor,
    cta_tiler_a: cute.Shape,
    cta_tiler_b: cute.Shape,
    cta_tiler_c: cute.Shape,
    tiled_copy_ab: cute.TiledCopy,
    tiled_copy_c: cute.TiledCopy,
    tiled_mma: cute.TiledMma,
):
    tid, _, _ = cute.arch.thread_idx()
    block_m, block_n, _ = cute.arch.block_idx()

    # A/B Tiles and C tile on Gmem
    cta_tile_a = cute.local_tile(mA, cta_tiler_a, (block_m, None))
    cta_tile_b = cute.local_tile(mB, cta_tiler_b, (block_n, None))
    cta_tile_c = cute.local_tile(mC, cta_tiler_c, (block_m, block_n))

    # A/B Tile on Smem
    smem = SmemAllocator()
    smem_tile_a = smem.allocate_tensor(mA.element_type, cute.make_layout(cta_tiler_a))
    smem_tile_b = smem.allocate_tensor(mB.element_type, cute.make_layout(cta_tiler_b))

    # Current thread slices for tiled copy and tiled mma
    thr_copy_ab = tiled_copy_ab.get_slice(tid)
    thr_copy_c = tiled_copy_c.get_slice(tid)
    thr_mma = tiled_mma.get_slice(tid)

    # Accumalator C in registers
    gmem_c_fragment = thr_mma.partition_C(cta_tile_c)
    accum_c = cute.make_rmem_tensor_like(gmem_c_fragment, cutlass.Float16)
    accum_c.fill(0.0)

    # Mainloop over K dimension tiles
    num_k_tiles = cute.size(cta_tile_a, mode=[2])
    for k_tile in cutlass.range(num_k_tiles):
        k_tile_coord = (None, None, k_tile)
        gmem_tile_a = cta_tile_a[k_tile_coord]
        gmem_tile_b = cta_tile_b[k_tile_coord]

        # GMEM -> SMEM copy
        gmem_a_fragment = thr_copy_ab.partition_S(gmem_tile_a)
        gmem_b_fragment = thr_copy_ab.partition_S(gmem_tile_b)
        smem_a_fragment = thr_copy_ab.partition_D(smem_tile_a)
        smem_b_fragment = thr_copy_ab.partition_D(smem_tile_b)
        cute.copy(tiled_copy_ab, gmem_a_fragment, smem_a_fragment)
        cute.copy(tiled_copy_ab, gmem_b_fragment, smem_b_fragment)
        cute.arch.sync_threads()

        # MMA on the current K tile
        smem_a_fragment_mma = thr_mma.partition_A(smem_tile_a)
        smem_b_fragment_mma = thr_mma.partition_B(smem_tile_b)
        cute.gemm(
            tiled_mma,
            accum_c,
            smem_a_fragment_mma,
            smem_b_fragment_mma,
            accum_c,
        )
        cute.arch.sync_threads()

    # Epilogue store
    store_c_fragment = thr_copy_c.partition_D(cta_tile_c)
    store_accum_c = tiled_copy_c.retile(accum_c)
    cute.copy(tiled_copy_c, store_accum_c, store_c_fragment)


@cute.jit
def fma_tile_gemm(mA: cute.Tensor, mB: cute.Tensor, mC: cute.Tensor):
    assert mA.element_type is cutlass.Float16
    assert mB.element_type is cutlass.Float16
    assert mC.element_type is cutlass.Float16
    assert mA.shape[1] == mB.shape[1]
    assert mA.shape[0] == mC.shape[0]
    assert mB.shape[0] == mC.shape[1]

    threads_per_cta = 256

    copy_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), cutlass.Float16)

    # GMEM -> SMEM copy tile: 256 threads cover a 128x64 tile.
    # Each thread copies one contiguous 1x32 FP16 fragment.
    tiled_copy_ab = cute.make_tiled_copy_tv(
        copy_atom,
        thr_layout=cute.make_layout((128, 2), stride=(2, 1)),
        val_layout=cute.make_layout((1, 32), stride=(32, 1)),
    )

    # CTA-level tiled FMA: 256 threads cover 128x128x64 tile.
    # Each thread computes an 8x8x64 fragment.
    tiled_mma = cute.make_tiled_mma(
        cute.make_mma_atom(cute.nvgpu.MmaUniversalOp(cutlass.Float16)),
        atom_layout_mnk=cute.make_layout((16, 16, 1), stride=(1, 16, 0)),
        permutation_mnk=(128, 128, 64),
    )
    tiled_copy_c = cute.make_tiled_copy_C(copy_atom, tiled_mma)

    cta_tiler_a = (tiled_mma.get_tile_size(0), tiled_mma.get_tile_size(2))
    cta_tiler_b = (tiled_mma.get_tile_size(1), tiled_mma.get_tile_size(2))
    cta_tiler_c = (tiled_mma.get_tile_size(0), tiled_mma.get_tile_size(1))

    assert mA.shape[0] % cta_tiler_c[0] == 0
    assert mB.shape[0] % cta_tiler_c[1] == 0
    assert mA.shape[1] % cta_tiler_a[1] == 0

    grid_m, grid_n = cute.ceil_div(mC.shape, cta_tiler_c)

    fma_tile_gemm_kernel(
        mA,
        mB,
        mC,
        cta_tiler_a,
        cta_tiler_b,
        cta_tiler_c,
        tiled_copy_ab,
        tiled_copy_c,
        tiled_mma,
    ).launch(
        grid=[grid_m, grid_n, 1],
        block=[threads_per_cta, 1, 1],
    )


def run_fma_tile_gemm_example():
    print("run_fma_tile_gemm_example()")

    # No-bounds path: M/N/K must be divisible by the 128x128x64 CTA tile.
    m, n, k = 4096, 2048, 1024
    a = torch.randn((m, k), device="cuda", dtype=torch.float16)
    b = torch.randn((n, k), device="cuda", dtype=torch.float16)
    c = torch.empty((m, n), device="cuda", dtype=torch.float16)

    fma_tile_gemm(from_dlpack(a), from_dlpack(b), from_dlpack(c))
    torch.cuda.synchronize()

    expected = a @ b.T
    torch.testing.assert_close(c, expected, rtol=5e-2, atol=1.5)
    print("passed: C == A @ B.T")
    print(c[:4, :6])


if __name__ == "__main__":
    run_fma_tile_gemm_example()
