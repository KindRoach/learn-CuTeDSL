import cutlass
import cutlass.cute as cute
import torch

from cutlass.cute.runtime import from_dlpack
from cutlass.utils import SmemAllocator


@cute.kernel
def ldmatrix_mma_gemm_kernel(
    mA: cute.Tensor,
    mB: cute.Tensor,
    mC: cute.Tensor,
    cta_tiler_a: cute.Shape,
    cta_tiler_b: cute.Shape,
    cta_tiler_c: cute.Shape,
    smem_layout_a: cute.Layout,
    smem_layout_b: cute.Layout,
    tiled_copy_a: cute.TiledCopy,
    tiled_copy_b: cute.TiledCopy,
    tiled_copy_c: cute.TiledCopy,
    tiled_ldmatrix_a: cute.TiledCopy,
    tiled_ldmatrix_b: cute.TiledCopy,
    tiled_mma: cute.TiledMma,
):
    tid, _, _ = cute.arch.thread_idx()
    block_m, block_n, _ = cute.arch.block_idx()

    # A/B Tiles and C tile on Gmem
    cta_tile_a = cute.local_tile(mA, cta_tiler_a, (block_m, None))
    cta_tile_b = cute.local_tile(mB, cta_tiler_b, (block_n, None))
    cta_tile_c = cute.local_tile(mC, cta_tiler_c, (block_m, block_n))

    # A/B Tile on Smem with ldmatrix-friendly layout
    smem = SmemAllocator()
    smem_tile_a = smem.allocate_tensor(mA.element_type, smem_layout_a, byte_alignment=16)
    smem_tile_b = smem.allocate_tensor(mB.element_type, smem_layout_b, byte_alignment=16)

    # Current thread slices for tiled copy and tiled mma
    thr_copy_a = tiled_copy_a.get_slice(tid)
    thr_copy_b = tiled_copy_b.get_slice(tid)
    thr_copy_c = tiled_copy_c.get_slice(tid)
    thr_mma = tiled_mma.get_slice(tid)

    # A/B/C Reg for mma
    smem_a_mma_fragment = thr_mma.partition_A(smem_tile_a)
    smem_b_mma_fragment = thr_mma.partition_B(smem_tile_b)
    gmem_c_mma_fragment = thr_mma.partition_C(cta_tile_c)
    reg_a = tiled_mma.make_fragment_A(smem_a_mma_fragment)
    reg_b = tiled_mma.make_fragment_B(smem_b_mma_fragment)
    accum_c = tiled_mma.make_fragment_C(gmem_c_mma_fragment)
    accum_c.fill(0.0)

    # A/B Reg for ldmatrix copy
    thr_ldmatrix_a = tiled_ldmatrix_a.get_slice(tid)
    thr_ldmatrix_b = tiled_ldmatrix_b.get_slice(tid)
    smem_a_ldmatrix_fragment = thr_ldmatrix_a.partition_S(smem_tile_a)
    smem_b_ldmatrix_fragment = thr_ldmatrix_b.partition_S(smem_tile_b)
    reg_a_ldmatrix_fragment = thr_ldmatrix_a.retile(reg_a)
    reg_b_ldmatrix_fragment = thr_ldmatrix_b.retile(reg_b)

    # Mainloop
    num_k_tiles = cute.size(cta_tile_a, mode=[2])
    for k_tile in cutlass.range(num_k_tiles):
        k_tile_coord = (None, None, k_tile)
        gmem_tile_a = cta_tile_a[k_tile_coord]
        gmem_tile_b = cta_tile_b[k_tile_coord]

        # GMEM -> SMEM universal copy
        gmem_a_fragment = thr_copy_a.partition_S(gmem_tile_a)
        gmem_b_fragment = thr_copy_b.partition_S(gmem_tile_b)
        smem_a_copy_fragment = thr_copy_a.partition_D(smem_tile_a)
        smem_b_copy_fragment = thr_copy_b.partition_D(smem_tile_b)
        cute.copy(tiled_copy_a, gmem_a_fragment, smem_a_copy_fragment)
        cute.copy(tiled_copy_b, gmem_b_fragment, smem_b_copy_fragment)
        cute.arch.sync_threads()

        # SMEM -> REG ldmatrix copy
        cute.copy(tiled_ldmatrix_a, smem_a_ldmatrix_fragment, reg_a_ldmatrix_fragment)
        cute.copy(tiled_ldmatrix_b, smem_b_ldmatrix_fragment, reg_b_ldmatrix_fragment)

        # MMA on REG
        cute.gemm(tiled_mma, accum_c, reg_a, reg_b, accum_c)
        cute.arch.sync_threads()

    # Epilogue: fp32 -> fp16 + store
    result_c = cute.make_fragment_like(accum_c, mC.element_type)
    result_c[None] = accum_c.load().to(mC.element_type)
    store_c_fragment = thr_copy_c.partition_D(cta_tile_c)
    store_result_c = tiled_copy_c.retile(result_c)
    cute.copy(tiled_copy_c, store_result_c, store_c_fragment)


@cute.jit
def ldmatrix_mma_gemm(mA: cute.Tensor, mB: cute.Tensor, mC: cute.Tensor):
    assert mA.element_type is cutlass.Float16
    assert mB.element_type is cutlass.Float16
    assert mC.element_type is cutlass.Float16
    assert mA.shape[1] == mB.shape[1]
    assert mA.shape[0] == mC.shape[0]
    assert mB.shape[0] == mC.shape[1]

    # 8 warps/256 threads per CTA
    threads_per_cta = 256

    # CTA Problem Shape: 128x128x64
    cta_m, cta_n, cta_k = 128, 128, 64
    cta_tiler_a = (cta_m, cta_k)
    cta_tiler_b = (cta_n, cta_k)
    cta_tiler_c = (cta_m, cta_n)

    # GMEM -> SMEM Universal copy. 256 threads cover a 128x64 tile.
    # Each thread copies one contiguous 1x32 FP16 fragment.
    universal_copy_atom = cute.make_copy_atom(
        cute.nvgpu.CopyUniversalOp(),
        cutlass.Float16,
    )
    copy_thr_layout = cute.make_layout((128, 2), stride=(2, 1))
    copy_val_layout = cute.make_layout((1, 32), stride=(32, 1))
    tiled_copy_a = cute.make_tiled_copy_tv(
        universal_copy_atom,
        thr_layout=copy_thr_layout,
        val_layout=copy_val_layout,
    )
    tiled_copy_b = cute.make_tiled_copy_tv(
        universal_copy_atom,
        thr_layout=copy_thr_layout,
        val_layout=copy_val_layout,
    )

    # SMEM -> REG ldmatrix Copy
    # One ldmatrix.m8n8.x4 reads four 8x8 FP16 matrices: an 8x32 tile.
    # A 16x2 grid of ldmatrix.m8n8.x4 cover 128x64 A/B tile.
    ldmatrix_copy_atom = cute.make_copy_atom(
        cute.nvgpu.warp.LdMatrix8x8x16bOp(
            transpose=False,
            num_matrices=4,
        ),
        cutlass.Float16,
    )

    # 128x64 A/B row-major SMEM layout. K is contiguous for ldmatrix.x4.
    smem_layout_a = cute.make_layout((cta_m, cta_k), stride=(cta_k, 1))
    smem_layout_b = cute.make_layout((cta_n, cta_k), stride=(cta_k, 1))

    # One warp-level mma.m16n8k16 covers 16x8x16.
    # The 8 CTA warps are arranged as (4,2,1) MMA atoms.
    # cute.gemm expands this tiled MMA over the full 128x128x64 CTA tile.
    mma_warp_m, mma_warp_n, mma_warp_k = 4, 2, 1
    assert threads_per_cta == mma_warp_m * mma_warp_n * mma_warp_k * 32
    tiled_mma = cute.make_tiled_mma(
        cute.nvgpu.warp.MmaF16BF16Op(
            cutlass.Float16,
            cutlass.Float32,
            (16, 8, 16),
        ),
        atom_layout_mnk=cute.make_layout((mma_warp_m, mma_warp_n, mma_warp_k)),
        permutation_mnk=(cta_m, cta_n, cta_k),
    )

    # CuTe DSL auto match register from ldmatrix to tiled_mma
    tiled_ldmatrix_a = cute.make_tiled_copy_A(ldmatrix_copy_atom, tiled_mma)
    tiled_ldmatrix_b = cute.make_tiled_copy_B(ldmatrix_copy_atom, tiled_mma)

    # REG -> GMEM Universal copy. 256 threads cover a 128x128 tile.
    tiled_copy_c = cute.make_tiled_copy_C(universal_copy_atom, tiled_mma)

    assert mA.shape[0] % cta_tiler_c[0] == 0
    assert mB.shape[0] % cta_tiler_c[1] == 0
    assert mA.shape[1] % cta_tiler_a[1] == 0

    grid_m, grid_n = cute.ceil_div(mC.shape, cta_tiler_c)

    ldmatrix_mma_gemm_kernel(
        mA,
        mB,
        mC,
        cta_tiler_a,
        cta_tiler_b,
        cta_tiler_c,
        smem_layout_a,
        smem_layout_b,
        tiled_copy_a,
        tiled_copy_b,
        tiled_copy_c,
        tiled_ldmatrix_a,
        tiled_ldmatrix_b,
        tiled_mma,
    ).launch(
        grid=[grid_m, grid_n, 1],
        block=[threads_per_cta, 1, 1],
    )


def run_ldmatrix_mma_gemm_example():
    print("run_ldmatrix_mma_gemm_example()")

    # No-bounds path: M/N/K must be divisible by the 128x128x64 CTA tile.
    m, n, k = 4096, 2048, 1024
    a = torch.randn((m, k), device="cuda", dtype=torch.float16)
    b = torch.randn((n, k), device="cuda", dtype=torch.float16)
    c = torch.empty((m, n), device="cuda", dtype=torch.float16)

    ldmatrix_mma_gemm(
        from_dlpack(a, assumed_align=16),
        from_dlpack(b, assumed_align=16),
        from_dlpack(c, assumed_align=16),
    )
    torch.cuda.synchronize()

    expected = a @ b.T
    torch.testing.assert_close(c, expected, rtol=5e-2, atol=1.5)
    print("passed: C == A @ B.T")
    print(c[:4, :6])


if __name__ == "__main__":
    run_ldmatrix_mma_gemm_example()
