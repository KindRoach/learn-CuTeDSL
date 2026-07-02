import cutlass
import cutlass.cute as cute
import torch

from cutlass.cute.runtime import from_dlpack
from cutlass.utils import SmemAllocator


@cute.kernel
def ampere_sol_gemm_kernel(
    mA: cute.Tensor,
    mB: cute.Tensor,
    mC: cute.Tensor,
    cta_tiler_a: cute.Shape,
    cta_tiler_b: cute.Shape,
    cta_tiler_c: cute.Shape,
    smem_layout_a: cute.ComposedLayout,
    smem_layout_b: cute.ComposedLayout,
    smem_layout_c: cute.ComposedLayout,
    tiled_copy_a: cute.TiledCopy,
    tiled_copy_b: cute.TiledCopy,
    tiled_copy_c: cute.TiledCopy,
    tiled_ldmatrix_a: cute.TiledCopy,
    tiled_ldmatrix_b: cute.TiledCopy,
    tiled_mma: cute.TiledMma,
    num_stages: cutlass.Constexpr,
):
    assert num_stages >= 2

    tid, _, _ = cute.arch.thread_idx()
    block_m, block_n, _ = cute.arch.block_idx()

    # CTA tiles on GMEM.
    cta_tile_a = cute.local_tile(mA, cta_tiler_a, (block_m, None))
    cta_tile_b = cute.local_tile(mB, cta_tiler_b, (block_n, None))
    cta_tile_c = cute.local_tile(mC, cta_tiler_c, (block_m, block_n))

    # A/B are dead before the epilogue, so C can alias the same SMEM storage.
    # cute.struct computes the field sizes, alignments, and tensor views; the
    # reuse itself comes from interpreting the same allocation as either type.
    @cute.struct
    class SharedStorageAB:
        a: cute.struct.Align[cute.struct.MemRange[mA.element_type, cute.cosize(smem_layout_a)], 16]
        b: cute.struct.Align[cute.struct.MemRange[mB.element_type, cute.cosize(smem_layout_b)], 16]

    @cute.struct
    class SharedStorageC:
        c: cute.struct.Align[cute.struct.MemRange[mC.element_type, cute.cosize(smem_layout_c)], 16]

    # Allocate only the larger lifetime region, rather than A/B plus C.
    smem = SmemAllocator()
    storage = smem.allocate(
        max(SharedStorageAB.size_in_bytes(), SharedStorageC.size_in_bytes()),
        byte_alignment=16,
    )
    smem_tile_a = SharedStorageAB(storage).a.get_tensor(smem_layout_a)
    smem_tile_b = SharedStorageAB(storage).b.get_tensor(smem_layout_b)
    smem_tile_c = SharedStorageC(storage).c.get_tensor(smem_layout_c)

    # A/B thread fragments for cp.async copy
    thr_copy_a = tiled_copy_a.get_slice(tid)
    thr_copy_b = tiled_copy_b.get_slice(tid)
    gmem_a_copy_fragment = thr_copy_a.partition_S(cta_tile_a)
    gmem_b_copy_fragment = thr_copy_b.partition_S(cta_tile_b)
    smem_a_copy_fragment = thr_copy_a.partition_D(smem_tile_a)
    smem_b_copy_fragment = thr_copy_b.partition_D(smem_tile_b)

    # A/B/C thread fragments for MMA
    thr_mma = tiled_mma.get_slice(tid)
    smem_a_mma_fragment = thr_mma.partition_A(smem_tile_a)
    smem_b_mma_fragment = thr_mma.partition_B(smem_tile_b)
    smem_c_mma_fragment = thr_mma.partition_C(smem_tile_c)
    gmem_c_mma_fragment = thr_mma.partition_C(cta_tile_c)
    reg_a = tiled_mma.make_fragment_A(smem_a_mma_fragment[None, None, None, 0])
    reg_b = tiled_mma.make_fragment_B(smem_b_mma_fragment[None, None, None, 0])
    accum_c = tiled_mma.make_fragment_C(gmem_c_mma_fragment)
    accum_c.fill(0.0)

    # A/B thread fragments for ldmatrix copy
    thr_ldmatrix_a = tiled_ldmatrix_a.get_slice(tid)
    thr_ldmatrix_b = tiled_ldmatrix_b.get_slice(tid)
    smem_a_ldmatrix_fragment = thr_ldmatrix_a.partition_S(smem_tile_a)
    smem_b_ldmatrix_fragment = thr_ldmatrix_b.partition_S(smem_tile_b)
    reg_a_ldmatrix_fragment = thr_ldmatrix_a.retile(reg_a)
    reg_b_ldmatrix_fragment = thr_ldmatrix_b.retile(reg_b)

    # Prologue: prefetch num_stages - 1 K tiles.
    for stage in range(num_stages - 1):
        cute.copy(
            tiled_copy_a,
            gmem_a_copy_fragment[None, None, None, stage],
            smem_a_copy_fragment[None, None, None, stage],
        )
        cute.copy(
            tiled_copy_b,
            gmem_b_copy_fragment[None, None, None, stage],
            smem_b_copy_fragment[None, None, None, stage],
        )
        cute.arch.cp_async_commit_group()

    # init states for multi stage mma pipeline
    smem_pipe_read = 0
    smem_pipe_write = num_stages - 1
    next_k_tile = num_stages - 1

    # Mainloops
    num_k_tiles = cute.size(cta_tile_a, mode=[2])
    for _ in cutlass.range(num_k_tiles):
        # Wait until the current read stage is ready.
        cute.arch.cp_async_wait_group(num_stages - 2)

        # CTA sync is necessary here as wait is thread wide.
        cute.arch.sync_threads()

        # Load the current K tile from SMEM into registers.
        cute.copy(
            tiled_ldmatrix_a,
            smem_a_ldmatrix_fragment[None, None, None, smem_pipe_read],
            reg_a_ldmatrix_fragment,
        )
        cute.copy(
            tiled_ldmatrix_b,
            smem_b_ldmatrix_fragment[None, None, None, smem_pipe_read],
            reg_b_ldmatrix_fragment,
        )

        # Prefetch the next not-yet-issued K tile into the write stage.
        if next_k_tile < num_k_tiles:
            cute.copy(
                tiled_copy_a,
                gmem_a_copy_fragment[None, None, None, next_k_tile],
                smem_a_copy_fragment[None, None, None, smem_pipe_write],
            )
            cute.copy(
                tiled_copy_b,
                gmem_b_copy_fragment[None, None, None, next_k_tile],
                smem_b_copy_fragment[None, None, None, smem_pipe_write],
            )

        # Commit an empty group during drain iterations to preserve the
        # wait_group(num_stages - 2) distance for the final real async copy.
        cute.arch.cp_async_commit_group()

        # Compute the current K tile.
        cute.gemm(tiled_mma, accum_c, reg_a, reg_b, accum_c)

        # Advance the GMEM tile index and the SMEM ring buffer.
        next_k_tile = next_k_tile + 1
        smem_pipe_write = smem_pipe_read
        smem_pipe_read = smem_pipe_read + 1
        if smem_pipe_read == num_stages:
            smem_pipe_read = 0

    # Conservatively drain all committed cp.async groups before the epilogue.
    cute.arch.cp_async_wait_group(0)
    cute.arch.sync_threads()

    # ===== Epilogue =====
    # fp32 -> fp16 in REG
    result_c = cute.make_fragment_like(accum_c, mC.element_type)
    result_c[None] = accum_c.load().to(mC.element_type)

    # store the MMA-owned register C fragment into smem.
    cute.autovec_copy(result_c, smem_c_mma_fragment)
    cute.arch.sync_threads()

    # copy the smem C tile into gmem via vectorized store.
    thr_copy_c = tiled_copy_c.get_slice(tid)
    smem_c_store_fragment = thr_copy_c.partition_S(smem_tile_c)
    gmem_c_store_fragment = thr_copy_c.partition_D(cta_tile_c)
    cute.copy(tiled_copy_c, smem_c_store_fragment, gmem_c_store_fragment)


def make_smem_layout(smem_shape, swizzle):
    """Tile a swizzled, contiguous-major atom to an SMEM shape."""
    # Step 1: Build the smallest atom containing every swizzle state.
    # A B-bit XOR mask has 2**B states, so its row pattern repeats after
    # 2**B rows. The atom's second mode remains fully contiguous.
    swizzle_rows = 1 << swizzle.num_bits
    major_mode_size = smem_shape[1]
    layout_atom_outer = cute.make_layout(
        (swizzle_rows, major_mode_size),
        stride=(major_mode_size, 1),
    )

    # Step 2: Compose the logical atom with the requested swizzle.
    layout_atom = cute.make_composed_layout(
        swizzle,
        0,
        layout_atom_outer,
    )

    # Step 3: Repeat the swizzled atom to cover the complete SMEM shape.
    # Natural mode order keeps earlier modes tighter and the stage mode outermost.
    return cute.tile_to_shape(
        layout_atom,
        smem_shape,
        tuple(range(len(smem_shape))),
    )


def make_vectorized_tiled_copy(copy_atom, threads_per_cta, tile_shape, copy_elems):
    """Build a vectorized TiledCopy for a rectangular 2-D tile.

    Threads are arranged along the contiguous second mode first. Each thread
    copies ``copy_elems`` adjacent values, and the remaining thread dimension
    covers the first mode. CuTe repeats this pattern along the first mode when
    it is smaller than the full tile.
    """
    tile_outer, tile_contiguous = tile_shape

    # One thread owns one vector, so covering a complete row requires one
    # thread for every copy_elems values along the contiguous second mode.
    assert tile_contiguous % copy_elems == 0
    threads_contiguous = tile_contiguous // copy_elems

    # Put all remaining threads along the first mode. Require the resulting
    # thread tile to divide the full tile exactly so it has no residue.
    assert threads_per_cta % threads_contiguous == 0
    threads_outer = threads_per_cta // threads_contiguous
    assert tile_outer % threads_outer == 0

    # Map consecutive thread IDs along the contiguous mode, then the outer
    # mode. Together with value_layout, this covers
    # (threads_outer, tile_contiguous) elements per thread/value tile.
    thread_layout = cute.make_layout(
        (threads_outer, threads_contiguous),
        stride=(threads_contiguous, 1),
    )

    # Each thread copies one vector along the contiguous second mode.
    value_layout = cute.make_layout((1, copy_elems), stride=(copy_elems, 1))

    # Partitioning the full tensor repeats this thread/value tile
    # tile_outer / threads_outer times along the first mode.
    return cute.make_tiled_copy_tv(copy_atom, thread_layout, value_layout)


@cute.jit
def ampere_sol_gemm(
    mA: cute.Tensor,
    mB: cute.Tensor,
    mC: cute.Tensor,
):
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
    num_stages = 3
    cta_tiler_a = (cta_m, cta_k)
    cta_tiler_b = (cta_n, cta_k)
    cta_tiler_c = (cta_m, cta_n)

    # Mainloop GMEM -> SMEM cp.async copy: 256 threads cover a 128x64 A/B tile.
    # Each thread owns one contiguous 1x8 FP16 value tile per repetition.
    # 8 threads cover one 1x64 row, so 256 threads cover a 32x64 tile.
    # TiledCopy partitions the full 128x64 CTA tile, and cute.copy() executes
    # the resulting 4 repetitions along the first mode. Each thread loads
    # 4x8 FP16 values in total.
    copy_bits = 128
    copy_elems = copy_bits // cutlass.Float16.width
    async_copy_atom = cute.make_copy_atom(
        cute.nvgpu.cpasync.CopyG2SOp(cache_mode=cute.nvgpu.cpasync.LoadCacheMode.GLOBAL),
        cutlass.Float16,
        num_bits_per_copy=copy_bits,
    )
    tiled_copy_a = make_vectorized_tiled_copy(
        async_copy_atom,
        threads_per_cta,
        cta_tiler_a,
        copy_elems,
    )
    tiled_copy_b = make_vectorized_tiled_copy(
        async_copy_atom,
        threads_per_cta,
        cta_tiler_b,
        copy_elems,
    )

    # SMEM -> REG ldmatrix copy
    # One ldmatrix.m8n8.x4 reads four 8x8 FP16 matrices: an 8x32 tile.
    # A 16x2 grid of ldmatrix.m8n8.x4 cover 128x64 A/B tile.
    ldmatrix_copy_atom = cute.make_copy_atom(
        cute.nvgpu.warp.LdMatrix8x8x16bOp(
            transpose=False,
            num_matrices=4,
        ),
        cutlass.Float16,
    )

    # Multi contiguous 128x64 ring-buffer stages. K remains contiguous for
    # ldmatrix.x4 while the final mode selects the pipeline stage.
    # The final mode adds num_stages copies of the 128x64 CTA tile in SMEM.
    smem_layout_a = make_smem_layout(
        (cta_m, cta_k, num_stages),
        cute.make_swizzle(3, 3, 3),
    )
    smem_layout_b = make_smem_layout(
        (cta_n, cta_k, num_stages),
        cute.make_swizzle(3, 3, 3),
    )

    # the C tile use smem as middle buffer for vectorized store
    # only one stage is needed for a CTA
    smem_layout_c = make_smem_layout(
        (cta_m, cta_n),
        cute.make_swizzle(3, 3, 4),
    )

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

    # Epilogue SMEM -> GMEM logical copy: 256 threads cover a 128x128 C tile.
    # Each thread owns one contiguous 1x8 FP16 value tile per repetition.
    # 16 threads cover one 1x128 row, so 256 threads cover a 16x128 tile.
    # TiledCopy partitions the full 128x128 CTA tile, and cute.copy() executes
    # the resulting 8 repetitions along M. Each thread stores 8x8 FP16 total.
    vector_copy_atom = cute.make_copy_atom(
        cute.nvgpu.CopyUniversalOp(),
        cutlass.Float16,
        num_bits_per_copy=copy_bits,
    )
    tiled_copy_c = make_vectorized_tiled_copy(
        vector_copy_atom,
        threads_per_cta,
        cta_tiler_c,
        copy_elems,
    )

    assert mA.shape[0] % cta_tiler_c[0] == 0
    assert mB.shape[0] % cta_tiler_c[1] == 0
    assert mA.shape[1] % cta_tiler_a[1] == 0
    assert mA.shape[1] >= (num_stages - 1) * cta_k

    grid_m, grid_n = cute.ceil_div(mC.shape, cta_tiler_c)
    ampere_sol_gemm_kernel(
        mA,
        mB,
        mC,
        cta_tiler_a,
        cta_tiler_b,
        cta_tiler_c,
        smem_layout_a,
        smem_layout_b,
        smem_layout_c,
        tiled_copy_a,
        tiled_copy_b,
        tiled_copy_c,
        tiled_ldmatrix_a,
        tiled_ldmatrix_b,
        tiled_mma,
        num_stages,
    ).launch(
        grid=[grid_m, grid_n, 1],
        block=[threads_per_cta, 1, 1],
    )


def run_ampere_sol_gemm_example():
    print("run_ampere_sol_gemm_example()")

    m, n, k = 4096, 2048, 1024
    a = torch.randn((m, k), device="cuda", dtype=torch.float16)
    b = torch.randn((n, k), device="cuda", dtype=torch.float16)
    c = torch.empty((m, n), device="cuda", dtype=torch.float16)

    ampere_sol_gemm(
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
    run_ampere_sol_gemm_example()
