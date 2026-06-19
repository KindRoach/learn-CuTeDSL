import cutlass
import cutlass.cute as cute
import torch

from cutlass.cute.runtime import from_dlpack


@cute.kernel
def vector_add_tv_kernel_no_bounds(
    mA: cute.Tensor,
    mB: cute.Tensor,
    mC: cute.Tensor,
    tiler_mn: cute.Shape,
    tv_layout: cute.Layout,
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx_m, bidx_n, _ = cute.arch.block_idx()

    # 1. Select one CTA tile from the global tensors.
    blk_coord = (bidx_m, bidx_n)
    blkA = cute.local_tile(mA, tiler_mn, blk_coord)  # (TileM, TileN) -> address
    blkB = cute.local_tile(mB, tiler_mn, blk_coord)
    blkC = cute.local_tile(mC, tiler_mn, blk_coord)

    # 2. Compose the CTA tile tensor with the TV layout.
    #    blkA:      (TileM, TileN) -> address
    #    tv_layout: (tid, vid)     -> (TileM, TileN)
    #    tidfrgA:   (tid, vid)     -> address
    tidfrgA = cute.composition(blkA, tv_layout)
    tidfrgB = cute.composition(blkB, tv_layout)
    tidfrgC = cute.composition(blkC, tv_layout)

    # 3. Slice by the current thread id. Each thread gets a value fragment.
    thr_coord = (tidx, None)
    thrA = tidfrgA[thr_coord]  # vid -> address
    thrB = tidfrgB[thr_coord]
    thrC = tidfrgC[thr_coord]

    # 4. No predicate: every element in the CTA tile must be valid.
    thrC.store(thrA.load() + thrB.load())


@cute.jit
def vector_add_tv_no_bounds(mA: cute.Tensor, mB: cute.Tensor, mC: cute.Tensor):
    assert all(t.element_type == mA.element_type for t in [mA, mB, mC])
    dtype = mA.element_type

    threads_m = 8
    threads_n = 64
    threads_per_cta = threads_m * threads_n

    # CTA has 8 * 64 = 512 threads. Consecutive thread ids run along N.
    thr_layout = cute.make_layout(
        (threads_m, threads_n),
        stride=(threads_n, 1),
    )

    # Value layout starts from byte units to describe 16B vectorized access,
    # then is recast to element units for the current dtype.
    value_rows_per_thread = 16
    coalesced_ldst_bytes = 16
    coalesced_unit_bits = 8
    val_layout = cute.make_layout(
        (value_rows_per_thread, coalesced_ldst_bytes),
        stride=(coalesced_ldst_bytes, 1),
    )
    val_layout = cute.recast_layout(
        dtype.width,
        coalesced_unit_bits,
        val_layout,
    )

    # tiler_mn is the CTA tile shape.
    # tv_layout maps (tid, vid) to tile-local (m, n).
    tiler_mn, tv_layout = cute.make_layout_tv(thr_layout, val_layout)
    cute.printf("[no bounds] CTA tiler: {}", tiler_mn)
    cute.printf("[no bounds] TV layout: {}", tv_layout)

    # This no-bounds path assumes mA/mB/mC shapes are divisible by tiler_mn.
    tile_count_mn = cute.ceil_div(mC.shape, tiler_mn)

    vector_add_tv_kernel_no_bounds(mA, mB, mC, tiler_mn, tv_layout).launch(
        grid=[tile_count_mn[0], tile_count_mn[1], 1],
        block=[threads_per_cta, 1, 1],
    )


@cute.kernel
def vector_add_tv_kernel_with_bounds(
    mA: cute.Tensor,
    mB: cute.Tensor,
    mC: cute.Tensor,
    cC: cute.Tensor,
    tiler_mn: cute.Shape,
    tv_layout: cute.Layout,
    tensor_shape: cute.Shape,
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx_m, bidx_n, _ = cute.arch.block_idx()

    # 1. Select one CTA tile and the matching coordinate tile.
    blk_coord = (bidx_m, bidx_n)
    blkA = cute.local_tile(mA, tiler_mn, blk_coord)  # (TileM, TileN) -> address
    blkB = cute.local_tile(mB, tiler_mn, blk_coord)
    blkC = cute.local_tile(mC, tiler_mn, blk_coord)
    blkCoord = cute.local_tile(cC, tiler_mn, blk_coord)  # (TileM, TileN) -> (m, n)

    # 2. Compose the CTA tile tensor with the TV layout.
    #    blkA:      (TileM, TileN) -> address
    #    tv_layout: (tid, vid)     -> (TileM, TileN)
    #    tidfrgA:   (tid, vid)     -> address
    tidfrgA = cute.composition(blkA, tv_layout)
    tidfrgB = cute.composition(blkB, tv_layout)
    tidfrgC = cute.composition(blkC, tv_layout)
    tidfrgCoord = cute.composition(blkCoord, tv_layout)

    # 3. Slice by the current thread id. Each thread gets a value fragment
    #    and the corresponding logical coordinates.
    thr_coord = (tidx, None)
    thrA = tidfrgA[thr_coord]  # vid -> address
    thrB = tidfrgB[thr_coord]
    thrC = tidfrgC[thr_coord]
    thrCoord = tidfrgCoord[thr_coord]  # vid -> (m, n)

    # 4. Predicate each value with the identity tensor coordinate.
    #    Invalid values belong to the padded part of the CTA tile, so skip them.
    for vid in cutlass.range_constexpr(cute.size(thrC)):
        pred = cute.elem_less(thrCoord[vid], tensor_shape)
        if pred:
            thrC[vid] = thrA[vid] + thrB[vid]


@cute.jit
def vector_add_tv_with_bounds(mA: cute.Tensor, mB: cute.Tensor, mC: cute.Tensor):
    assert all(t.element_type == mA.element_type for t in [mA, mB, mC])
    dtype = mA.element_type

    threads_m = 8
    threads_n = 64
    threads_per_cta = threads_m * threads_n

    # CTA has 8 * 64 = 512 threads. Consecutive thread ids run along N.
    thr_layout = cute.make_layout(
        (threads_m, threads_n),
        stride=(threads_n, 1),
    )

    # Value layout starts from byte units to describe 16B vectorized access,
    # then is recast to element units for the current dtype.
    value_rows_per_thread = 16
    coalesced_ldst_bytes = 16
    coalesced_unit_bits = 8
    val_layout = cute.make_layout(
        (value_rows_per_thread, coalesced_ldst_bytes),
        stride=(coalesced_ldst_bytes, 1),
    )
    val_layout = cute.recast_layout(
        dtype.width,
        coalesced_unit_bits,
        val_layout,
    )

    # tiler_mn is the CTA tile shape.
    # tv_layout maps (tid, vid) to tile-local (m, n).
    tiler_mn, tv_layout = cute.make_layout_tv(thr_layout, val_layout)
    cute.printf("[with bounds] CTA tiler: {}", tiler_mn)
    cute.printf("[with bounds] TV layout: {}", tv_layout)

    # Identity tensor gives a coordinate for every logical element in mC.
    cC = cute.make_identity_tensor(mC.shape)
    tile_count_mn = cute.ceil_div(mC.shape, tiler_mn)

    vector_add_tv_kernel_with_bounds(
        mA,
        mB,
        mC,
        cC,
        tiler_mn,
        tv_layout,
        mC.shape,
    ).launch(
        grid=[tile_count_mn[0], tile_count_mn[1], 1],
        block=[threads_per_cta, 1, 1],
    )


def run_vector_add_no_bounds_example():
    print("run_vector_add_no_bounds_example()")

    # For float32 this TV layout covers a CTA tile of (128, 256).
    # The no-bounds version deliberately uses a divisible shape.
    shape = (128, 256)
    a = torch.arange(0, shape[0] * shape[1], device="cuda", dtype=torch.float32)
    a = a.reshape(shape)
    b = torch.full(shape, 2.0, device="cuda", dtype=torch.float32)
    c = torch.empty_like(a)

    vector_add_tv_no_bounds(from_dlpack(a), from_dlpack(b), from_dlpack(c))
    torch.cuda.synchronize()

    expected = a + b
    torch.testing.assert_close(c, expected)
    print("passed: no-bounds C == A + B")
    print(c[:2, :8])
    print()


def run_vector_add_with_bounds_example():
    # Check for out-of-bounds accesses with:
    #   compute-sanitizer --tool memcheck python 02_layout/08_layout_kernel.py
    
    print("run_vector_add_with_bounds_example()")
    
    # For float32 this TV layout covers a CTA tile of (128, 256).
    # This shape intentionally leaves the last row/column outside the tensor.
    shape = (127, 255)
    a = torch.arange(0, shape[0] * shape[1], device="cuda", dtype=torch.float32)
    a = a.reshape(shape)
    b = torch.full(shape, 2.0, device="cuda", dtype=torch.float32)
    c = torch.empty_like(a)

    vector_add_tv_with_bounds(from_dlpack(a), from_dlpack(b), from_dlpack(c))
    torch.cuda.synchronize()

    expected = a + b
    torch.testing.assert_close(c, expected)
    print("passed: with-bounds C == A + B")
    print(c[:2, :8])


if __name__ == "__main__":
    run_vector_add_no_bounds_example()
    run_vector_add_with_bounds_example()
