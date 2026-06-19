import cutlass
import cutlass.cute as cute

from tensor_layouts import ComposedLayout, Layout, Swizzle

from .utils import visualize_layout


def simple_swizzle_layout():
    print("simple_swizzle_layout()")

    shape = (32, 32)
    stride = (32, 1)
    b, m, s = 2, 4, 3

    @cute.jit
    def cute_jit():
        # original smem layout
        layout = cute.make_layout(shape, stride=stride)

        # create a swizzle S<2,4,3> mannually: BBits, MBase, SShift
        swizzle = cute.make_swizzle(b, m, s)

        # apply the swizzle to the original layout
        swizzled_layout = cute.make_composed_layout(swizzle, 0, layout)

        cute.printf("layout:          {}", layout)
        cute.printf("swizzle:         S<2,4,3>")
        cute.printf("swizzled layout: {}", swizzled_layout)
        cute.printf("coord\t\tlayout(coord)\tswizzled(coord)")
        for coord in ((0, 0), (3, 16), (4, 16), (7, 25), (11, 26), (17, 31), (31, 31)):
            cute.printf("{}\t\t{}\t\t{}", coord, layout(coord), swizzled_layout(coord))

    cute_jit()

    layout = Layout(shape, stride)
    swizzled_layout = ComposedLayout(Swizzle(b, m, s), layout)

    visualize_layout(layout, file_name="swizzle_layout_before")
    visualize_layout(swizzled_layout, file_name="swizzle_layout_after")
    print()


def warpgroup_swizzle_layout():
    print("warpgroup_swizzle_layout()")

    tile_shape = (32, 32)

    @cute.jit
    def cute_jit():
        helper_atom = cute.nvgpu.warpgroup.make_smem_layout_atom(
            cute.nvgpu.warpgroup.SmemLayoutAtomKind.K_SW64,
            cutlass.BFloat16,
        )
        swizzled_layout = cute.tile_to_shape(helper_atom, tile_shape, order=(0, 1))

        cute.printf("helper K_SW64 atom: {}", helper_atom)
        cute.printf("tiled to {}: {}", tile_shape, swizzled_layout)
        cute.printf("coord\t\tswizzled(coord)")
        for coord in ((0, 0), (3, 16), (4, 16), (7, 25), (11, 26), (17, 31), (31, 31)):
            cute.printf("{}\t\t{}", coord, swizzled_layout(coord))

    cute_jit()
    print()


if __name__ == "__main__":
    simple_swizzle_layout()
    warpgroup_swizzle_layout()
