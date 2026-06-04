import cutlass.cute as cute

from tensor_layouts import Layout, logical_product

from .utils import visualize_layout


def product_layout():
    print("product_layout()")

    shape, stride = (2, 2), (4, 1)
    tiler_shape, tiler_stride = 6, 1

    @cute.jit
    def cute_jit():
        layout = cute.make_layout(shape, stride=stride)
        tiler = cute.make_layout(tiler_shape, stride=tiler_stride)

        # Other product variants change coord packing or traversal order:
        # cute.blocked_product(layout, tiler=tiler)
        # cute.raked_product(layout, tiler=tiler)
        # cute.zipped_product(layout, tiler=tiler)
        # cute.tiled_product(layout, tiler=tiler)
        # cute.flat_product(layout, tiler=tiler)
        product_layout = cute.logical_product(layout, tiler=tiler)

        cute.printf("layout: {}", layout)
        cute.printf("tiler:  {}", tiler)
        cute.printf("product( {}, {} ) = {}", layout, tiler, product_layout)

    cute_jit()

    layout = Layout(shape, stride)
    tiler = Layout(tiler_shape, tiler_stride)
    product_layout = logical_product(layout, tiler)

    visualize_layout(layout, file_name="product_layout")
    visualize_layout(tiler, file_name="product_tiler")
    visualize_layout(product_layout, file_name="product_output")


if __name__ == "__main__":
    product_layout()
