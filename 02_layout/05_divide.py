import cutlass.cute as cute

from tensor_layouts import Layout, logical_divide

from .utils import visualize_layout


def divide_layout():
    print("divide_layout()")

    shape, stride = (4, 2, 3), (2, 1, 8)
    tiler_shape, tiler_stride = 4, 2

    @cute.jit
    def cute_jit():
        layout = cute.make_layout(shape, stride=stride)
        tiler = cute.make_layout(tiler_shape, stride=tiler_stride)
        
        # Other divide variants only change how tile/rest coords are packed:
        # cute.zipped_divide(layout, tiler=tiler)
        # cute.tiled_divide(layout, tiler=tiler)
        # cute.flat_divide(layout, tiler=tiler)
        divided_layout = cute.logical_divide(layout, tiler=tiler)

        cute.printf("layout: {}", layout)
        cute.printf("tiler:  {}", tiler)
        cute.printf("divide( {}, {} ) = {}", layout, tiler, divided_layout)


    cute_jit()

    layout = Layout(shape, stride)
    tiler = Layout(tiler_shape, tiler_stride)
    divided_layout = logical_divide(layout, tiler)

    visualize_layout(layout, file_name="divide_layout")
    visualize_layout(tiler, file_name="divide_tiler")
    visualize_layout(divided_layout, file_name="divide_output")


if __name__ == "__main__":
    divide_layout()
