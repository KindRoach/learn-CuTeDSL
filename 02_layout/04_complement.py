import cutlass.cute as cute

from tensor_layouts import Layout, complement

from .utils import visualize_layout


def complement_layout():
    print("complement_layout()")

    shape, stride = 4, 2
    cotarget = 24

    @cute.jit
    def cute_jit():
        layout = cute.make_layout(shape, stride=stride)
        complement_layout = cute.complement(layout, cotarget)
        cute.printf("complement( {}, {} ) = {}", layout, cotarget, complement_layout)

    cute_jit()

    layout = Layout(shape, stride)
    complement_layout = complement(layout, cotarget)

    visualize_layout(layout, file_name="complement_a")
    visualize_layout(complement_layout, file_name="complement_output")


if __name__ == "__main__":
    complement_layout()
