import cutlass.cute as cute

from tensor_layouts import Layout, compose

from .utils import visualize_layout


def composition_layout():
    print("composition_layout()")

    a_shape, a_stride = (6, 2), (8, 2)
    b_shape, b_stride = (4, 3), (3, 1)

    @cute.jit
    def cute_jit():
        a = cute.make_layout(a_shape, stride=a_stride)
        b = cute.make_layout(b_shape, stride=b_stride)
        composed_layout = cute.composition(a, b)
        cute.printf("composition( {}, {} ) = {}", a, b, composed_layout)

    cute_jit()

    a = Layout(a_shape, a_stride)
    b = Layout(b_shape, b_stride)
    composed_layout = compose(a, b)

    visualize_layout(a, file_name="composition_a")
    visualize_layout(b, file_name="composition_b")
    visualize_layout(composed_layout, file_name="composition_output")


if __name__ == "__main__":
    composition_layout()
