import cutlass
import cutlass.cute as cute
import torch

from tensor_layouts import Layout
from tensor_layouts.viz import draw_layout


def visualize_layout(x, file_name: str):
    if isinstance(x, cute.Tensor):
        layout = Layout(x.shape, x.stride)
    else:
        layout = x

    draw_layout(layout, title=str(layout), colorize=True, filename=f"{file_name}.svg")


def pytorch_tensor_layout():
    print("pytorch_tensor_layout()")

    x = torch.arange(0, 16, dtype=torch.float16)
    visualize_layout(cute.runtime.from_dlpack(x), file_name="torch_16_tensor")

    x_2x8 = x.reshape(2, 8)
    visualize_layout(cute.runtime.from_dlpack(x_2x8), file_name="torch_2x8_tensor")

    x_2x2x4 = x.reshape(2, 2, 4)
    visualize_layout(cute.runtime.from_dlpack(x_2x2x4), file_name="torch_2x2x4_tensor")


def nest_layout():
    print("nest_layout()")

    layout = Layout((4, (3, 2)))
    visualize_layout(layout, file_name=str(layout))

    layout = Layout(((2, 3), (4, 5)))
    visualize_layout(layout, file_name=str(layout))


def boardcast_layout():
    print("boardcast_layout()")

    layout = Layout((4, 8), (1, 0))
    visualize_layout(layout, file_name=str(layout))


@cute.jit
def print_layout(x: cutlass.Int64, y: cutlass.Int64):
    layout = cute.make_layout((x, y))
    print(f"Compile Time Layout: {layout}")
    cute.printf("Runtime Layout: {}", layout)

    layout = cute.make_layout((2, 3))
    print(f"Compile Time Layout: {layout}")
    cute.printf("Runtime Layout: {}", layout)


def static_vs_dynamic_layout():
    print("static_vs_dynamic_layout()")
    print_layout(4, 8)


if __name__ == "__main__":
    pytorch_tensor_layout()
    nest_layout()
    boardcast_layout()
    static_vs_dynamic_layout()
