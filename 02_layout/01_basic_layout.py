import cutlass.cute as cute
import torch

from cutlass.cute.runtime import from_dlpack
from tensor_layouts import Layout

from .utils import visualize_layout


def pytorch_tensor_layout():
    print("pytorch_tensor_layout()")
    shape = (4, 8)
    tensor = from_dlpack(torch.empty(shape))

    @cute.jit
    def cute_jit():
        cute.printf("Tensor layout in CuTe: {}", tensor.layout)

    cute_jit()
    visualize_layout(Layout(shape, (shape[1], 1)), file_name="pytorch_tensor_layout")


def cute_layout():
    print("cute_layout()")

    shape = (4, (3, 2))
    stride = (6, (2, 1))

    @cute.jit
    def cute_jit():
        layout = cute.make_layout(shape, stride=stride)
        cute.printf("Nested layout in CuTe: {}", layout)

    cute_jit()
    visualize_layout(Layout(shape, stride), file_name="cute_layout")


if __name__ == "__main__":
    pytorch_tensor_layout()
    cute_layout()
