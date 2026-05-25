import cutlass
import cutlass.cute as cute
import numpy as np
import torch

from cutlass.cute.runtime import from_dlpack


def print_values_example():
    print("print_values_example()")

    @cute.jit
    def print_values(a: cutlass.Int32, b: cutlass.Constexpr[int]):
        # Use Python `print` to print static information
        print(f"b = {b}")  # => 2
        # `a` is dynamic value
        print(f"a = {a}")  # => ?

        # Use `cute.printf` to print dynamic information
        cute.printf("a = {}", a)  # => 8
        cute.printf("b = {}", b)  # => 2

        print(f"type(a) = {type(a)}")  # => <class 'cutlass.Int32'>
        print(f"type(b) = {type(b)}")  # => <class 'int'>

        layout = cute.make_layout((a, b))
        print(f"layout = {layout}")  # => (?,2):(1,?)
        cute.printf("layout = {}", layout)  # => (8,2):(1,8)

    print_values(cutlass.Int32(8), 2)

    compiled_print = cute.compile(print_values, cutlass.Int32(8), 2)

    compiled_print(cutlass.Int32(8))


def print_tensor_example():
    print("print_tensor_example()")

    @cute.jit
    def print_tensor(x: cute.Tensor):
        cute.print_tensor(x)

    shape = (4, 3, 2)
    data = np.arange(24, dtype=np.float32).reshape(*shape)

    print_tensor(from_dlpack(data))


def print_tensor_verbose_example():
    print("print_tensor_verbose_example()")

    @cute.jit
    def print_tensor_verbose(x: cute.Tensor):
        cute.print_tensor(x, verbose=True)

    shape = (4, 3)
    data = np.arange(12, dtype=np.float32).reshape(*shape)

    print_tensor_verbose(from_dlpack(data))


def print_device_tensor_example():
    print("print_device_tensor_example()")

    @cute.kernel
    def print_device_kernel(src: cute.Tensor):
        print(src)
        cute.print_tensor(src)

    @cute.jit
    def print_device_tensor(src: cute.Tensor):
        print_device_kernel(src).launch(grid=(1, 1, 1), block=(1, 1, 1))

    a = torch.randn(4, 3, device="cuda")
    cutlass.cuda.initialize_cuda_context()
    print_device_tensor(from_dlpack(a))


if __name__ == "__main__":
    examples = [
        print_values_example,
        print_tensor_example,
        print_tensor_verbose_example,
        print_device_tensor_example,
    ]

    for i, example in enumerate(examples):
        if i:
            print()
        example()
