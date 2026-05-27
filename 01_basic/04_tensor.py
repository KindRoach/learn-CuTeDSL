import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

import torch


@cute.jit
def print_tensor_info(tensor: cute.Tensor):
    cute.print_tensor(tensor)


def create_tensor_from_dlpack():
    print("create_tensor_from_dlpack()")
    x = torch.arange(0, 16, dtype=torch.float16).reshape(2, 8)

    # explict convert
    tensor = from_dlpack(x)
    print_tensor_info(tensor)

    # implicit convert (not recommended), need to remove type annotation of `print_tensor_info`
    # print_tensor_info(x)


@cute.jit
def print_tensor_info_ptr(ptr: cute.Pointer):
    layout = cute.make_layout((8, 2), stride=(2, 1))
    tensor = cute.make_tensor(ptr, layout)
    cute.print_tensor(tensor)


def create_tensor_from_ptr():
    print("create_tensor_from_ptr()")
    x = torch.arange(0, 16, dtype=torch.float16).reshape(2, 8)
    tensor = from_dlpack(x)
    print_tensor_info(tensor)


@cute.jit
def access_tensor(tensor: cute.Tensor):
    # access by logical index
    i, j = 1, 2
    value = tensor[i, j]
    cute.printf("tensor[{}, {}] = {}", i, j, value)

    # access by linear index
    idx = 10
    value = tensor[idx]
    cute.printf("tensor[{}] = {}", idx, value)

    # linear index to logical coordinate
    coord = cute.make_identity_tensor(tensor.shape)
    idx = 10
    logical_coord = coord[idx]
    cute.printf("coord[{}] = {}", idx, logical_coord)
    cute.printf("tensor[{}] = {}", idx, tensor[idx])
    cute.printf("tensor[coord[{}]] = {}", idx, tensor[logical_coord])

    # access by slice
    row1 = tensor[1, None]
    col2 = tensor[None, 2]
    cute.print_tensor(row1)
    cute.print_tensor(col2)


def access_tensor_example():
    print("access_tensor_example()")
    x = torch.arange(0, 16, dtype=torch.float16).reshape(2, 8)
    tensor = from_dlpack(x)
    access_tensor(tensor)


@cute.jit
def add_one(tensor: cute.Tensor):
    # user TensorSSA for element-wise operation
    tv = tensor.load()
    tv = tv + cutlass.Float16(1.0)
    tensor.store(tv)

    cute.print_tensor(tensor)


def add_one_example():
    print("add_one_example()")
    x = torch.arange(0, 16, dtype=torch.float16).reshape(2, 8)
    tensor = from_dlpack(x)
    add_one(tensor)


if __name__ == "__main__":
    create_tensor_from_dlpack()
    create_tensor_from_ptr()
    access_tensor_example()
    add_one_example()
