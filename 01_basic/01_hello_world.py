import cutlass
import cutlass.cute as cute

from cutlass.cute import KeepPTX, KeepCUBIN, GenerateLineInfo


@cute.kernel
def kernel():
    # Get the x component of the thread index (y and z components are unused)
    tidx, _, _ = cute.arch.thread_idx()
    # Only the first thread (thread 0) prints the message
    if cutlass.dynamic_expr(tidx == 0):
        cute.printf("Hello world from device code!")


@cute.jit
def hello_world():
    # Print hello world from host code
    cute.printf("hello world from host code!")

    # Launch kernel
    kernel().launch(
        grid=(4, 1, 1),  # 4 CTAs (thread block)
        block=(256, 1, 1),  # 4 warps per thread block
    )


if __name__ == "__main__":

    # auto JIT
    hello_world()

    # manual JIT with PTX, CUBIN, line info
    print("Compiling with PTX/CUBIN dumped...")
    hello_world_compiled = cute.compile[KeepPTX, KeepCUBIN, GenerateLineInfo](hello_world)

    # Run the pre-compiled version
    print("Running compiled version...")
    hello_world_compiled()
