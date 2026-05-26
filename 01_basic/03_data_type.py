import cutlass
import cutlass.cute as cute


def dynamic_vs_static_value_example():
    print("dynamic_vs_static_value_example()")

    @cute.jit
    def dynamic_vs_static_values(static_parmeter: cutlass.Constexpr[float]):
        # Python-native values and Constexpr parameters are static.
        static_value = 42
        print("static_parmeter =", static_parmeter)
        print("static_value =", static_value)

        # cutlass types are dynamic.
        a = cutlass.Float32(3.14)
        b = cutlass.Int32(5)

        # Python print runs while JIT-compiling, so dynamic values are ?.
        print("a(compile time) =", a)
        print("b(compile time) =", b)

        # cute.printf runs in the generated code and prints runtime values.
        cute.printf("a(run time) = {}", a)
        cute.printf("b(run time) = {}", b)

    dynamic_vs_static_values(0.5)


def type_conversion_example():
    print("type_conversion_example()")

    @cute.jit
    def type_conversion():
        x = cutlass.Int32(42)
        y = x.to(cutlass.Float32)
        cute.printf("Int32({}) => Float32({})", x, y)

        a = cutlass.Float32(3.14)
        b = a.to(cutlass.Int32)
        cute.printf("Float32({}) => Int32({})", a, b)

        c = cutlass.Int32(127)
        d = c.to(cutlass.Int8)
        cute.printf("Int32({}) => Int8({})", c, d)

        e = cutlass.Int32(300)
        f = e.to(cutlass.Int8)
        cute.printf("Int32({}) => Int8({}) (truncated to target range)", e, f)

    type_conversion()


def operator_example():
    print("operator_example()")

    @cute.jit
    def operator_demo():
        a = cutlass.Int32(10)
        b = cutlass.Int32(3)
        x = cutlass.Float32(5.5)

        cute.printf("a: Int32({}), b: Int32({})", a, b)
        cute.printf("x: Float32({})", x)

        cute.printf("a + b = {}", a + b)
        cute.printf("x * 2 = {}", x * 2)
        cute.printf("a + x = {} (Int32 + Float32 promotes to Float32)", a + x)
        cute.printf("a / b = {}", a / b)
        cute.printf("x / 2.0 = {}", x / cutlass.Float32(2.0))

        cute.printf("a > b = {}", a > b)
        cute.printf("a & b = {}", a & b)
        cute.printf("-a = {}", -a)
        cute.printf("~a = {}", ~a)

    operator_demo()


if __name__ == "__main__":
    examples = [
        dynamic_vs_static_value_example,
        type_conversion_example,
        operator_example,
    ]

    for i, example in enumerate(examples):
        if i:
            print()
        example()
