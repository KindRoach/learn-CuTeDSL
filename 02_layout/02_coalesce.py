import cutlass
import cutlass.cute as cute

import tensor_layouts
from tensor_layouts.viz import draw_layout


def visualize_layout(layout, file_name: str):
    draw_layout(layout, title=str(layout), colorize=True, filename=f"{file_name}.svg")


@cute.jit
def coalesce_in_cute_jit(
    shape: cutlass.Constexpr,
    stride: cutlass.Constexpr,
    target_profile: cutlass.Constexpr,
):
    layout = cute.make_layout(shape, stride=stride)
    cute.printf("Runtime Layout: {}", layout)

    coalesced_layout = cute.coalesce(layout)
    cute.printf("Coalesced Layout: {}", coalesced_layout)

    by_mode_coalesced_layout = cute.coalesce(layout, target_profile=target_profile)
    cute.printf("By-mode Coalesced Layout: {}", by_mode_coalesced_layout)


def coalesce_layout():
    print("coalesce_layout()")
    shape = (2, (1, 6))
    stride = (1, (6, 2))
    target_profile = (2, 6)
    coalesce_in_cute_jit(shape, stride, target_profile)

    layout = tensor_layouts.Layout(shape, stride)
    visualize_layout(layout, file_name=str(layout))

    coalesced_layout = tensor_layouts.coalesce(layout)
    visualize_layout(coalesced_layout, file_name=str(layout) + "_coalesced")

    by_mode_coalesced_layout = tensor_layouts.coalesce(layout, target_profile)
    visualize_layout(
        by_mode_coalesced_layout, file_name=str(layout) + "_by_mode_coalesced"
    )


if __name__ == "__main__":
    coalesce_layout()
