import cutlass.cute as cute

from tensor_layouts import Layout, coalesce

from .utils import visualize_layout


def coalesce_layout():
    print("coalesce_layout()")

    shape = (2, 3)
    stride = (1, 2)

    @cute.jit
    def cute_jit():
        layout = cute.make_layout(shape, stride=stride)
        coalesced_layout = cute.coalesce(layout)
        cute.printf("coalesce( {} ) = {}", layout, coalesced_layout)

    cute_jit()

    layout = Layout(shape, stride)
    coalesced_layout = coalesce(layout)
    visualize_layout(layout, file_name="coalesce_layout_before")
    visualize_layout(coalesced_layout, file_name="coalesce_layout_after")


def coalesce_layout_by_mode():
    print("coalesce_layout_by_mode()")

    shape = (2, (1, 6))
    stride = (1, (6, 2))
    target_profile = (2, 6)

    @cute.jit
    def cute_jit():
        layout = cute.make_layout(shape, stride=stride)
        coalesced_layout = cute.coalesce(layout, target_profile=target_profile)
        cute.printf("coalesce( {}, target_profile={} ) = {}", layout, target_profile, coalesced_layout)

    cute_jit()

    layout = Layout(shape, stride)
    coalesced_layout = coalesce(layout, target_profile)
    visualize_layout(layout, file_name="coalesce_layout_by_mode_before")
    visualize_layout(coalesced_layout, file_name="coalesce_layout_by_mode_after")


if __name__ == "__main__":
    coalesce_layout()
    coalesce_layout_by_mode()
