from tensor_layouts import Layout
from tensor_layouts.viz import draw_layout


def visualize_layout(layout: Layout, file_name: str):
    draw_layout(layout, title=str(layout), colorize=True, filename=f"{file_name}.png")
