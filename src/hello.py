import taichi as ti
import utils
import numpy as np

ti.init(arch=ti.cuda)
data = utils.load_head_data()
data_f = data / float(np.iinfo(np.uint16).max)
width, height, slice_num = data.shape
scaling = 5
pixels = ti.field(dtype=float, shape=(width * scaling, height * scaling))
data_field = ti.field(ti.f64, shape=(width, height, slice_num))
data_field.from_numpy(data_f)


@ti.kernel
def paint_slice(slice_idx: int):
    for i, j in pixels:
        idx_x = i // scaling
        idx_y = j // scaling
        pixels[i, j] = data_field[idx_x, idx_y, slice_idx]


gui = ti.GUI("SciVis Slicing", res=(width * scaling, height * scaling))
slice_num = int(slice_num)
slice_idx = 3
count = 0
while gui.running:
    paint_slice(slice_idx)
    gui.set_image(pixels)
    gui.show()
    count += 1
    if count % 60 == 0:
        count = 0
        slice_idx += 1
        slice_idx = slice_idx % slice_num
