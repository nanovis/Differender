import taichi as ti
import utils
import numpy as np

ti.init(arch=ti.cuda)
# TODO: check dimension issue, see data placement in np and in ti.field
dims, data = utils.load_head_data()
data_f = data / float(np.iinfo(np.uint16).max)
data_f = data_f.reshape(-1)
width, height, slice_num = dims[0], dims[1], dims[2]
pixels = ti.field(dtype=float, shape=(width * 5, height * 5))
data_field = ti.field(ti.f64, shape=(data_f.shape[0]))
data_field.from_numpy(data_f)


@ti.kernel
def paint_slice(slice_idx: int):
    for i, j in pixels:
        idx_x = i // 5
        idx_y = j // 5
        pixels[i, j] = data_field[idx_x + idx_y * width + slice_idx * (width * height)]


gui = ti.GUI("SciVis Slicing", res=(width * 5, height * 5))
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
