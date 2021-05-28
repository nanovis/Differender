import taichi as ti
import utils
import numpy as np
import taichi_glsl as tl

ti.init(arch=ti.cuda)
data = utils.load_head_data()
data_f = data / float(np.iinfo(np.uint16).max)
print(data_f.shape)
transfer_function_data = utils.load_transfer_function()
width, height, slice_num = data.shape
scaling = 5
data_field = ti.field(ti.f64, shape=(width, height, slice_num))
# try looking from different axis to check memory layout issues
width, height, slice_num = width, slice_num, height
pixels = ti.Vector.field(3, dtype=float, shape=(width * scaling, height * scaling))
transfer_function = ti.Vector.field(4, dtype=float, shape=transfer_function_data.shape[0])
# materialization
data_field.from_numpy(data_f)
transfer_function.from_numpy(transfer_function_data)
# mesc
slice_num = int(slice_num)
# DVR parameters
step_size = 0.425  # TODO: tune the step size
opacity_threshold = 0.95
ambient = 0.5
diffuse = 0.5
specular = 0.5
shininess = 32.0
delta = step_size / 4.0


@ti.func
def low_high_frac(x: float):
    low = int(x)
    high = low + 1
    frac = x - low
    return low, high, frac


@ti.func
def sample_volume(x: int, y: int, z: float):
    z_low, z_high, fraction = low_high_frac(z)
    scalar_low = data_field[x, y, z_low]
    scalar_high = data_field[x, y, z_high]
    return tl.mix(scalar_low, scalar_high, fraction)


@ti.func
def sample_volume_trilinear(x: float, y: float, z: float):
    x_low, x_high, x_frac = low_high_frac(x)
    y_low, y_high, y_frac = low_high_frac(y)
    z_low, z_high, z_frac = low_high_frac(z)
    # on z_low
    v000 = data_field[x_low, y_low, z_low]
    v100 = data_field[x_high, y_low, z_low]
    x_val_y_low = tl.mix(v000, v100, x_frac)
    v010 = data_field[x_low, y_high, z_low]
    v110 = data_field[x_high, y_high, z_low]
    x_val_y_high = tl.mix(v010, v110, x_frac)
    xy_val_z_low = tl.mix(x_val_y_low, x_val_y_high, y_frac)
    # on z_high
    v001 = data_field[x_low, y_low, z_high]
    v101 = data_field[x_high, y_low, z_high]
    x_val_y_low = tl.mix(v001, v101, x_frac)
    v011 = data_field[x_low, y_high, z_high]
    v111 = data_field[x_high, y_high, z_high]
    x_val_y_high = tl.mix(v011, v111, x_frac)
    xy_val_z_high = tl.mix(x_val_y_low, x_val_y_high, y_frac)
    return tl.mix(xy_val_z_low, xy_val_z_high, z_frac)


@ti.func
def sample_transfer_function(scalar: float):
    length = transfer_function.shape[0]
    val = length * scalar
    low, high, frac = low_high_frac(val)
    val_low = transfer_function[low]
    val_high = transfer_function[high]
    return tl.mix(val_low, val_high, frac)


@ti.func
def get_normal(x: int, y: int, z: float):
    x = float(x)
    y = float(y)
    dx = sample_volume_trilinear(x + delta, y, z) - sample_volume_trilinear(x - delta, y, z)
    dy = sample_volume_trilinear(x, y + delta, z) - sample_volume_trilinear(x, y - delta, z)
    dz = sample_volume_trilinear(x, y, z + delta) - sample_volume_trilinear(x, y, z - delta)
    n = ti.Vector([dx, dy, dz])
    return n / n.norm()


@ti.kernel
def simple_DVR():
    for x, y in pixels:
        idx_x = x // scaling
        idx_y = y // scaling
        ray_direction = tl.vec3(0.0, 0.0, 1.0)
        I_ambient = tl.vec3(ambient)
        I_diffuse = tl.vec3(diffuse)
        I_specular = tl.vec3(specular)
        composite_color = tl.vec4(0.0)
        max_marching_steps = int(slice_num / step_size)
        position = tl.vec3(idx_x, idx_y, 0.0)
        for step in range(max_marching_steps):
            marching_z_pos = position.z
            scalar = sample_volume(idx_x, idx_y, marching_z_pos)
            src_color = sample_transfer_function(scalar)
            opacity = src_color.w
            new_src = tl.vec4(src_color.xyz * opacity, opacity)
            # shading
            normal = get_normal(idx_x, idx_y, marching_z_pos)
            dir_dot_norm = ray_direction.dot(normal)

            diffuse_color = max(dir_dot_norm, 0.0) * I_diffuse
            v = (-position) / position.norm()
            r = tl.reflect(-ray_direction, normal)
            r_dot_v = max(r.dot(v), 0.0)
            pf = pow(r_dot_v, shininess)
            specular_color = I_specular * pf

            shading_color = tl.vec4(I_ambient + diffuse_color + specular_color, 1.0) * new_src
            # compositing
            composite_color = (1.0 - composite_color.w) * shading_color + composite_color
            if composite_color.w > opacity_threshold:
                break
            position += ray_direction * step_size

        pixels[x, y] = composite_color.xyz


# TODO: super slow, need to improve speed
gui = ti.GUI("SciVis Slicing", res=(width * scaling, height * scaling), fast_gui=True)
while gui.running:
    simple_DVR()
    gui.set_image(pixels)
    gui.show()
