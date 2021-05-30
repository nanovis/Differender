import taichi as ti
import utils
import numpy as np
import taichi_glsl as tl
import math

ti.init(arch=ti.cuda)
data = utils.load_head_data()
data_f = (data / float(np.iinfo(np.uint16).max)).astype(np.float32)
print(data_f.shape)
transfer_function_data = utils.load_transfer_function()
width, height, slice_num = data.shape
scaling = 5
data_field = ti.field(ti.f32, shape=(width, height, slice_num))
# try looking from different axis to check memory layout issues
width, height, slice_num = width, slice_num, height
pixels = ti.Vector.field(3, dtype=ti.f32, shape=(width * scaling, height * scaling))
transfer_function = ti.Vector.field(4, dtype=ti.f32, shape=transfer_function_data.shape[0])
# materialization
data_field.from_numpy(data_f)
transfer_function.from_numpy(transfer_function_data)
# mesc
slice_num = int(slice_num)
# DVR parameters
step_size = 0.0025
opacity_threshold = 0.95
ambient = 0.5
diffuse = 0.5
specular = 0.5
shininess = 32.0
delta = step_size / 4.0
cam_up = tl.vec3(0.0, 1.0, 0.0)
cam_center = tl.vec3(0.0)


@ti.func
def low_high_frac(x: float):
    low = int(x)
    high = low + 1
    frac = x - low
    return low, high, frac


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
    length = ti.static(transfer_function.shape[0])
    val = length * scalar
    low, high, frac = low_high_frac(val)
    val_low = transfer_function[low]
    val_high = transfer_function[high]
    return tl.mix(val_low, val_high, frac)


@ti.func
def get_normal(x: float, y: float, z: float):
    dx = sample_volume_trilinear(x + delta, y, z) - sample_volume_trilinear(x - delta, y, z)
    dy = sample_volume_trilinear(x, y + delta, z) - sample_volume_trilinear(x, y - delta, z)
    dz = sample_volume_trilinear(x, y, z + delta) - sample_volume_trilinear(x, y, z - delta)
    n = ti.Vector([dx, dy, dz])
    return n.normalized()


@ti.func
def on_box(pos):
    eps = ti.static(0.00001)
    x = ti.static(pos.x)
    y = ti.static(pos.y)
    z = ti.static(pos.z)
    return (((1.0 - eps <= x <= 1.0 + eps) or (-1.0 - eps <= x <= -1.0 + eps)) and -1 <= y <= 1 and -1 <= z <= 1) \
           or (((1.0 - eps <= y <= 1.0 + eps) or (-1.0 - eps <= y <= -1.0 + eps)) and -1 <= x <= 1 and -1 <= z <= 1) \
           or (((1.0 - eps <= z <= 1.0 + eps) or (-1.0 - eps <= z <= -1.0 + eps)) and -1 <= y <= 1 and -1 <= x <= 1)


@ti.func
def calc_in_out(view_pos, view_dir):
    a = ti.static(view_dir.x)
    b = ti.static(view_dir.y)
    c = ti.static(view_dir.z)
    x = ti.static(view_pos.x)
    y = ti.static(view_pos.y)
    z = ti.static(view_pos.z)
    t_x_pos1 = (1 - x) / a
    t_x_min1 = (-1 - x) / a
    t_y_pos1 = (1 - y) / b
    t_y_min1 = (-1 - y) / b
    t_z_pos1 = (1 - z) / c
    t_z_min1 = (-1 - z) / c
    vec = [t_x_pos1, t_x_min1, t_y_pos1, t_y_min1, t_z_pos1, t_z_min1]
    ts = tl.vec2(0.0)
    found_one = False
    for i in ti.static(range(6)):
        t = vec[i]
        if not tl.isnan(t) and t >= 0 and on_box(view_pos + t * view_dir):
            if found_one:
                ts[1] = t
            else:
                ts[0] = t
                found_one = True

    t_in = ts.min()
    t_out = ts.max()
    in_pos = view_pos + t_in * view_dir
    out_pos = view_pos + t_out * view_dir
    return in_pos, out_pos, found_one


@ti.kernel
def render_box(cam_pos_x: float, cam_pos_y: float, cam_pos_z: float, view_u: float, view_v: float):
    psfx = ti.static(float(pixels.shape[0]))
    psfy = ti.static(float(pixels.shape[1]))
    cam_position = tl.vec3(cam_pos_x, cam_pos_y, cam_pos_z)
    looking_direction = (cam_center - cam_position).normalized()
    right_dir = tl.cross(looking_direction, cam_up).normalized()
    up_dir = tl.cross(right_dir, looking_direction).normalized()
    right_axis_len_vec = right_dir * view_u
    up_axis_len_vec = up_dir * view_v

    for x, y in pixels:
        pixels[x, y].fill(0.0)
        view_pos = cam_position - 0.5 * up_axis_len_vec - 0.5 * right_axis_len_vec
        view_pos += float(x) / psfx * right_axis_len_vec
        view_pos += float(y) / psfy * up_axis_len_vec
        in_pos, out_pos, intersected = calc_in_out(view_pos, looking_direction)
        if intersected:
            in_pos = in_pos * 0.5 + 0.5
            out_pos = out_pos * 0.5 + 0.5
            pixels[x, y] = in_pos - out_pos


@ti.kernel
def direct_volume_rendering(cam_pos_x: float, cam_pos_y: float, cam_pos_z: float, view_u: float, view_v: float):
    psfx = ti.static(float(pixels.shape[0]))
    psfy = ti.static(float(pixels.shape[1]))
    I_ambient = ti.static(tl.vec3(ambient))
    I_diffuse = ti.static(tl.vec3(diffuse))
    I_specular = ti.static(tl.vec3(specular))
    shape = ti.static(tl.vec3(data_field.shape[0], data_field.shape[1], data_field.shape[2]))

    cam_position = tl.vec3(cam_pos_x, cam_pos_y, cam_pos_z)
    looking_direction = (cam_center - cam_position).normalized()
    right_dir = tl.cross(looking_direction, cam_up).normalized()
    up_dir = tl.cross(right_dir, looking_direction).normalized()
    right_axis_len_vec = right_dir * view_u
    up_axis_len_vec = up_dir * view_v
    for x, y in pixels:
        # orthogonal projection
        view_pos = cam_position - 0.5 * up_axis_len_vec - 0.5 * right_axis_len_vec
        view_pos += float(x) / psfx * right_axis_len_vec
        view_pos += float(y) / psfy * up_axis_len_vec
        # test ray intersection
        in_pos, out_pos, intersected = calc_in_out(view_pos, looking_direction)
        if intersected:
            tex_in_pos = in_pos * 0.5 + 0.5
            tex_out_pos = out_pos * 0.5 + 0.5
            max_marching_steps = int((tex_in_pos - tex_out_pos).norm() / step_size)
            ray_direction = (tex_in_pos - tex_out_pos).normalized()
            composite_color = tl.vec4(0.0)
            tex_position = tex_in_pos
            for step in range(max_marching_steps):
                unnormalized_tex_pos = tex_position * shape
                scalar = sample_volume_trilinear(unnormalized_tex_pos.x, unnormalized_tex_pos.y,
                                                 unnormalized_tex_pos.z)
                src_color = sample_transfer_function(scalar)
                opacity = src_color.w
                new_src = tl.vec4(src_color.xyz * opacity, opacity)
                # shading
                normal = get_normal(unnormalized_tex_pos.x, unnormalized_tex_pos.y, unnormalized_tex_pos.z)
                dir_dot_norm = ray_direction.dot(normal)
                diffuse_color = max(dir_dot_norm, 0.0) * I_diffuse
                v = -tex_position.normalized()
                r = tl.reflect(-ray_direction, normal)
                r_dot_v = max(r.dot(v), 0.0)
                pf = pow(r_dot_v, shininess)
                specular_color = I_specular * pf
                shading_color = tl.vec4(I_ambient + diffuse_color + specular_color, 1.0) * new_src
                # compositing
                composite_color = (1.0 - composite_color.w) * shading_color + composite_color
                if composite_color.w > opacity_threshold:
                    break
                tex_position += ray_direction * step_size
            pixels[x, y] = composite_color.xyz
        else:
            pixels[x, y] = tl.vec3(0.0)


def keyboard_input(gui, camera_angles, increment=1.0):
    gui.get_event()
    if gui.is_pressed('w'):
        camera_angles.y += increment
    elif gui.is_pressed('s'):
        camera_angles.y -= increment
    elif gui.is_pressed('a'):
        camera_angles.x = min(179.0, camera_angles.x + increment)
    elif gui.is_pressed('d'):
        camera_angles.x = max(1.0, camera_angles.x - increment)


# TODO: super slow, need to improve speed
# FIXME: render incorrect, seems to be data loading problem related to coordinates
gui = ti.GUI("SciVis Slicing", res=(width * scaling, height * scaling), fast_gui=True)
camera_angles = tl.vec2(1.0, 0.0)
radius = 3.0
camera_pos = tl.vec3(0.0, 0.0, radius)
while gui.running:
    # direct_volume_rendering(0.0, 0.0, 0.0, 0.0, 0.0)
    keyboard_input(gui, camera_angles)
    camera_pos.x = radius * math.cos(math.radians(camera_angles.y)) * math.sin(math.radians(camera_angles.x))
    camera_pos.y = radius * math.sin(math.radians(camera_angles.y)) * math.sin(math.radians(camera_angles.x))
    camera_pos.z = radius * math.cos(math.radians(camera_angles.x))
    print(camera_angles)
    direct_volume_rendering(camera_pos.x, camera_pos.y, camera_pos.z, 4.0, 4.0)
    gui.set_image(pixels)
    gui.show()
