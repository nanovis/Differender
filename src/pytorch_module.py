import torch
import taichi as ti
import taichi_glsl as tl
import numpy as np


@ti.func
def low_high_frac(x: float):
    ''' Returns the integer value below and above, as well as the frac

    Args:
        x (float): Floating point number

    Returns:
        int, int, float: floor, ceil, frac of `x`
    '''
    x = ti.max(x, 0.0)
    low = ti.floor(x)
    high = low + 1
    frac = x - float(low)
    return int(low), int(high), frac

@ti.func
def premultiply_alpha(rgba):
    rgba.xyz *= rgba.w
    return rgba

@ti.func
def get_entry_exit_points(look_from, view_dir, bl, tr):
    ''' Computes the entry and exit points of a given ray

    Args:
        look_from (tl.vec3): Camera Position as vec3
        view_dir (tl.vec): View direction as vec3, normalized
        bl (tl.vec3): Bottom left of the bounding box
        tr (tl.vec3): Top right of the bounding box

    Returns:
        float, float, bool: Distance to entry, distance to exit, bool whether box is hit
    '''
    dirfrac = 1.0 / view_dir
    t1 = (bl.x - look_from.x) * dirfrac.x
    t2 = (tr.x - look_from.x) * dirfrac.x
    t3 = (bl.y - look_from.y) * dirfrac.y
    t4 = (tr.y - look_from.y) * dirfrac.y
    t5 = (bl.z - look_from.z) * dirfrac.z
    t6 = (tr.z - look_from.z) * dirfrac.z

    tmin = max(max(min(t1, t2), min(t3, t4)), min(t5, t6))
    tmax = min(min(max(t1, t2), max(t3, t4)), max(t5, t6))
    hit = True
    if tmax < 0.0 or tmin > tmax: hit = False
    return tmin, tmax, hit


@ti.data_oriented
class VolumeRaycaster():
    def __init__(self,
                 volume_resolution,
                 render_resolution,
                 max_samples=512,
                 tf_resolution=128,
                 fov=30.0,
                 nearfar=(0.1, 100.0)):
        ''' Initializes Volume Raycaster. Make sure to .set_volume() and .set_tf_tex() after initialization

        Args:
            volume_resolution (3-tuple of int): Resolution of the volume data (w,h,d)
            render_resolution (2-tuple of int): Resolution of the rendering (w,h)
            tf_resolution (int): Resolution of the transfer function texture
            fov (float, optional): Field of view of the camera in degrees. Defaults to 60.0.
            nearfar (2-tuple of float, optional): Near and far plane distance used for perspective projection. Defaults to (0.1, 100.0).
        '''
        self.resolution = render_resolution
        self.aspect = render_resolution[0] / render_resolution[1]
        self.fov_deg = fov
        self.fov_rad = np.radians(fov)
        self.near, self.far = nearfar
        # Taichi Fields
        self.volume = ti.field(ti.f32, needs_grad=True)
        self.tf_tex = ti.Vector.field(4, dtype=ti.f32, needs_grad=True)
        self.render_tape = ti.Vector.field(4, dtype=ti.f32, needs_grad=True)
        self.output_rgba = ti.Vector.field(4, dtype=ti.f32, needs_grad=True)
        self.valid_sample_step_count = ti.field(ti.i32)
        self.sample_step_nums = ti.field(ti.i32)
        self.entry = ti.field(ti.f32)
        self.exit = ti.field(ti.f32)
        self.rays = ti.Vector.field(3, dtype=ti.f32)
        self.max_valid_sample_step_count = ti.field(ti.i32, ())
        self.max_samples = max_samples
        self.ambient = 0.4
        self.diffuse = 0.8
        self.specular = 0.3
        self.shininess = 32.0
        self.light_color = tl.vec3(1.0)
        self.cam_pos = ti.Vector.field(3, dtype=ti.f32)
        volume_resolution = tuple(map(lambda d: d // 4, volume_resolution))
        render_resolution = tuple(map(lambda d: d // 16, render_resolution))
        ti.root.dense(ti.ijk,
                      volume_resolution).dense(ti.ijk,
                                               (4, 4, 4)).place(self.volume)
        ti.root.dense(ti.ijk, (*render_resolution, max_samples)).dense(
            ti.ijk, (16, 16, 1)).place(self.render_tape)
        ti.root.dense(ti.ij, render_resolution).dense(ti.ij, (16, 16)).place(
            self.valid_sample_step_count, self.sample_step_nums)
        ti.root.dense(ti.ij, render_resolution).dense(ti.ij, (16, 16)).place(
            self.output_rgba)
        ti.root.dense(ti.ij, render_resolution).dense(ti.ij, (16, 16)).place(
            self.entry, self.exit)
        ti.root.dense(ti.ij,
                      render_resolution).dense(ti.ij,
                                               (16, 16)).place(self.rays)
        ti.root.dense(ti.i, tf_resolution).place(self.tf_tex)
        ti.root.dense(ti.i, tf_resolution).place(self.tf_tex.grad)
        ti.root.place(self.cam_pos)
        ti.root.lazy_grad()

    def set_volume(self, volume):
        self.volume.from_torch(volume.float())

    def set_tf_tex(self, tf_tex):
        self.tf_tex.from_torch(tf_tex.float())

    @ti.func
    def get_ray_direction(self, orig, view_dir, x: float, y: float):
        ''' Compute ray direction for perspecive camera.

        Args:
            orig (tl.vec3): Camera position
            view_dir (tl.vec3): View direction, normalized
            x (float): Image coordinate in [0,1] along width
            y (float): Image coordinate in [0,1] along height

        Returns:
            tl.vec3: Ray direction from camera origin to pixel specified through `x` and `y`
        '''
        u = x - 0.5
        v = y - 0.5

        up = tl.vec3(0.0, 1.0, 0.0)
        right = tl.cross(view_dir, up).normalized()
        up = tl.cross(right, view_dir).normalized()
        near_h = 2.0 * ti.tan(self.fov_rad) * self.near
        near_w = near_h * self.aspect
        near_m = orig + self.near * view_dir
        near_pos = near_m + u * near_w * right + v * near_h * up

        return (near_pos - orig).normalized()

    @ti.func
    def sample_volume_trilinear(self, pos):
        ''' Samples volume data at `pos` and trilinearly interpolates the value

        Args:
            pos (tl.vec3): Position to sample the volume in [-1, 1]^3

        Returns:
            float: Sampled interpolated intensity
        '''
        pos = tl.clamp(
            ((0.5 * pos) + 0.5), 0.0,
            1.0) * ti.static(tl.vec3(*self.volume.shape) - 1.0 - 1e-4)
        x_low, x_high, x_frac = low_high_frac(pos.x)
        y_low, y_high, y_frac = low_high_frac(pos.y)
        z_low, z_high, z_frac = low_high_frac(pos.z)

        x_high = min(x_high, ti.static(self.volume.shape[0] - 1))
        y_high = min(y_high, ti.static(self.volume.shape[1] - 1))
        z_high = min(z_high, ti.static(self.volume.shape[2] - 1))
        # on z_low
        v000 = self.volume[x_low, y_low, z_low]
        v100 = self.volume[x_high, y_low, z_low]
        x_val_y_low = tl.mix(v000, v100, x_frac)
        v010 = self.volume[x_low, y_high, z_low]
        v110 = self.volume[x_high, y_high, z_low]
        x_val_y_high = tl.mix(v010, v110, x_frac)
        xy_val_z_low = tl.mix(x_val_y_low, x_val_y_high, y_frac)
        # on z_high
        v001 = self.volume[x_low, y_low, z_high]
        v101 = self.volume[x_high, y_low, z_high]
        x_val_y_low = tl.mix(v001, v101, x_frac)
        v011 = self.volume[x_low, y_high, z_high]
        v111 = self.volume[x_high, y_high, z_high]
        x_val_y_high = tl.mix(v011, v111, x_frac)
        xy_val_z_high = tl.mix(x_val_y_low, x_val_y_high, y_frac)
        return tl.mix(xy_val_z_low, xy_val_z_high, z_frac)

    @ti.func
    def get_volume_normal(self, pos):
        delta = 1e-3
        x_delta = tl.vec3(delta, 0.0, 0.0)
        y_delta = tl.vec3(0.0, delta, 0.0)
        z_delta = tl.vec3(0.0, 0.0, delta)
        dx = self.sample_volume_trilinear(
            pos + x_delta) - self.sample_volume_trilinear(pos - x_delta)
        dy = self.sample_volume_trilinear(
            pos + y_delta) - self.sample_volume_trilinear(pos - y_delta)
        dz = self.sample_volume_trilinear(
            pos + z_delta) - self.sample_volume_trilinear(pos - z_delta)
        return tl.vec3(dx, dy, dz).normalized()

    @ti.func
    def apply_transfer_function(self, intensity: float):
        ''' Applies a 1D transfer function to a given intensity value

        Args:
            intensity (float): Intensity in [0,1]

        Returns:
            tl.vec4: Color and opacity for given `intensity`
        '''
        length = ti.static(float(self.tf_tex.shape[0] - 1))
        low, high, frac = low_high_frac(intensity * length)
        return tl.mix(
            self.tf_tex[low],
            self.tf_tex[min(high, ti.static(self.tf_tex.shape[0] - 1))], frac)

    @ti.kernel
    def compute_entry_exit(self, sampling_rate: float, jitter: int):
        ''' Produce entry, exit, rays, mask buffers

        Args:
            sampling_rate (float): Sampling rate (multiplier to Nyquist criterium)
            jitter (int): Bool whether to apply jitter or not
        '''
        for i, j in self.entry:  # For all pixels
            max_x = ti.static(float(self.render_tape.shape[0]))
            max_y = ti.static(float(self.render_tape.shape[1]))
            look_from = self.cam_pos[None]
            view_dir = (-look_from).normalized()

            bb_bl = ti.static(tl.vec3(-1.0, -1.0,
                                      -1.0))  # Bounding Box bottom left
            bb_tr = ti.static(tl.vec3(1.0, 1.0,
                                      1.0))  # Bounding Box bottom right
            x = (float(i) + 0.5) / max_x  # Get pixel centers in range (0,1)
            y = (float(j) + 0.5) / max_y  #
            vd = self.get_ray_direction(
                look_from, view_dir, x,
                y)  # Get exact view direction to this pixel
            tmin, tmax, hit = get_entry_exit_points(
                look_from, vd, bb_bl, bb_tr
            )  # distance along vd till volume entry and exit, hit bool

            vol_diag = ti.static(
                (tl.vec3(*self.volume.shape) - tl.vec3(1.0)).norm())
            ray_len = tmax - tmin
            n_samples = hit * (
                ti.floor(sampling_rate * ray_len * vol_diag) + 1
            )  # Number of samples according to https://osf.io/u9qnz
            if jitter:
                tmin += ti.random(dtype=float) * ray_len / n_samples
            self.entry[i, j] = tmin
            self.exit[i, j] = tmax
            self.rays[i, j] = vd
            self.sample_step_nums[i, j] = n_samples

    @ti.kernel
    def raycast(self, sampling_rate: float):
        ''' Produce a rendering. Run compute_entry_exit first! '''
        for i, j in self.valid_sample_step_count:  # For all pixels
            for sample_idx in range(self.sample_step_nums[i, j]):
                look_from = self.cam_pos[None]
                if self.render_tape[i, j, sample_idx -
                                    1].w < 0.99 and sample_idx < ti.static(
                                        self.max_samples):
                    tmax = self.exit[i, j]
                    n_samples = self.sample_step_nums[i, j]
                    ray_len = (tmax - self.entry[i, j])
                    tmin = self.entry[
                        i,
                        j] + 0.5 * ray_len / n_samples  # Offset tmin as t_start
                    vd = self.rays[i, j]
                    pos = look_from + tl.mix(
                        tmin, tmax,
                        float(sample_idx) /
                        float(n_samples - 1)) * vd  # Current Pos
                    light_pos = look_from + tl.vec3(0.0, 1.0, 0.0)
                    intensity = self.sample_volume_trilinear(pos)
                    sample_color = self.apply_transfer_function(intensity)
                    opacity = 1.0 - ti.pow(1.0 - sample_color.w,
                                           1.0 / sampling_rate)
                    # if sample_color.w > 1e-3:
                    normal = self.get_volume_normal(pos)
                    light_dir = (
                        pos -
                        light_pos).normalized()  # Direction to light source
                    n_dot_l = max(normal.dot(light_dir), 0.0)
                    diffuse = self.diffuse * n_dot_l
                    r = tl.reflect(light_dir,
                                   normal)  # Direction of reflected light
                    r_dot_v = max(r.dot(-vd), 0.0)
                    specular = self.specular * pow(r_dot_v, self.shininess)
                    shaded_color = tl.vec4(
                        (diffuse + specular + self.ambient) *
                        sample_color.xyz * opacity * self.light_color, opacity)
                    self.render_tape[i, j, sample_idx] = (
                        1.0 - self.render_tape[i, j, sample_idx - 1].w
                    ) * shaded_color + self.render_tape[i, j, sample_idx - 1]
                    self.valid_sample_step_count[i, j] += 1
                else:
                    self.render_tape[i, j, sample_idx] = self.render_tape[
                        i, j, sample_idx - 1]


    @ti.kernel
    def get_final_image(self):
        for i, j in self.valid_sample_step_count:
            valid_sample_step_count = self.valid_sample_step_count[i, j] - 1
            ns = self.sample_step_nums[i, j]
            self.output_rgba[i, j] += self.render_tape[i, j, ns - 1]
            if valid_sample_step_count > self.max_valid_sample_step_count[None]:
                self.max_valid_sample_step_count[
                    None] = valid_sample_step_count

    @ti.kernel
    def clear_framebuffer(self):
        self.max_valid_sample_step_count[None] = 0
        for i, j, k in self.render_tape:
            self.render_tape[i, j, k] = tl.vec4(0.0)
        for i, j in self.valid_sample_step_count:
            self.valid_sample_step_count[i, j] = 1
            self.output_rgba[i, j] = tl.vec4(0.0)


class RaycastFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, vr, volume, tf, look_from, sampling_rate, jitter=True):
        ''' Performs Volume Raycasting with the given `volume` and `tf`

        Args:
            ctx (obj): Context used for torch.autograd.Function
            vr (VolumeRaycaster): VolumeRaycaster taichi class
            volume (Tensor): PyTorch Tensor representing the volume of shape (1, D, H, W)
            tf (Tensor): PyTorch Tensor representing a transfer fucntion texture of shape (C, W)
            look_from (Tensor): Look From for Raycaster camera. Shape (3,)
            sampling_rate (float): Sampling rate as multiplier to the Nyquist frequency
            jitter (bool, optional): Turn on ray jitter (random shift of ray starting points). Defaults to True.

        Returns:
            Tensor: Resulting rendered image of shape (C, H, W)
        '''
        # ctx.save_for_backward(volume, tf, look_from)
        ctx.vr = vr # Save Volume Raycaster for backward
        ctx.sampling_rate = sampling_rate
        vr.cam_pos[None] = tl.vec3(look_from.tolist())
        vr.set_volume(volume.squeeze(0).permute(2, 0, 1).contiguous())
        vr.set_tf_tex(tf.permute(1,0).contiguous())
        vr.clear_framebuffer()
        vr.compute_entry_exit(sampling_rate, jitter)
        vr.raycast(sampling_rate)
        vr.get_final_image()
        return vr.output_rgba.to_torch(device=volume.device).permute(2, 1, 0).contiguous()

    @staticmethod
    def backward(ctx, grad_output):
        ctx.vr.output_rga.grad.from_torch(grad_output)
        ctx.vr.get_final_image.grad()
        ctx.vr.raycast.grad(ctx.sampling_rate)

        return ctx.vr.volume.grad.to_torch(device=grad_output.device), \
               ctx.vr.tf_tex.grad.to_torch(device=grad_output.device), \
               None, None, None

class Raycaster(torch.nn.Module):
    def __init__(self, volume_shape, output_shape, tf_shape, sampling_rate=1.0, jitter=True, max_samples=512, fov=30.0, near=0.1, far=100.0):
        super().__init__()
        self.volume_shape = volume_shape
        self.output_shape = output_shape
        self.tf_shape = tf_shape
        self.sampling_rate = sampling_rate
        self.jitter = jitter
        self.vr = VolumeRaycaster(volume_shape, output_shape,
            max_samples=max_samples, tf_resolution=tf_shape, fov=fov, nearfar=(near, far))

    def forward(self, volume, tf, look_from):
        # TODO: assert volume shape
        # TODO: assert tf shape
        return RaycastFunction.apply(self.vr, volume, tf, look_from, self.sampling_rate, self.jitter)

    def extra_repr(self):
        return f'{self.volume_shape=}, {self.output_shape=}, {self.tf_shape=}, {self.vr.max_samples=}'