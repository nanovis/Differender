import torch
from torch.cuda.amp import autocast, custom_fwd, custom_bwd
import taichi as ti
import taichi.math as tm
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
        look_from (tm.vec3): Camera Position as vec3
        view_dir (tm.vec): View direction as vec3, normalized
        bl (tm.vec3): Bottom left of the bounding box
        tr (tm.vec3): Top right of the bounding box

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
    if tmax < 0.0 or tmin > tmax:
        hit = False
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
        self.light_color = tm.vec3(1.0)
        self.cam_pos = ti.Vector.field(3, dtype=ti.f32)
        volume_resolution = tuple(map(lambda d: d // 4, volume_resolution))
        render_resolution = tuple(map(lambda d: d // 8, render_resolution))
        ti.root.dense(ti.ijk, volume_resolution) \
            .dense(ti.ijk, (4, 4, 4)) \
            .place(self.volume)
        ti.root.dense(ti.ijk, (*render_resolution, max_samples)) \
            .dense(ti.ijk, (8, 8, 1)) \
            .place(self.render_tape)
        ti.root.dense(ti.ij, render_resolution) \
            .dense(ti.ij, (8, 8)) \
            .place(self.valid_sample_step_count, self.sample_step_nums)
        ti.root.dense(ti.ij, render_resolution).dense(ti.ij, (8, 8)).place(self.output_rgba)
        ti.root.dense(ti.ij, render_resolution).dense(ti.ij, (8, 8)).place(self.entry, self.exit)
        ti.root.dense(ti.ij, render_resolution).dense(ti.ij, (8, 8)).place(self.rays)
        ti.root.dense(ti.i, tf_resolution).place(self.tf_tex, self.tf_tex.grad)
        ti.root.place(self.cam_pos)
        ti.root.lazy_grad()

    def set_volume(self, volume):
        self.volume.from_torch(volume.float())

    def set_tf_tex(self, tf_tex):
        self.tf_tex.from_torch(tf_tex.float())

    def set_cam_pos(self, cam_pos):
        self.cam_pos.from_torch(cam_pos.float())

    @ti.func
    def get_ray_direction(self, orig, view_dir, x: float, y: float):
        ''' Compute ray direction for perspecive camera.

        Args:
            orig (tm.vec3): Camera position
            view_dir (tm.vec3): View direction, normalized
            x (float): Image coordinate in [0,1] along width
            y (float): Image coordinate in [0,1] along height

        Returns:
            tm.vec3: Ray direction from camera origin to pixel specified through `x` and `y`
        '''
        u = x - 0.5
        v = y - 0.5

        up = tm.vec3(0.0, 1.0, 0.0)
        right = tm.cross(view_dir, up).normalized()
        up = tm.cross(right, view_dir).normalized()
        near_h = 2.0 * ti.tan(self.fov_rad) * self.near
        near_w = near_h * self.aspect
        near_m = orig + self.near * view_dir
        near_pos = near_m + u * near_w * right + v * near_h * up

        return (near_pos - orig).normalized()

    @ti.func
    def sample_volume_trilinear(self, pos):
        ''' Samples volume data at `pos` and trilinearly interpolates the value

        Args:
            pos (tm.vec3): Position to sample the volume in [-1, 1]^3

        Returns:
            float: Sampled interpolated intensity
        '''
        pos = tm.clamp(
            ((0.5 * pos) + 0.5), 0.0,
            1.0) * ti.static(tm.vec3(*self.volume.shape) - 1.0 - 1e-4)
        x_low, x_high, x_frac = low_high_frac(pos.x)
        y_low, y_high, y_frac = low_high_frac(pos.y)
        z_low, z_high, z_frac = low_high_frac(pos.z)

        x_high = min(x_high, ti.static(self.volume.shape[0] - 1))
        y_high = min(y_high, ti.static(self.volume.shape[1] - 1))
        z_high = min(z_high, ti.static(self.volume.shape[2] - 1))
        # on z_low
        v000 = self.volume[x_low, y_low, z_low]
        v100 = self.volume[x_high, y_low, z_low]
        x_val_y_low = tm.mix(v000, v100, x_frac)
        v010 = self.volume[x_low, y_high, z_low]
        v110 = self.volume[x_high, y_high, z_low]
        x_val_y_high = tm.mix(v010, v110, x_frac)
        xy_val_z_low = tm.mix(x_val_y_low, x_val_y_high, y_frac)
        # on z_high
        v001 = self.volume[x_low, y_low, z_high]
        v101 = self.volume[x_high, y_low, z_high]
        x_val_y_low = tm.mix(v001, v101, x_frac)
        v011 = self.volume[x_low, y_high, z_high]
        v111 = self.volume[x_high, y_high, z_high]
        x_val_y_high = tm.mix(v011, v111, x_frac)
        xy_val_z_high = tm.mix(x_val_y_low, x_val_y_high, y_frac)
        return tm.mix(xy_val_z_low, xy_val_z_high, z_frac)

    @ti.func
    def get_volume_normal(self, pos):
        delta = 1e-3
        x_delta = tm.vec3(delta, 0.0, 0.0)
        y_delta = tm.vec3(0.0, delta, 0.0)
        z_delta = tm.vec3(0.0, 0.0, delta)
        dx = self.sample_volume_trilinear(
            pos + x_delta) - self.sample_volume_trilinear(pos - x_delta)
        dy = self.sample_volume_trilinear(
            pos + y_delta) - self.sample_volume_trilinear(pos - y_delta)
        dz = self.sample_volume_trilinear(
            pos + z_delta) - self.sample_volume_trilinear(pos - z_delta)
        return tm.vec3(dx, dy, dz).normalized()

    @ti.func
    def apply_transfer_function(self, intensity: float):
        ''' Applies a 1D transfer function to a given intensity value

        Args:
            intensity (float): Intensity in [0,1]

        Returns:
            tm.vec4: Color and opacity for given `intensity`
        '''
        length = ti.static(float(self.tf_tex.shape[0] - 1))
        low, high, frac = low_high_frac(intensity * length)
        return tm.mix(
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

            bb_bl = ti.static(tm.vec3(-1.0, -1.0, -1.0))  # Bounding Box bottom left
            bb_tr = ti.static(tm.vec3(1.0, 1.0, 1.0))  # Bounding Box bottom right
            x = (float(i) + 0.5) / max_x  # Get pixel centers in range (0,1)
            y = (float(j) + 0.5) / max_y  #
            vd = self.get_ray_direction(
                look_from, view_dir, x,
                y)  # Get exact view direction to this pixel
            tmin, tmax, hit = get_entry_exit_points(
                look_from, vd, bb_bl, bb_tr
            )  # distance along vd till volume entry and exit, hit bool

            vol_diag = ti.static((tm.vec3(*self.volume.shape) - tm.vec3(1.0)).norm())
            ray_len = tmax - tmin
            # Number of samples according to https://osf.io/u9qnz
            n_samples = hit * (ti.floor(sampling_rate * ray_len * vol_diag) + 1)
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
                if self.render_tape[i, j, sample_idx - 1].w < 0.99 and sample_idx < ti.static(self.max_samples):
                    tmax = self.exit[i, j]
                    n_samples = self.sample_step_nums[i, j]
                    ray_len = (tmax - self.entry[i, j])
                    tmin = self.entry[i, j] + 0.5 * ray_len / n_samples  # Offset tmin as t_start
                    vd = self.rays[i, j]
                    pos = look_from + tm.mix(tmin, tmax, float(sample_idx) / float(n_samples - 1)) * vd  # Current Pos
                    light_pos = look_from + tm.vec3(0.0, 1.0, 0.0)
                    intensity = self.sample_volume_trilinear(pos)
                    sample_color = self.apply_transfer_function(intensity)
                    opacity = 1.0 - ti.pow(1.0 - sample_color.w, 1.0 / sampling_rate)
                    # if sample_color.w > 1e-3:
                    normal = self.get_volume_normal(pos)
                    light_dir = (pos - light_pos).normalized()  # Direction to light source
                    n_dot_l = max(normal.dot(light_dir), 0.0)
                    diffuse = self.diffuse * n_dot_l
                    r = tm.reflect(light_dir, normal)  # Direction of reflected light
                    r_dot_v = max(r.dot(-vd), 0.0)
                    specular = self.specular * pow(r_dot_v, self.shininess)
                    shaded_color = tm.vec4(ti.min(1.0, diffuse + specular + self.ambient) *
                                           sample_color.xyz * opacity * self.light_color, opacity)
                    self.render_tape[i, j, sample_idx] = (1.0 - self.render_tape[i, j, sample_idx - 1].w) \
                                                         * shaded_color + self.render_tape[i, j, sample_idx - 1]
                    self.valid_sample_step_count[i, j] += 1
                else:
                    self.render_tape[i, j, sample_idx] = self.render_tape[i, j, sample_idx - 1]

    @ti.kernel
    def raycast_nondiff(self, sampling_rate: float):
        ''' Raycasts in a non-differentiable (but faster and cleaner) way. Use `get_final_image_nondiff` with this.

        Args:
            sampling_rate (float): Sampling rate (multiplier with Nyquist frequence)
        '''
        for i, j in self.valid_sample_step_count:  # For all pixels
            for cnt in range(self.sample_step_nums[i, j]):
                look_from = self.cam_pos[None]
                if self.render_tape[i, j, 0].w < 0.99:
                    tmax = self.exit[i, j]
                    n_samples = self.sample_step_nums[i, j]
                    ray_len = (tmax - self.entry[i, j])
                    tmin = self.entry[i, j] + 0.5 * ray_len / n_samples  # Offset tmin as t_start
                    vd = self.rays[i, j]
                    pos = look_from + tm.mix(
                        tmin, tmax,
                        float(cnt) / float(n_samples - 1)) * vd  # Current Pos
                    light_pos = look_from + tm.vec3(0.0, 1.0, 0.0)
                    intensity = self.sample_volume_trilinear(pos)
                    sample_color = self.apply_transfer_function(intensity)
                    opacity = 1.0 - ti.pow(1.0 - sample_color.w,
                                           1.0 / sampling_rate)
                    if sample_color.w > 1e-3:
                        normal = self.get_volume_normal(pos)
                        light_dir = (pos - light_pos).normalized(
                        )  # Direction to light source
                        n_dot_l = max(normal.dot(light_dir), 0.0)
                        diffuse = self.diffuse * n_dot_l
                        r = tm.reflect(light_dir,
                                       normal)  # Direction of reflected light
                        r_dot_v = max(r.dot(-vd), 0.0)
                        specular = self.specular * pow(r_dot_v, self.shininess)
                        shaded_color = tm.vec4((diffuse + specular + self.ambient) *
                                               sample_color.xyz * opacity * self.light_color,
                                               opacity)
                        self.render_tape[i, j, 0] = (1.0 - self.render_tape[i, j, 0].w) \
                                                    * shaded_color + self.render_tape[i, j, 0]

    @ti.kernel
    def get_final_image_nondiff(self):
        ''' Retrieves the final image from the tape if the `raycast_nondiff` method was used. '''
        for i, j in self.valid_sample_step_count:
            valid_sample_step_count = self.valid_sample_step_count[i, j] - 1
            self.output_rgba[i, j] = ti.min(1.0, self.render_tape[i, j, 0])
            if valid_sample_step_count > self.max_valid_sample_step_count[None]:
                self.max_valid_sample_step_count[None] = valid_sample_step_count

    @ti.kernel
    def get_final_image(self):
        ''' Retrieves the final image from the `render_tape` to `output_rgba`. '''
        for i, j in self.valid_sample_step_count:
            valid_sample_step_count = self.valid_sample_step_count[i, j] - 1
            ns = self.sample_step_nums[i, j]
            self.output_rgba[i, j] += self.render_tape[i, j, ns - 1]
            if valid_sample_step_count > self.max_valid_sample_step_count[None]:
                self.max_valid_sample_step_count[None] = valid_sample_step_count

    @ti.kernel
    def clear_framebuffer(self):
        ''' Clears the framebuffer `output_rgba` and the `render_tape`'''
        self.max_valid_sample_step_count[None] = 0
        for i, j, k in self.render_tape:
            self.render_tape[i, j, k] = tm.vec4(0.0)
        for i, j in self.valid_sample_step_count:
            self.valid_sample_step_count[i, j] = 1
            self.output_rgba[i, j] = tm.vec4(0.0)

    def clear_grad(self):
        ''' Clears the Taichi gradients. '''
        self.volume.grad.fill(0.0)
        self.tf_tex.grad.fill(0.0)
        self.render_tape.grad.fill(0.0)
        self.output_rgba.grad.fill(0.0)


class RaycastFunction(torch.autograd.Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, vr, volume, tf, look_from, sampling_rate, batched, jitter=True):
        ''' Performs Volume Raycasting with the given `volume` and `tf`

        Args:
            ctx (obj): Context used for torch.autograd.Function
            vr (VolumeRaycaster): VolumeRaycaster taichi class
            volume (Tensor): PyTorch Tensor representing the volume of shape ([BS,] W, H, D)
            tf (Tensor): PyTorch Tensor representing a transfer fucntion texture of shape ([BS,] W, C)
            look_from (Tensor): Look From for Raycaster camera. Shape ([BS,] 3)
            sampling_rate (float): Sampling rate as multiplier to the Nyquist frequency
            batched (4-bool): Whether the input is batched (i.e. has an extra dimension or is a list) and a bool for each volume, tf and look_from
            jitter (bool, optional): Turn on ray jitter (random shift of ray starting points). Defaults to True.

        Returns:
            Tensor: Resulting rendered image of shape (C, H, W)
        '''
        ctx.vr = vr  # Save Volume Raycaster for backward
        ctx.sampling_rate = sampling_rate
        ctx.batched, ctx.bs = batched
        ctx.jitter = jitter
        if ctx.batched:  # Batched Input
            ctx.save_for_backward(volume, tf, look_from)  # unwrap tensor if it's a list
            result = torch.zeros(ctx.bs, *vr.resolution, 4, dtype=torch.float32, device=volume.device)
            for i, vol, tf_, lf in zip(range(ctx.bs), volume, tf, look_from):
                vr.set_cam_pos(lf)
                vr.set_volume(vol)
                vr.set_tf_tex(tf_)
                vr.clear_framebuffer()
                vr.compute_entry_exit(sampling_rate, jitter)
                vr.raycast(sampling_rate)
                vr.get_final_image()
                result[i] = vr.output_rgba.to_torch(device=volume.device)
            return result
        else:  # Non-batched, single item
            # No saving via ctx.save_for_backward needed for single example, as it's saved inside vr
            # TODO: is this a problem when using the Raycast multiple times, before calling backward()?
            vr.set_cam_pos(look_from)
            vr.set_volume(volume)
            vr.set_tf_tex(tf)
            vr.clear_framebuffer()
            vr.compute_entry_exit(sampling_rate, jitter)
            vr.raycast(sampling_rate)
            vr.get_final_image()
            return vr.output_rgba.to_torch(device=volume.device)

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        dev = grad_output.device
        if ctx.batched:  # Batched Gradient
            vols, tfs, lfs = ctx.saved_tensors
            # Volume Grad Shape (BS, W, H, D)
            volume_grad = torch.zeros(ctx.bs, *ctx.vr.volume.shape, dtype=torch.float32, device=dev)
            # TF Grad Shape (BS, W, C)
            tf_grad = torch.zeros(ctx.bs, *ctx.vr.tf_tex.shape, ctx.vr.tf_tex.n, dtype=torch.float32, device=dev)
            for i, vol, tf, lf in zip(range(ctx.bs), vols, tfs, lfs):
                ctx.vr.clear_grad()
                ctx.vr.set_cam_pos(lf)
                ctx.vr.set_volume(vol)
                ctx.vr.set_tf_tex(tf)
                ctx.vr.clear_framebuffer()
                ctx.vr.compute_entry_exit(ctx.sampling_rate, ctx.jitter)
                ctx.vr.raycast(ctx.sampling_rate)
                ctx.vr.get_final_image()
                ctx.vr.output_rgba.grad.from_torch(grad_output[i])
                ctx.vr.get_final_image.grad()
                ctx.vr.raycast.grad(ctx.sampling_rate)

                volume_grad[i] = torch.nan_to_num(ctx.vr.volume.grad.to_torch(device=dev))
                tf_grad[i] = torch.nan_to_num(ctx.vr.tf_tex.grad.to_torch(device=dev))
            return None, volume_grad, tf_grad, None, None, None, None

        else:  # Non-batched, single item
            ctx.vr.clear_grad()
            ctx.vr.output_rgba.grad.from_torch(grad_output)
            ctx.vr.get_final_image.grad()
            ctx.vr.raycast.grad(ctx.sampling_rate)

            return None, \
                   torch.nan_to_num(ctx.vr.volume.grad.to_torch(device=dev)), \
                   torch.nan_to_num(ctx.vr.tf_tex.grad.to_torch(device=dev)), \
                   None, None, None, None


class Raycaster(torch.nn.Module):
    def __init__(self, volume_shape, output_shape, tf_shape, sampling_rate=1.0, jitter=True, max_samples=512, fov=30.0,
                 near=0.1, far=100.0, ti_kwargs={}):
        super().__init__()
        self.volume_shape = (volume_shape[2], volume_shape[0], volume_shape[1])
        self.output_shape = output_shape
        self.tf_shape = tf_shape
        self.sampling_rate = sampling_rate
        self.jitter = jitter
        ti.init(arch=ti.cuda, default_fp=ti.f32, **ti_kwargs)
        self.vr = VolumeRaycaster(self.volume_shape, output_shape,
                                  max_samples=max_samples, tf_resolution=tf_shape, fov=fov, nearfar=(near, far))

    def raycast_nondiff(self, volume, tf, look_from, sampling_rate=None):
        with torch.no_grad() as _, autocast(False) as _:
            batched, bs, vol_in, tf_in, lf_in = self._determine_batch(volume, tf, look_from)
            sr = sampling_rate if sampling_rate is not None else 4.0 * self.sampling_rate
            if batched:  # Batched Input
                result = torch.zeros(bs,
                                     *self.vr.resolution,
                                     4,
                                     dtype=torch.float32,
                                     device=volume.device)
                # Volume: remove intensity dim, reorder to (BS, W, H, D)
                # TF: Reorder to (BS, W, 4)
                for i, vol, tf_, lf in zip(range(bs), vol_in, tf_in, lf_in):
                    with autocast(False):
                        self.vr.set_cam_pos(lf)
                    self.vr.set_volume(vol)
                    self.vr.set_tf_tex(tf_)
                    self.vr.clear_framebuffer()
                    self.vr.compute_entry_exit(sr, False)
                    self.vr.raycast_nondiff(sr)
                    self.vr.get_final_image_nondiff()
                    result[i] = self.vr.output_rgba.to_torch(device=volume.device)
                # First reorder render to (BS, C, H, W), then flip Y to correct orientation
                return torch.flip(result, (2,)).permute(0, 3, 2, 1).contiguous()
            else:
                self.vr.set_cam_pos(lf_in)
                self.vr.set_volume(vol_in)
                self.vr.set_tf_tex(tf_in)
                self.vr.clear_framebuffer()
                self.vr.compute_entry_exit(sr, False)
                self.vr.raycast_nondiff(sr)
                self.vr.get_final_image_nondiff()
                # First reorder to (C, H, W), then flip Y to correct orientation
                return torch.flip(self.vr.output_rgba.to_torch(device=volume.device), (1,)) \
                    .permute(2, 1, 0).contiguous()

    def forward(self, volume, tf, look_from):
        ''' Raycasts through `volume` using the transfer function `tf` from given camera position (volume is in [-1,1]^3, centered around 0)

        Args:
            volume (Tensor): Volume Tensor of shape ([BS,] 1, D, H, W)
            tf (Tensor): Transfer Function Texture of shape ([BS,] 4, W)
            look_from (Tensor): Camera position of shape ([BS,] 3)

        Returns:
            Tensor: Rendered image of shape ([BS,] 4, H, W)
        '''
        batched, bs, vol_in, tf_in, lf_in = self._determine_batch(volume, tf, look_from)
        if batched:  # Anything batched Batched
            return torch.flip(
                RaycastFunction.apply(self.vr, vol_in, tf_in, lf_in, self.sampling_rate, (batched, bs), self.jitter),
                (2,)  # First reorder render to (BS, C, H, W), then flip Y to correct orientation
            ).permute(0, 3, 2, 1).contiguous()
        else:
            return torch.flip(RaycastFunction.apply(self.vr, vol_in, tf_in, lf_in, self.sampling_rate, (batched, bs),
                                                    self.jitter),
                              (1,)  # First reorder to (C, H, W), then flip Y to correct orientation
                              ).permute(2, 1, 0).contiguous()

    def _determine_batch(self, volume, tf, look_from):
        ''' Determines whether there's a batched input and returns lists of non-batched inputs.

        Args:
            volume (Tensor): Volume input, either 4D or 5D (batched)
            tf (Tensor): Transfer Function input, either 2D or 3D (batched)
            look_from (Tensor): Camera Look From input, either 1D or 2D (batched)

        Returns:
            ([bool], Tensor, Tensor, Tensor): (is anything batched?, batched input or list of non-batched inputs (for all inputs))
        '''
        batched = torch.tensor([volume.ndim == 5, tf.ndim == 3, look_from.ndim == 2])

        if batched.any():
            bs = [volume, tf, look_from][batched.long().argmax().item()].size(0)
            vol_out = volume.squeeze(1).permute(0, 3, 1, 2).contiguous() if batched[0].item() \
                else volume.squeeze(0).permute(2, 0, 1).expand(bs, -1, -1, -1).clone()
            tf_out = tf.permute(0, 2, 1).contiguous() if batched[1].item() \
                else tf.permute(1, 0).expand(bs, -1, -1).clone()
            lf_out = look_from if batched[2].item() else look_from.expand(bs, -1).clone()
            return True, bs, vol_out, tf_out, lf_out
        else:
            return False, 0, volume.squeeze(0).permute(2, 0, 1).contiguous(), tf.permute(1, 0).contiguous(), look_from

    def extra_repr(self):
        return f'Volume ({self.volume_shape}), Output Render ({self.output_shape}), TF ({self.tf_shape}), Max Samples = {self.vr.max_samples}'
