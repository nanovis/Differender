import taichi as ti
import taichi_glsl as tl
from argparse import ArgumentParser

import numpy as np
import torch
import matplotlib.pyplot as plt

from torchvtk.rendering import plot_tf, plot_tfs

from transfer_functions import get_tf

def fig_to_img(fig):
    fig.set_tight_layout(True)
    fig.set_dpi(100)
    fig.canvas.draw()
    w, h = fig.get_size_inches() * fig.get_dpi()
    w, h = int(w),int(h)
    im = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape((h, w, 3))
    plt.close(fig)
    return im

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

# @ti.func
# def random (vec2 st) {
#     return fract(sin(dot(st.xy,
#                          vec2(12.9898,78.233)))*
#         43758.5453123);
# }

@ti.data_oriented
class VolumeRaycaster():
    def __init__(self, volume_resolution, render_resolution, max_samples=512, tf_resolution=128, fov=30.0, nearfar=(0.1, 100.0)):
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
        self.tf_tex    = ti.Vector.field(4, dtype=ti.f32, needs_grad=True)
        self.tf_momentum = ti.Vector.field(4, dtype=ti.f32)
        self.render_tape    = ti.Vector.field(4, dtype=ti.f32, needs_grad=True)
        self.output_rgba    = ti.Vector.field(4, dtype=ti.f32, needs_grad=True)
        self.reference = ti.Vector.field(4, dtype=ti.f32)
        self.output_rgb   = ti.Vector.field(3, dtype=ti.f32)
        self.valid_sample_step_count   = ti.field(ti.i32)
        self.sample_step_nums = ti.field(ti.i32)
        self.entry = ti.field(ti.f32)
        self.exit  = ti.field(ti.f32)
        self.rays  = ti.Vector.field(3, dtype=ti.f32)
        self.loss = ti.field(ti.f32, (), needs_grad=True)
        self.max_valid_sample_step_count = ti.field(ti.i32, ())
        self.max_samples = max_samples
        self.ambient = 0.4
        self.diffuse = 0.8
        self.specular = 0.3
        self.shininess = 32.0
        self.light_color = tl.vec3(1.0)
        self.cam_pos = ti.Vector.field(3, dtype=ti.f32)
        self.cuda = torch.device("cuda")
        volume_resolution = tuple(map(lambda d: d//4, volume_resolution))
        render_resolution = tuple(map(lambda d: d//16, render_resolution))
        ti.root.dense(ti.ijk, volume_resolution).dense(ti.ijk, (4,4,4)).place(self.volume)
        ti.root.dense(ti.ijk, (*render_resolution, max_samples)).dense(ti.ijk, (16, 16, 1)).place(self.render_tape)
        ti.root.dense(ti.ij,  render_resolution).dense(ti.ij, (16, 16)).place(self.valid_sample_step_count, self.sample_step_nums)
        ti.root.dense(ti.ij,  render_resolution).dense(ti.ij, (16, 16)).place(self.output_rgba, self.reference, self.output_rgb)
        ti.root.dense(ti.ij,  render_resolution).dense(ti.ij, (16, 16)).place(self.entry, self.exit)
        ti.root.dense(ti.ij,  render_resolution).dense(ti.ij, (16, 16)).place(self.rays)
        ti.root.dense(ti.i, tf_resolution).place(self.tf_tex)
        ti.root.dense(ti.i, tf_resolution).place(self.tf_tex.grad, self.tf_momentum)
        ti.root.place(self.cam_pos)
        ti.root.lazy_grad()

    def set_volume(self, volume):self.volume.from_numpy(volume.astype(np.float32))
    def set_tf_tex(self, tf_tex): self.tf_tex.from_numpy(tf_tex.astype(np.float32))
    def set_reference(self, reference): self.reference.from_numpy(reference.astype(np.float32))

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
        up    = tl.cross(right, view_dir).normalized()
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
        pos = tl.clamp(((0.5 * pos) + 0.5), 0.0, 1.0) * ti.static(tl.vec3(*self.volume.shape) - 1.0 - 1e-4)
        x_low, x_high, x_frac = low_high_frac(pos.x)
        y_low, y_high, y_frac = low_high_frac(pos.y)
        z_low, z_high, z_frac = low_high_frac(pos.z)

        x_high = min(x_high, ti.static(self.volume.shape[0]-1))
        y_high = min(y_high, ti.static(self.volume.shape[1]-1))
        z_high = min(z_high, ti.static(self.volume.shape[2]-1))
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
        dx = self.sample_volume_trilinear(pos + x_delta) - self.sample_volume_trilinear(pos - x_delta)
        dy = self.sample_volume_trilinear(pos + y_delta) - self.sample_volume_trilinear(pos - y_delta)
        dz = self.sample_volume_trilinear(pos + z_delta) - self.sample_volume_trilinear(pos - z_delta)
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
        return tl.mix(self.tf_tex[low], self.tf_tex[min(high, ti.static(self.tf_tex.shape[0]-1))], frac)

    @ti.kernel
    def compute_entry_exit(self, sampling_rate: float, jitter: int):
        ''' Produce entry, exit, rays, mask buffers

        Args:
            sampling_rate (float): Sampling rate (multiplier to Nyquist criterium)
            jitter (int): Bool whether to apply jitter or not
        '''
        for i, j in self.entry: # For all pixels
            max_x = ti.static(float(self.render_tape.shape[0]))
            max_y = ti.static(float(self.render_tape.shape[1]))
            look_from = self.cam_pos[None]
            view_dir = (-look_from).normalized()

            bb_bl = ti.static(tl.vec3(-1.0, -1.0, -1.0)) # Bounding Box bottom left
            bb_tr = ti.static(tl.vec3( 1.0,  1.0,  1.0)) # Bounding Box bottom right
            x = (float(i) + 0.5) / max_x # Get pixel centers in range (0,1)
            y = (float(j) + 0.5) / max_y #
            vd = self.get_ray_direction(look_from, view_dir, x, y) # Get exact view direction to this pixel
            tmin, tmax, hit = get_entry_exit_points(look_from, vd, bb_bl, bb_tr) # distance along vd till volume entry and exit, hit bool

            vol_diag = ti.static((tl.vec3(*self.volume.shape) -tl.vec3(1.0)).norm())
            ray_len = tmax - tmin
            n_samples = hit * (ti.floor(sampling_rate * ray_len * vol_diag) + 1) # Number of samples according to https://osf.io/u9qnz
            if jitter:
                tmin += ti.random(dtype=float) * ray_len / n_samples
            self.entry[i, j] = tmin
            self.exit[i, j] = tmax
            self.rays[i, j] = vd
            self.sample_step_nums[i, j] = n_samples

    @ti.kernel
    def raycast(self, sampling_rate: float):
        ''' Produce a rendering. Run compute_entry_exit first! '''
        for i, j in self.valid_sample_step_count: # For all pixels
            for sample_idx in range(self.sample_step_nums[i, j]):
                look_from = self.cam_pos[None]
                if self.render_tape[i, j, sample_idx - 1].w < 0.99 and sample_idx < ti.static(self.max_samples):
                    tmax = self.exit[i, j]
                    n_samples = self.sample_step_nums[i, j]
                    ray_len = (tmax - self.entry[i, j])
                    tmin = self.entry[i, j] + 0.5 * ray_len / n_samples  # Offset tmin as t_start
                    vd = self.rays[i, j]
                    pos = look_from + tl.mix(tmin, tmax, float(sample_idx)/float(n_samples-1)) * vd # Current Pos
                    light_pos = look_from + tl.vec3(0.0, 1.0, 0.0)
                    intensity = self.sample_volume_trilinear(pos)
                    sample_color = self.apply_transfer_function(intensity)
                    opacity = 1.0 - ti.pow(1.0 - sample_color.w, 1.0 / sampling_rate)
                    # if sample_color.w > 1e-3:
                    normal = self.get_volume_normal(pos)
                    light_dir = (pos - light_pos).normalized() # Direction to light source
                    n_dot_l = max(normal.dot(light_dir), 0.0)
                    diffuse = self.diffuse * n_dot_l
                    r = tl.reflect(light_dir, normal) # Direction of reflected light
                    r_dot_v = max(r.dot(-vd), 0.0)
                    specular = self.specular * pow(r_dot_v, self.shininess)
                    shaded_color = tl.vec4((diffuse + specular + self.ambient) * sample_color.xyz * opacity * self.light_color, opacity)
                    self.render_tape[i, j, sample_idx] = (1.0 - self.render_tape[i, j, sample_idx - 1].w) * shaded_color + self.render_tape[i, j, sample_idx - 1]
                    self.valid_sample_step_count[i, j] += 1
                else:
                    self.render_tape[i, j, sample_idx] = self.render_tape[i, j, sample_idx - 1]

    @ti.kernel
    def raycast_nondiff(self, sampling_rate: float):
        for i, j in self.valid_sample_step_count: # For all pixels
            for cnt in range(self.sample_step_nums[i, j]):
                look_from = self.cam_pos[None]
                if self.render_tape[i, j, 0].w < 0.99:
                    tmax = self.exit[i, j]
                    n_samples = self.sample_step_nums[i, j]
                    ray_len = (tmax - self.entry[i, j])
                    tmin = self.entry[i, j] + 0.5 * ray_len / n_samples  # Offset tmin as t_start
                    vd = self.rays[i, j]
                    pos = look_from + tl.mix(tmin, tmax, float(cnt)/float(n_samples-1)) * vd # Current Pos
                    light_pos = look_from + tl.vec3(0.0, 1.0, 0.0)
                    intensity = self.sample_volume_trilinear(pos)
                    sample_color = self.apply_transfer_function(intensity)
                    opacity = 1.0 - ti.pow(1.0 - sample_color.w, 1.0 / sampling_rate)
                    # if sample_color.w > 1e-3:
                    normal = self.get_volume_normal(pos)
                    light_dir = (pos - light_pos).normalized() # Direction to light source
                    n_dot_l = max(normal.dot(light_dir), 0.0)
                    diffuse = self.diffuse * n_dot_l
                    r = tl.reflect(light_dir, normal) # Direction of reflected light
                    r_dot_v = max(r.dot(-vd), 0.0)
                    specular = self.specular * pow(r_dot_v, self.shininess)
                    shaded_color = tl.vec4((diffuse + specular + self.ambient) * sample_color.xyz * opacity * self.light_color, opacity)
                    self.render_tape[i, j, 0] = (1.0 - self.render_tape[i, j, 0].w) * shaded_color + self.render_tape[i, j, 0]
                    self.valid_sample_step_count[i, j] += 1

    @ti.kernel
    def compute_loss(self):
        for i, j in self.output_rgba:
            self.loss[None] += tl.summation((self.output_rgba[i, j] - self.reference[i, j]) ** 2) / ti.static(3.0 * float(self.output_rgba.shape[0] * self.output_rgba.shape[1]))

    @ti.kernel
    def apply_grad(self, lr: float, gamma: float, max_grad: float):
        for i in self.tf_tex:
            self.tf_momentum[i] = gamma * self.tf_momentum[i] + lr * tl.clamp(self.tf_tex.grad[i], -max_grad, max_grad)
            self.tf_tex[i] -= self.tf_momentum[i]
            self.tf_tex[i] = ti.max(self.tf_tex[i], 0)

    @ti.kernel
    def get_final_image(self):
        for i,j in self.valid_sample_step_count:
            valid_sample_step_count = self.valid_sample_step_count[i, j] - 1
            ns = self.sample_step_nums[i, j]
            self.output_rgba[i, j]  += self.render_tape[i, j, ns - 1]
            self.output_rgb[i, j] += self.render_tape[i, j, ns - 1].xyz
            if valid_sample_step_count > self.max_valid_sample_step_count[None]:
                self.max_valid_sample_step_count[None] = valid_sample_step_count

    @ti.kernel
    def get_final_image_nondiff(self):
        for i,j in self.valid_sample_step_count:
            valid_sample_step_count = self.valid_sample_step_count[i, j] - 1
            self.output_rgba[i, j] = self.render_tape[i, j, 0]
            self.output_rgb[i, j] = self.render_tape[i, j, 0].xyz
            if valid_sample_step_count > self.max_valid_sample_step_count[None]:
                self.max_valid_sample_step_count[None] = valid_sample_step_count

    @ti.kernel
    def clear_framebuffer(self):
        self.max_valid_sample_step_count[None] = 0
        for i,j,k in self.render_tape:
            self.render_tape[i, j, k] = tl.vec4(0.0)
        for i,j in self.valid_sample_step_count:
            self.valid_sample_step_count[i, j] = 1
            self.output_rgba[i, j] = tl.vec4(0.0)
            self.output_rgb[i, j] = tl.vec3(0.0)

    def forward(self, sampling_rate=4.0, jitter=False):
        self.clear_framebuffer()
        self.compute_entry_exit(sampling_rate, jitter)
        self.raycast_nondiff(sampling_rate)
        self.get_final_image_nondiff()

    def clear_grad(self):
        self.volume.grad.fill(0.0)
        self.tf_tex.grad.fill(0.0)
        self.render_tape.grad.fill(0.0)
        self.output_rgba.grad.fill(0.0)


    def backward(self, sampling_rate=0.7, jitter=True):
        # cleaning
        self.clear_framebuffer()
        self.clear_grad()
        self.loss[None] = 0.0
        # preparations
        self.compute_entry_exit(sampling_rate, jitter)
        # forward pass and backward pass
        self.raycast(sampling_rate)
        self.get_final_image()

        # self.compute_loss()
        # self.loss.grad[None] = 1.0
        # self.compute_loss.grad()
        reference_tensor = self.reference.to_torch(device=self.cuda)
        output_rgba_tensor = self.output_rgba.to_torch(device=self.cuda).requires_grad_(True)
        loss = torch.nn.functional.mse_loss(output_rgba_tensor, reference_tensor)
        loss.backward()
        self.loss.from_torch(loss)
        self.output_rgba.grad.from_torch(output_rgba_tensor.grad)

        self.get_final_image.grad()
        self.raycast.grad(sampling_rate)


def in_circles(i, y=0.7, dist=2.5):
    x = np.cos(i) * dist
    z = np.sin(i) * dist
    return x, y, z

def rotate_camera(gui):
    gui.get_event()
    if gui.is_pressed('d'):
        return -0.1
    elif gui.is_pressed('a'):
        return 0.1
    else:
        return 0.0



if __name__ == '__main__':
    parser = ArgumentParser('Volume Raycaster')
    parser.add_argument('task', type=str, help='Either forward or backward')
    parser.add_argument('--res', type=int, default=240, help='Render Resolution')
    parser.add_argument('--tf-res', type=int, default=128, help='Transfer Function Texture Resolution')
    parser.add_argument('--iterations', type=int, default=240, help='Number of iterations for TF optimization in backward')
    parser.add_argument('--debug', action='store_true', help='Turns on fancy logs')
    parser.add_argument('--ref', action='store_true', help='Create Reference Images')
    parser.add_argument('--max-samples', type=int, default=2048, help='Max number of samples to use in backward pass')
    parser.add_argument('--fw-sampling-rate', type=float, default=8.0, help='Sampling Rate to use during forward pass')
    parser.add_argument('--bw-sampling-rate', type=float, default=2.0, help='Sampling Rate to use during backward pass')
    parser.add_argument('--lr', type=float, default=0.1, help='Learning rate for TF optimization')
    parser.add_argument('--mom', type=float, default=0.9, help='Momentum for gradient descent')
    parser.add_argument('--clip-grads', type=float, default=0.1, help='Clips Gradient absolute to this value')
    parser.add_argument('--lr-decay', type=float, default=0.99, help='Learning rate is multiplied with this factor every iteration')
    parser.add_argument('--bw-jitter', action='store_true', help='Enable ray jitter when in backward mode')
    parser.add_argument('--target-tf', type=str, default='tf5', help='Target Transfer Function, see transfer_function.py')
    parser.add_argument('--init-tf', type=str, default='black', help='Initial Transfer function (when optimizing)')


    args = parser.parse_args()
    RESOLUTION = (args.res, args.res)
    RESOLUTION_T = tuple(map(lambda d: d//16, RESOLUTION))
    TF_RESOLUTION = args.tf_res
    if args.debug:
        ti.init(arch=ti.cuda, debug=True, excepthook=True, log_level=ti.TRACE, kernel_profiler=True, default_fp=ti.f32)
    else:
        ti.init(arch=ti.cuda, default_fp=ti.f32)
    # Data
    tf = get_tf(args.target_tf, TF_RESOLUTION)
    tf_init = get_tf(args.init_tf, TF_RESOLUTION)
    # vol_ds = TorchDataset('/run/media/dome/Data/data/torchvtk/CQ500_256')
    # vol = vol_ds[0]['vol'].squeeze().permute(2, 0, 1).contiguous().numpy()
    vol = np.swapaxes(np.fromfile('data/skull.raw', dtype=np.uint8).reshape(256,256,256), 0, 1).astype(np.float32) / 255.0

    # Renderer
    vr = VolumeRaycaster(volume_resolution=vol.shape, max_samples=args.max_samples, render_resolution=RESOLUTION, tf_resolution=TF_RESOLUTION)
    t = np.pi * 1.5

    if args.task == 'backward':
        gui_bw = ti.GUI("Volume Raycaster (Backward)", res=RESOLUTION, fast_gui=True)
        gui_fw = ti.GUI("Volume Raycaster (Forward)", res=RESOLUTION, fast_gui=True)
        gui_tf = ti.GUI("Transfer Function Comparison", res=(640,480))
        render_video = ti.VideoManager(output_dir='results/bw_render', framerate=24, automatic_build=False)
        render_video_fw = ti.VideoManager(output_dir='results/fw_render', framerate=24, automatic_build=False)
        tf_video = ti.VideoManager(output_dir='results/tf', framerate=24, automatic_build=False)
        # Setup Raycaster
        vr.set_volume(vol)
        vr.cam_pos[None] = tl.vec3(*in_circles(t))
        # Create Reference if necessary
        if args.ref: # Generate new reference
            vr.set_tf_tex(tf)
            vr.forward(args.fw_sampling_rate, jitter=False)
            plot_tf(vr.tf_tex.to_torch().permute(1,0).contiguous()).savefig('temp_tf_reference.png')
            gui_fw.set_image(vr.output_rgb)
            gui_fw.show()
            gui_bw.set_image(vr.output_rgb)
            gui_bw.show()
            ti.imwrite(vr.output_rgba, 'temp_reference.png')
            vr.set_reference(vr.output_rgba.to_numpy())
        else: # Use old reference
            vr.set_reference(ti.imread('temp_reference.png') / 255.0)
        # Optimize for Transfer Function
        vr.set_tf_tex(tf_init) # initial tf
        lr = args.lr
        for i in range(args.iterations):
            # Optimization
            vr.backward(args.bw_sampling_rate, jitter=args.bw_jitter) # computes grads
            vr.apply_grad(lr, args.mom, args.clip_grads) # Gradient descent with momentum
            lr *= args.lr_decay # decay

            # Log Backward Pass
            gui_bw.set_image(vr.output_rgb)
            render_video.write_frame(vr.output_rgb)
            gui_bw.show()
            tf_pt = vr.tf_tex.to_torch().permute(1,0).contiguous()
            tf_grad_np = vr.tf_tex.grad.to_numpy()
            print(f'\n[{i:05d} ========== Loss: ', vr.loss, ' ==========]')
            print('Max Samples:', vr.max_valid_sample_step_count, '/', vr.max_samples)
            print('Learning Rate:', lr)
            print(f'TF Gradients: {np.abs(tf_grad_np).max(axis=0)}')
            # Log Transfer Function
            tf_im = np.rot90(fig_to_img(plot_tfs([tf_pt, torch.from_numpy(tf).permute(1,0)], ['Prediction', 'Target'])), k=3)
            gui_tf.set_image(tf_im)
            gui_tf.show()
            tf_video.write_frame(tf_im)

            # Standard forward pass for reference image
            vr.forward(args.fw_sampling_rate, jitter=False)
            gui_fw.set_image(vr.output_rgb)
            render_video_fw.write_frame(vr.output_rgb)
            gui_fw.show()
        # Make GIFs
        render_video.make_video(gif=True, mp4=False)
        render_video_fw.make_video(gif=True, mp4=False)
        tf_video.make_video(gif=True, mp4=False)

    elif args.task == 'forward':
        gui = ti.GUI("Volume Raycaster", res=RESOLUTION, fast_gui=True)
        # Setup Raycaster
        vr.set_volume(vol)
        vr.set_tf_tex(tf)
        # Render volume
        while gui.running:
            vr.cam_pos[None] = tl.vec3(*in_circles(t))
            vr.forward(args.fw_sampling_rate, False)
            print('Max number of steps:', vr.max_valid_sample_step_count, '/', vr.max_samples)

            t += rotate_camera(gui)
            gui.set_image(vr.output_rgb)
            gui.show()
            if args.ref:
                ti.imwrite(vr.output_rgba, 'reference_skull.png')
                plot_tf(vr.tf_tex.to_torch().permute(1,0).contiguous()).savefig('reference_tf_skull.png')
                break

    else:
        raise Exception(f'invalid task given: {args.task}')
