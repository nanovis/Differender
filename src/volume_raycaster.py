import taichi as ti
import taichi_glsl as tl

import numpy as np

from utils import load_head_data

from torchvtk.utils import TFGenerator, tex_from_pts

import matplotlib.pyplot as plt

# Taichi global fields

@ti.func
def low_high_frac(x: float):
    ''' Returns the integer value below and above, as well as the frac

    Args:
        x (float): Floating point number

    Returns:
        int, int, float: floor, ceil, frac of `x`
    '''
    low = tl.floor(x)
    high = low + 1.0
    frac = x - low
    return int(low), int(high), frac

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
    def __init__(self, resolution=(240, 240), fov=60.0, nearfar=(0.1, 100.0)):
        ''' Initializes Volume Raycaster. Make sure to .set_volume() and .set_tf_tex() after initialization

        Args:
            resolution (2-tuple of int, optional): Resolution of the rendering (w,h). Defaults to (240, 240).
            fov (float, optional): Field of view of the camera in degrees. Defaults to 60.0.
            nearfar (2-tuple of float, optional): Near and far plane distance used for perspective projection. Defaults to (0.1, 100.0).
        '''
        self.resolution = resolution
        self.aspect = resolution[0] / resolution[1]
        self.fov_deg = fov
        self.fov_rad = np.radians(fov)
        self.near, self.far = nearfar
        # Taichi Fields
        self.volume = ti.field(ti.f32, shape=(184, 256, 170))
        self.tf_tex = ti.Vector.field(4, dtype=ti.f32, shape=(128,))
        self.render = ti.Vector.field(3, dtype=ti.f32, shape=resolution)
        self.opacity = ti.field(ti.f32, shape=resolution)

    def set_volume(self, volume): self.volume.from_numpy(volume)
    def set_tf_tex(self, tf_tex): self.tf_tex.from_numpy(tf_tex)

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
        pos = (0.5*pos) + 0.5
        x_low, x_high, x_frac = low_high_frac(pos.x)
        y_low, y_high, y_frac = low_high_frac(pos.y)
        z_low, z_high, z_frac = low_high_frac(pos.z)
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
    def apply_transfer_function(self, intensity: float):
        ''' Applies a 1D transfer function to a given intensity value

        Args:
            intensity (float): Intensity

        Returns:
            tl.vec4: Color and opacity for given `intensity`
        '''
        length = ti.static(float(self.tf_tex.shape[0]))
        low, high, frac = low_high_frac(intensity * length)
        return tl.mix(self.tf_tex[low], self.tf_tex[high], frac)

    @ti.kernel
    def raycast(self, cam_pos_x: float, cam_pos_y: float, cam_pos_z: float, sampling_rate: float):
        ''' Produce a rendering

        Args:
            cam_pos_x (float): Camera Pos x
            cam_pos_y (float): Camera Pos y
            cam_pos_z (float): Camera Pos z
            sampling_rate (float): Sampling rate along the ray
        '''
        max_x = ti.static(float(self.render.shape[0]))
        max_y = ti.static(float(self.render.shape[1]))
        vol_diag = ti.static((tl.vec3(*self.volume.shape) - tl.vec3(1.0)).norm())
        look_from = tl.vec3(cam_pos_x, cam_pos_y, cam_pos_z)
        view_dir = (-look_from).normalized()
        right_dir = tl.cross(view_dir, ti.static(tl.vec3(0.0, 1.0, 0.0))).normalized()
        up_dir    = tl.cross(right_dir, view_dir).normalized()

        bb_bl = ti.static(tl.vec3(-1.0, -1.0, -1.0)) # Bounding Box bottom left
        bb_tr = ti.static(tl.vec3( 1.0,  1.0,  1.0)) # Bounding Box bottom right
        for i, j in self.render: # For all pixels
            x = (float(i) + 0.5) / max_x # Get pixel centers in range (0,1)
            y = (float(j) + 0.5) / max_y #
            vd = self.get_ray_direction(look_from, view_dir, x, y) # Get exact view direction to this pixel
            tmin, tmax, hit = get_entry_exit_points(look_from, vd, bb_bl, bb_tr) # distance along vd till volume entry and exit, hit bool
            if hit: # Shoot the ray
                ray_len = (tmax - tmin)
                n_samples = ti.ceil(sampling_rate * ray_len * vol_diag) # Number of samples according to https://osf.io/u9qnz
                t_inc = ray_len / n_samples # Increment for steps along vd
                t = tmin + 0.5 * t_inc # Start of ray is look_from + t * vd
                color = tl.vec4(0.0) # Color to be composited
                cnt = 0
                while cnt < n_samples and color.w < 0.99: # Early ray termination
                    pos = look_from + t * vd # Current pos at this step
                    intensity = self.sample_volume_trilinear(pos)     # Sample intensity
                    s_color = self.apply_transfer_function(intensity) # Sample Color and opacity
                    s_color = tl.vec4(s_color.xyz * s_color.w, s_color.w) # pre-multiply alpha
                    color += (1.0 - color.w) * s_color  # Composite color
                    t += t_inc
                    cnt += 1
                # Save result to global buffers
                self.render[i, j] = color.xyz
                self.opacity[i,j] = color.w
            else: # Skip empty space
                self.render[i, j] = tl.vec3(0.0)
                self.opacity[i,j] = 0.0


if __name__ == '__main__':
    ti.init(arch=ti.cpu, debug=True, advanced_optimization=False, excepthook=True, log_level=ti.TRACE)
    window = ti.GUI("Volume Raycaster", res=(400, 400), fast_gui=False)
    vr = VolumeRaycaster(resolution=(400,400))
    print('GUI Resolution:', window.res, '  Framebuffer shape: ', vr.render.shape)
    tfgen = TFGenerator()
    tf = tex_from_pts(tfgen.generate(), 128).numpy().astype(np.float32)
    vol = (load_head_data() / float(np.iinfo(np.uint16).max)).astype(np.float32)
    vr.set_volume(vol)
    vr.set_tf_tex(tf)
    while window.running:
        vr.raycast(0.0, 0.1, 3.0, 2.0)
        render_np = vr.render.to_numpy()
        # plt.imshow(render_np)
        # plt.show()
        print(vr.render.dtype, vr.render.shape, vr.render.n, vr.render.m)
        print(render_np.dtype, render_np.shape, render_np.min(), render_np.max())
        window.set_image(vr.render)
        window.show()
        break
