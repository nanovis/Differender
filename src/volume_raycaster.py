import taichi as ti
import taichi_glsl as tl

import timeit

import numpy as np
import matplotlib.pyplot as plt

from torchvtk.utils import TFGenerator, tex_from_pts
from torchvtk.datasets import TorchDataset

from utils import load_head_data

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
    def __init__(self, volume_resolution, render_resolution, tf_resolution=128, fov=30.0, nearfar=(0.1, 100.0)):
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
        self.volume = ti.field(ti.f32)
        self.tf_tex = ti.Vector.field(4, dtype=ti.f32, shape=tf_resolution)
        self.render = ti.Vector.field(3, dtype=ti.f32)
        self.ambient = 0.4
        self.diffuse = 0.8
        self.specular = 0.3
        self.shininess = 32.0
        self.light_color = tl.vec3(1.0)
        volume_resolution = tuple(map(lambda d: d//4, volume_resolution))
        render_resolution = tuple(map(lambda d: d//16, render_resolution))
        ti.root.dense(ti.ijk, volume_resolution).dense(ti.ijk, (4,4,4)).place(self.volume)
        ti.root.dense(ti.ij,  render_resolution).dense(ti.ij,  (16, 16)).place(self.render)


    def set_volume(self, volume):
        self.volume.from_numpy(volume)

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
    def sample_volume_trilinear(self, x: float, y:float, z:float):
        ''' Samples volume data at `pos` and trilinearly interpolates the value

        Args:
            pos (tl.vec3): Position to sample the volume in [-1, 1]^3

        Returns:
            float: Sampled interpolated intensity
        '''
        pos = ((0.5*tl.vec3(x, y, z)) + 0.5) * ti.static(tl.vec3(*self.volume.shape) - 1 -1e-6)
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
    def get_volume_normal(self, pos):
        delta = 1e-3
        dx = self.sample_volume_trilinear(pos.x + delta, pos.y, pos.z) - self.sample_volume_trilinear(pos.x - delta, pos.y, pos.z)
        dy = self.sample_volume_trilinear(pos.x, pos.y + delta, pos.z) - self.sample_volume_trilinear(pos.x, pos.y - delta, pos.z)
        dz = self.sample_volume_trilinear(pos.x, pos.y, pos.z + delta) - self.sample_volume_trilinear(pos.x, pos.y, pos.z - delta)
        return tl.vec3(dx, dy, dz).normalized()

    @ti.func
    def apply_transfer_function(self, intensity: float):
        ''' Applies a 1D transfer function to a given intensity value

        Args:
            intensity (float): Intensity in [0,1]

        Returns:
            tl.vec4: Color and opacity for given `intensity`
        '''
        length = ti.static(float(self.tf_tex.shape[0] - 1) - 1e-4)
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
        light_pos = look_from + tl.vec3(0.0, 1.0, 0.0)
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
                tmin += 0.5 * ray_len / n_samples # Offset tmin as t_start
                color = tl.vec4(0.0) # Color to be composited
                cnt = 0
                while cnt < n_samples and color.w < 0.99: # Early ray termination
                    pos = look_from + tl.mix(tmin, tmax, float(cnt)/float(n_samples)) * vd # Current Pos
                    intensity = self.sample_volume_trilinear(pos.x, pos.y, pos.z)
                    sample_color = self.apply_transfer_function(intensity)
                    if sample_color.w > 1e-3:
                        normal = self.get_volume_normal(pos)
                        light_dir = (pos - light_pos).normalized() # Direction to light source
                        n_dot_l = max(normal.dot(light_dir), 0.0)
                        diffuse = self.diffuse * n_dot_l
                        r = tl.reflect(light_dir, normal) # Direction of reflected light
                        r_dot_v = max(r.dot(-vd), 0.0)
                        specular = self.specular * pow(r_dot_v, self.shininess)
                        shaded_color = tl.vec4((self.ambient + diffuse + specular) * sample_color.xyz * sample_color.w * self.light_color, sample_color.w)
                        color += (1.0 - color.w) * shaded_color # Composite color
                    cnt += 1
                # Save result to global buffers
                self.render[i, j] = color.xyz
            else: # Skip empty space
                self.render[i, j] = tl.vec3(0.0)

    @ti.kernel
    def draw_entry_points(self, cam_pos_x: float, cam_pos_y: float, cam_pos_z: float):
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
                self.render[i, j] = tl.vec3(look_from + tmin * vd)
            else:
                self.render[i, j] = tl.vec3(0.0)

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
    RESOLUTION = (640, 640)
    TF_RESOLUTION = 128
    # ti.init(arch=ti.cpu, debug=True, excepthook=True, log_level=ti.TRACE, print_ir=True)
    ti.init(arch=ti.gpu)
    gui = ti.GUI("Volume Raycaster", res=RESOLUTION, fast_gui=True)
    # Data
    tf = tex_from_pts(np.array([[0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                [0.3308, 0.1512, 0.6418, 0.8293, 0.0000],
                                [0.3508, 0.1512, 0.6418, 0.8293, 0.5465],
                                [0.3714, 0.1512, 0.6418, 0.8293, 0.7660],
                                [0.4297, 0.1512, 0.6418, 0.8293, 0.7660],
                                [0.4504, 0.1512, 0.6418, 0.8293, 0.5465],
                                [0.4704, 0.1512, 0.6418, 0.8293, 0.0000],
                                [1.0000, 0.0000, 0.0000, 0.0000, 0.0000]]), TF_RESOLUTION).permute(1, 0).contiguous().numpy().astype(np.float32)
    vol_ds = TorchDataset('/run/media/dome/Data/data/torchvtk/CQ500')
    vol = vol_ds[0]['vol'].permute(2, 0, 1).contiguous().numpy().astype(np.float32)
    # Renderer
    vr = VolumeRaycaster(volume_resolution=vol.shape, render_resolution=RESOLUTION, tf_resolution=TF_RESOLUTION)
    vr.set_volume(vol)
    vr.set_tf_tex(tf)
    t = np.pi * 1.5
    while gui.running:
        vr.raycast(*in_circles(t), 4.0)
        # vr.draw_entry_points(0.0, 0.1, 3.0)
        t += rotate_camera(gui)
        gui.set_image(vr.render)
        gui.show()
