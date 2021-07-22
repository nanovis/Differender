import numpy as np

def identity(): return np.eye(4)

def look_at(look_from=(0,0,3), look_to=(0,0,0), look_up=(0,1,0)):
    look_from = np.array(look_from, dtype=float)
    look_to   = np.array(look_to,   dtype=float)
    look_up   = np.array(look_up,   dtype=float)

    view_dir = np.linalg.norm(look_to - look_from)
    right    = np.linalg.norm(np.cross(view_dir, np.linalg.norm(up)))
    up       = np.linalg.norm(np.cross(right, view_dir))

    mat = identity()
    mat[0, :3] = right
    mat[1, :3] = up
    mat[2, :3] = -view_dir
    mat[:3, 3] = look_to - look_from
    return np.stack([right, up, -view_dir])

def perspective_projection(near=0.1, far=100, fov=60, aspect=1):
    q = 1 / np.tan(np.radians(fov) * 0.5)
    a = q / aspect
    b = -far / (far-near)
    c = - (far*near)/(far-near)

    proj = np.array([[a, 0, 0, 0],
                     [0, q, 0, 0],
                     [0, 0, b,-1],
                     [0, 0, c, 0]], dtype=float)
    invp = np.array([[1/a, 0, 0, 0],
                     [0, 1/q, 0, 0],
                     [0, 0, 0, 1/c],
                     [0, 0,-1, b/c]], dtype=float)
    return proj, invp

def compute_camera_distance(obj_radius, fov_rad):
    return obj_radius * (np.sin(fov_rad) + np.cos(fov_rad) * np.arctan(fov_rad))
