

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
