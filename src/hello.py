import taichi as ti

ti.init(arch=ti.cuda)
n = 320
pixels = ti.field(dtype=float, shape=(n * 2, n))


@ti.func  # __device__
def complex_sqr(z):
    return ti.Vector([z[0] ** 2 - z[1] ** 2, z[0] * z[1] * 2])


@ti.kernel  # __global__
def paint(t: float):  # must be type-hinted
    for i, j in pixels:
        c = ti.Vector([-0.8, ti.cos(t) * 0.2])
        z = ti.Vector([i / n - 1, j / n - 0.5])
        iterations = 0
        while z.norm() < 20 and iterations < 50:
            z = complex_sqr(z) + c
            iterations += 1
        pixels[i, j] = 1 - iterations * 0.02


gui = ti.GUI("Julia Set", res=(n * 2, n))

for i in range(600):
    paint(i * 0.03)
    gui.set_image(pixels)
    gui.show()
