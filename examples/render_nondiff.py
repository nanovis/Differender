import torch
import torch.nn.functional as F
import math
from itertools import count
from torchvtk.datasets import TorchDataset
from torchvtk.rendering import plot_comp_render_tf
from torchvtk.utils import pool_map, make_4d
import matplotlib.pyplot as plt

from differender.utils import get_tf
from differender import Raycaster

from torchvision.utils import save_image


def in_circles(i, y=0.7, dist=2.5):
    x = math.cos(i) * dist
    z = math.sin(i) * dist
    return torch.tensor([x, y, z], dtype=torch.float32)


if __name__ == '__main__':
    vol_ds = TorchDataset('/run/media/dome/Data/data/torchvtk/CQ500')
    vol = vol_ds[0]['vol'].float()
    tf = torch.from_numpy(get_tf('tf1', 128)).permute(1,0).float().contiguous()
    raycaster = Raycaster(vol.shape[-3:], (800, 800), 128, jitter=False, max_samples=1)

    vol = vol.to('cuda').requires_grad_(True)
    tf = tf.to('cuda').requires_grad_(True)
    lf = in_circles(1.7 * math.pi).float().to('cuda')

    print(vol.shape, raycaster.volume_shape, tf.shape, lf)
    im = raycaster.raycast_nondiff(vol[None], tf[None], lf[None], sampling_rate=16.0)

    save_image(im, 'nondiff_render.png')
