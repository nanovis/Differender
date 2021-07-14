import torch
from torch.autograd import gradcheck
from torchvtk.datasets import TorchDataset
import matplotlib.pyplot as plt

from transfer_functions import get_tf
from pytorch_module import Raycaster

from torchvision.utils import save_image

tf = torch.from_numpy(get_tf('gray', 128)).permute(1,0).float().contiguous()
vol_ds = TorchDataset('/run/media/dome/Data/data/torchvtk/CQ500_128')
vol = vol_ds[0]['vol'].float()
look_from = torch.tensor([0.0, 1.0, -2.5])

r = Raycaster((128,128,128), (240,240), 128, jitter=False)


print(f'{vol.shape=}, {vol.min()=}, {vol.max()=}')
print(f'{tf.shape=}, {tf.min()=}, {tf.max()=}')


if __name__ == '__main__':
    vol = vol.to('cuda').float().requires_grad_(True)
    tf = tf.to('cuda').float().requires_grad_(False)
    look_from = look_from.to('cuda').float().requires_grad_(False)
    target = torch.load('target.pt').float().to('cuda')
    test = gradcheck(r, [vol, tf, look_from], eps=1e-3, atol=1e-3, check_forward_ad=False, fast_mode=True, raise_exception=True, nondet_tol=1e-2)
    print(test)
    opt = torch.optim.Adam([vol])
    for i in range(100):
        opt.zero_grad()
        res = r(vol, tf, look_from)
        loss = torch.nn.functional.mse_loss(res, target)
        loss.backward()
        save_image(res, f'test/render_{i:03d}.png')
        print(f'Step {i:03d} ----------------------------------')
        print(f'{loss=}\n{vol.grad.abs().max()=}')
        opt.step()
        # with torch.no_grad():
        #     tf = torch.clamp(tf, 0.0, 1.0)
