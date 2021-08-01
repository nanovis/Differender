import torch
from torch.autograd import gradcheck
from torchvtk.datasets import TorchDataset
import matplotlib.pyplot as plt

from transfer_functions import get_tf
from pytorch_module import Raycaster

from torchvision.utils import save_image

tf = torch.from_numpy(get_tf('tf1', 128)).permute(1,0).float().contiguous()
vol_ds = TorchDataset('/run/media/dome/Data/data/torchvtk/CQ500_128')
vol = vol_ds[0]['vol'].float()
look_from = torch.tensor([0.0, 1.0, -2.5])

r = Raycaster((128,128,128), (240,240), 128, jitter=False)


print(f'{vol.shape=}, {vol.min()=}, {vol.max()=}')
print(f'{tf.shape=}, {tf.min()=}, {tf.max()=}')


if __name__ == '__main__':
    # vol += torch.randn_like(vol) * 0.1
    vol = vol.to('cuda').float().requires_grad_(True)
    tf = tf.to('cuda').float().requires_grad_(True)
    look_from = look_from.to('cuda').float().requires_grad_(False)
    target = torch.load('target.pt').float().to('cuda')
    # test = gradcheck(r, [vol, tf, look_from], eps=1e-3, atol=1e-3, check_forward_ad=False, fast_mode=True, raise_exception=True, nondet_tol=1e-2)
    # print(test)
    res_single  = r(vol, tf, look_from)
    print(f'{res_single.shape=}')
    res_batched = r(torch.stack([vol]*2), torch.stack([tf]*2), torch.stack([look_from, look_from + 1.0]))
    print(f'{res_batched.shape=}')
    print(f'Equal?{(res_single == res_batched.squeeze(0)).all()}')
    # save_image(res_single, 'res_single.png')
    # save_image(res_batched[0], 'res_batched0.png')
    # save_image(res_batched[1],'res_batched1.png')

    opt = torch.optim.Adam([vol, tf])
    opt.zero_grad()
    res = r(vol, tf, look_from)
    loss = torch.nn.functional.mse_loss(res, target)
    loss.backward()
    print(f'SINGLE ----------------------------------')
    print(f'{loss=}')
    print(f'{vol.grad.shape=},  {vol.grad.abs().max()=}')
    print(f'{tf.grad.shape=},  {tf.grad.abs().max()=}')
    vol_grad = vol.grad.clone()
    tf_grad = tf.grad.clone()

    opt.zero_grad()
    res = r(torch.stack([vol]*4), torch.stack([tf]*4), torch.stack([look_from]*4))
    loss = torch.nn.functional.mse_loss(res, torch.stack([target]*4))
    loss.backward()
    print(f'BATCHED ----------------------------------')
    print(f'{loss=}')
    print(f'{vol.grad.shape=},  {vol.grad.abs().max()=}')
    print(f'{tf.grad.shape=},  {tf.grad.abs().max()=}') 

    print(f'COMPARE ----------------------------------')
    print(f'Vol Gradients equal? {torch.allclose(vol.grad, vol_grad, atol=1e-5, rtol=1e-3)}')
    print(f'TF Gradients equal? {torch.allclose(tf.grad, tf_grad, atol=1e-5, rtol=1e-3)}')
    # opt = torch.optim.Adam([vol, tf])
    # for i in range(100):
    #     opt.zero_grad()
    #     res = r(vol, tf, look_from)
    #     loss = torch.nn.functional.mse_loss(res, target)
    #     loss.backward()
    #     save_image(res, f'test/render_{i:03d}.png')
    #     print(f'Step {i:03d} ----------------------------------')
    #     print(f'{loss=}')
    #     print(f'{vol.grad.abs().max()=}')
    #     print(f'{tf.grad.abs().max()=}')
    #     opt.step()
