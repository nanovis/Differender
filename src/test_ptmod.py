import torch
from torchvtk.datasets import TorchDataset
import matplotlib.pyplot as plt

from transfer_functions import get_tf
from pytorch_module import Raycaster

tf = torch.from_numpy(get_tf('gray', 128)).permute(1,0).float().contiguous()
vol_ds = TorchDataset('/run/media/dome/Data/data/torchvtk/CQ500_256')
vol = vol_ds[0]['vol'].float()
look_from = torch.tensor([0.0, 1.0, -2.5])

r = Raycaster((256,256,256), (240,240), 128)


print(f'{vol.shape=}, {vol.min()=}, {vol.max()=}')
print(f'{tf.shape=}, {tf.min()=}, {tf.max()=}')


if __name__ == '__main__':
    target = torch.load('target.pt')
    res = r.forward(vol.requires_grad_(True), tf.requires_grad_(True), look_from.requires_grad_(True))
    loss = torch.nn.functional.mse_loss(res, target)
    print(f'{loss=}')
    loss.backward()
    print(tf.grad)
