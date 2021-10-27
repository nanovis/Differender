import torch
import torch.nn.functional as F
import math
from itertools import count
from torchvtk.datasets import TorchDataset
from torchvtk.rendering import plot_comp_render_tf
from torchvtk.utils import pool_map, make_4d
import matplotlib.pyplot as plt

from differender.utils import get_tf, in_circles, get_rand_pos
from differender.volume_raycaster import Raycaster

from torchvision.utils import save_image, make_grid
from pytorch_msssim import ssim as ssim2d

from ranger import Ranger
from taichi import print_profile_info, print_kernel_profile_info, clear_kernel_profile_info


def fig_to_img(fig):
    fig.set_tight_layout(True)
    fig.set_dpi(100)
    fig.canvas.draw()
    w, h = fig.get_size_inches() * fig.get_dpi()
    w, h = int(w), int(h)
    im = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(
        (h, w, 3))
    plt.close(fig)
    return im



if __name__ == '__main__':
    TF_RES = 128
    BS = 1
    IM_RES = 256
    ITERATIONS = 200
    tf = get_tf('tf1', TF_RES)
    tf_gt = get_tf('tf1', TF_RES).to('cuda').expand(BS, -1,-1).float()
    vol_ds = TorchDataset('/run/media/dome/Data/data/torchvtk/CQ500_128')
    # vol = make_4d(vol_ds[0]['vol'].float())
    # vol = torch.rand(1,256,256,256)
    vol_gt = make_4d(vol_ds[1]['vol'].float()).to('cuda')
    vol = vol_gt.clone()
    print(f'{vol.shape=}, {vol.min()=}, {vol.max()=}')
    print(f'{tf.shape=}, {tf.min()=}, {tf.max()=}')

    raycast = Raycaster(vol.shape[-3:], (IM_RES,IM_RES), TF_RES, jitter=False, max_samples=1024, ti_kwargs={
        'debug': True, 
        'excepthook': True, 
        'device_memory_fraction': 0.5,
        # 'kernel_profiler': True
        })


    vol = vol.to('cuda').float().requires_grad_(True)
    tf = tf.to('cuda').float().requires_grad_(True)
    # lf_gt = in_circles(0.0).float().to('cuda')
    lf = torch.stack([in_circles(t) for t in [-0.0]]).float().to('cuda').requires_grad_(True)
    # lf = (lf_gt.expand(BS,3).clone() + torch.randn(BS, 3, device='cuda', dtype=torch.float32)).requires_grad_(True)
    lf_gt = lf.detach()[0].clone()
    opt = torch.optim.AdamW([lf], weight_decay=0, lr=1e-2)
    # sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=5e-2, total_steps=ITERATIONS)
    # sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, 50)
    pred_images = []
    targ_images = []
    pred_tfs = []
    log_strs = []
    histograms = []
    try:
        for i in range(ITERATIONS):
            # lf = in_circles(0.1* i).to(vol.dtype).to(vol.device)
            with torch.no_grad():
                gt = raycast.raycast_nondiff(vol_gt.detach(), tf_gt.detach(), lf_gt, sampling_rate=1.0)
            opt.zero_grad()
            res = raycast(vol, tf, lf)
            dssim_loss = 1.0 - ssim2d(res, gt, data_range=1.0, size_average=True, nonnegative_ssim=True)
            mse_loss = F.mse_loss(res, gt)
            loss = torch.nan_to_num(dssim_loss) + mse_loss
            loss.backward()
            # Log
            # print_kernel_profile_info('trace')
            # clear_kernel_profile_info()
            pred_images.append(torch.clamp(res.detach()[0], 0.0, 1.0).cpu())
            targ_images.append(torch.clamp(gt.detach()[0], 0.0, 1.0).cpu())
            pred_tfs.append(tf.detach().cpu())
            with torch.cuda.amp.autocast(False):
                histograms.append(torch.histc(torch.clamp(vol.detach().float(), 0.0, 1.0), bins=128, min=0.0, max=1.0).cpu())
            with torch.no_grad():
                angle_diff = torch.rad2deg(torch.acos(torch.dot(F.normalize(lf_gt, dim=0), F.normalize(lf[0], dim=0))))
                dist_diff = (torch.linalg.norm(lf_gt, dim=0) - torch.linalg.norm(lf[0], dim=0))
            log_str = f'Step {i:03d}:   Loss: {loss.detach().item():0.3f}   SSIM: {1.0 - dssim_loss.detach().item():0.3f}   MSE: {mse_loss.detach().item():0.5f}   Angle/Dist: {angle_diff.detach().item():.1f}, {dist_diff.detach().item():.1f}   Grad: {lf.grad.squeeze()}'
            log_strs.append(log_str)
            print(log_str)
            opt.step()
            # sched.step()
            with torch.no_grad():
                tf.clamp_(0.0, 1.0)
                vol.clamp_(0.0, 1.0)

    except KeyboardInterrupt:
        print(f'Ctrl+C stopped after {len(log_strs) +1} iterations. Saving logs now.')
    finally:
        print('-'*40)
        # print_profile_info()

    targ_tf = tf_gt.detach()[0].cpu()
    def save_comparison_fig(tup):
        i, pred_im, targ_im, pred_tf, log_str, hist = tup
        diff_im = (targ_im - pred_im).abs()[:3] * 5.0
        fig = plot_comp_render_tf([(pred_im, pred_tf, 'Prediction'),
                                   (targ_im, targ_tf, 'Target'),
                                   (diff_im, targ_tf, 'Difference x5')])
        fig.suptitle(log_str, fontsize=16)
        fig.savefig(f'results/ptmod/comparison_plot_{i:03d}.png', dpi=100)
        fig.clear()
        plt.close(fig)
        f, ax = plt.subplots()
        ax.bar(torch.arange(128), hist)
        # f.savefig(f'results/ptmod/hist_{i:03d}.png', dpi=200)


    pool_map(save_comparison_fig,
             zip(count(), pred_images, targ_images, pred_tfs, log_strs, histograms), num_workers=0, dlen=len(pred_images), title='Saving Comparison Plots')

    # for i, pred_im, targ_im, pred_tf in zip(count(), pred_images, targ_images, pred_tfs):
    #     fig = plot_comp_render_tf([(pred_im, pred_tf, 'Prediction'), (targ_im, targ_tf, 'Target')])
    #     fig.savefig(f'results/ptmod/comparison_plot_{i:03d}.png', dpi=100)
