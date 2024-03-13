# The base class for GaussianSplatting Loss.
# Contributer(s): Neil Z. Shao
# All rights reserved. Prometheus 2022-2024.
import os
import cv2
import numpy as np
import json
import torch
import torch.nn.functional as thf
from utils.loss_utils import l1_loss, ssim, LPIPS
from utils.image_utils import psnr
from utils.metrics import img_mse, img_ssim, img_psnr, perceptual
from dataset.dataset_helper import make_dataloader
from gaussian_renderer import network_gui
from tqdm import tqdm

class LossBase(torch.nn.Module):
    def __init__(self, gs_model, optimizer_config) -> None:
        super().__init__()
        self.gs_model = gs_model
        self.optimizer_config = optimizer_config
        self.lpips = LPIPS(eval=False).cuda()

    def collect_loss(self, gt_image, image, gt_alpha_mask=None):
        Ll1 = l1_loss(image, gt_image)

        if self.optimizer_config.get('lambda_ssim', 0) > 0:
            Lssim = 1.0 - ssim(image, gt_image)
            loss = (1.0 - self.optimizer_config.lambda_ssim) * Ll1 + self.optimizer_config.lambda_ssim * Lssim
        else:
            loss = Ll1

        if self.optimizer_config.get('lambda_rgb_mse', 0) > 0:
            Ll2 = thf.mse_loss(image, gt_image)
            loss += self.optimizer_config.lambda_rgb_mse * Ll2

        if self.optimizer_config.get('lambda_perceptual', 0) > 0:
            if gt_alpha_mask is not None:
                Llpips = self.lpips(gt_image * gt_alpha_mask, image * gt_alpha_mask).squeeze()
            else:
                Llpips = self.lpips(gt_image, image).squeeze()
            loss += self.optimizer_config.lambda_perceptual * Llpips

        if self.optimizer_config.get('lambda_sparsity', 0) > 0:
            loss += self.optimizer_config.lambda_sparsity * self.gs_model.get_opacity.mean()

        if self.optimizer_config.get('lambda_scaling', 0) > 0:
            thresh_scaling_max = self.optimizer_config.get('thresh_scaling_max', 0.008)
            thresh_scaling_ratio = self.optimizer_config.get('thresh_scaling_ratio', 10.0)
            max_vals = self.gs_model.get_scaling.max(dim=-1).values
            min_vals = self.gs_model.get_scaling.min(dim=-1).values
            ratio = max_vals / min_vals
            thresh_idxs = (max_vals > thresh_scaling_max) & (ratio > thresh_scaling_ratio)
            if thresh_idxs.sum() > 0:
                loss += self.optimizer_config.lambda_scaling * max_vals[thresh_idxs].mean()

        psnr_full = psnr(image, gt_image).mean().float().item()

        return {
            'loss': loss,
            'psnr_full': psnr_full,
        }

########## testing routine ##########
def visualize_compare(gt_image, image, psnr, ssim, lpips):
    compare = torch.concat([gt_image, image], dim=2)
    compare = (compare.permute([1, 2, 0]) * 255)[:, :, [2, 1, 0]].detach().cpu().numpy()
    compare = cv2.putText(compare, f'psnr/ssim/lpips', (20, compare.shape[0] - 50), 0, 1, (0, 0, 255))
    compare = cv2.putText(compare, f'{psnr:.4f}/{ssim:.4f}/{lpips:.4f}', 
                            (20, compare.shape[0] - 10), 0, 1, (0, 0, 255))
    
    err = (image - gt_image).abs().max(dim=0)[0].clip(0, 1)     
    from model import libcore       
    err_map = libcore.colorizeWeightsMap(err.detach().cpu().numpy(), min_val=0, max_val=1)
    compare = np.concatenate([compare, err_map], axis=1)
    return compare

# tensor to image
def write_tensor_image(fn, tensor, rgb2bgr=False):
    if len(tensor.shape) == 3:
        if tensor.shape[0] == 3 or tensor.shape[0] == 4:
            tensor = tensor.permute([1, 2, 0])

    if rgb2bgr:
        if tensor.shape[2] == 3:
            tensor = tensor[:, :, [2, 1, 0]]
        else:
            tensor = tensor[:, :, [2, 1, 0, 3]]
    
    cv2.imwrite(fn, (tensor.clamp(0, 1) * 255).detach().cpu().numpy().astype(np.uint8))

# testing routine
def testing_routine(pipe, frameset, gs_model, render_dir=None, compare_dir=None, verify=None):
    if render_dir is not None:
        os.makedirs(render_dir, exist_ok=True)
    if compare_dir is not None:
        os.makedirs(compare_dir, exist_ok=True)

    psnr_full = 0
    ssim_full = 0
    lpips_full = 0
    count = 0

    dataloader = make_dataloader(frameset, shuffle=False)
    data_iterator = iter(dataloader)

    with torch.no_grad():
        num_frames = len(frameset)
        pbar = tqdm(range(num_frames))
        for idx in pbar:
            batch = next(data_iterator)[0]
            frm_idx = batch['frm_idx']
            scene_cameras = batch['scene_cameras']

            # update to current posed mesh
            gs_model.update_to_posed_mesh(batch['mesh_info'])
                
            # there should be only one camera
            viewpoint_cam = scene_cameras[0].cuda()

            # render
            render_pkg = gs_model.render_to_camera(viewpoint_cam, pipe, background='white')
            image = render_pkg['render']
            gt_image = render_pkg['gt_image']

            _rmse = img_mse(image[None, ...], gt_image[None, ...], mask=None, error_type='rmse', use_mask=False)
            _ssim = img_ssim(image[None, ...], gt_image[None, ...])
            _psnr = img_psnr(image[None, ...], gt_image[None, ...], rmse=_rmse)
            _lpips = perceptual(image[None, ...], gt_image[None, ...], mask=None, use_mask=False)

            #############
            if verify is not None:
                network_gui.send_image_to_network(image, verify)
        
            #############
            if render_dir is not None:
                write_tensor_image(os.path.join(render_dir, f'{frm_idx:05d}.png'), image, rgb2bgr=True)
            if compare_dir is not None:
                compare = visualize_compare(gt_image, image, _psnr.item(), _ssim.item(), _lpips.item())
                cv2.imwrite(os.path.join(compare_dir, f'{frm_idx:05d}.jpg'), compare)
            
                # err_map = cv2.putText(err_map, f'psnr/ssim/lpips', (20, err_map.shape[0] - 50), 0, 1, (255, 255, 255))
                # err_map = cv2.putText(err_map, f'{_psnr.item():.4f}/{_ssim.item():.4f}/{_lpips.item():.4f}', 
                #                       (20, err_map.shape[0] - 10), 0, 1, (255, 255, 255))
                # cv2.imwrite(os.path.join(err_dir, f'{frm_idx:05d}.jpg'), err_map)
            #############

            psnr_full += _psnr.item()
            ssim_full += _ssim.item()
            lpips_full += _lpips.item()
            count += 1

            pbar.set_postfix({
                'psnr': f'{(psnr_full / count):.4f}({_psnr.item():.4f})',
                'ssim': f'{(ssim_full / count):.4f}({_ssim.item():.4f})',
                'lpips': f'{(lpips_full / count):.4f}({_lpips.item():.4f})',
            })

    return {
        'psnr': psnr_full / count,
        'ssim': ssim_full / count,
        'lpips': lpips_full / count,
        'n_gauss': gs_model._xyz.shape[0],
    }

def run_testing(pipe, frameset_test, gs_model, model_path=None, iteration=None, verify=None):
    if model_path is not None:
        if iteration is not None:
            render_dir = os.path.join(model_path, f'eval_{iteration}/render')
            compare_dir = os.path.join(model_path, f'eval_{iteration}/compare')
            stats_fn = os.path.join(model_path, f'eval_{iteration}/stats.json')
        else:
            render_dir = os.path.join(model_path, f'eval/render')
            compare_dir = os.path.join(model_path, f'eval/compare')
            stats_fn = os.path.join(model_path, f'eval/stats.json')
    else:
        stats_fn = None

    stats = testing_routine(pipe, frameset_test, gs_model,
                            render_dir=render_dir, 
                            compare_dir=compare_dir,
                            verify=verify)
    
    if stats_fn is not None:
        with open(stats_fn, 'w') as fp:
            json.dump(stats, fp)



