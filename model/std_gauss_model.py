import os
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from model import libcore
from utils.sh_utils import eval_sh, RGB2SH
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from .gauss_base import GaussianBase, to_abs_path, to_cache_path

# standard 3dgs
class StandardGaussModel(GaussianBase):
    def __init__(self, config,
                 device=torch.device('cuda'),
                 verbose=False):
        super().__init__()
        self.config = config
        self.device = device
        self.verbose = verbose

        self.register_buffer('_xyz', torch.Tensor(0))
        self.register_buffer('_features_dc', torch.Tensor(0))
        self.register_buffer('_features_rest', torch.Tensor(0))
        self.register_buffer('_scaling', torch.Tensor(0))
        self.register_buffer('_rotation', torch.Tensor(0))
        self.register_buffer('_opacity', torch.Tensor(0))

        if config is not None:
            self.setup_config(config)

    ##################################################
    @property
    def num_gauss(self):
        return self._xyz.shape[0]

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)

    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    def get_params(self, device='cpu'):
        return {
            '_xyz': self._xyz.detach().to(device),
            '_rotation': self._rotation.detach().to(device),
            '_scaling': self._scaling.detach().to(device),
            '_features_dc': self._features_dc.detach().to(device),
            '_features_rest': self._features_rest.detach().to(device),
            '_opacity': self._opacity.detach().to(device),
        }
    
    def set_params(self, params):
        if '_xyz' in params:
            self._xyz = params['_xyz'].to(self.device)
        if '_rotation' in params:
            self._rotation = params['_rotation'].to(self.device)
        if '_scaling' in params:
            self._scaling = params['_scaling'].to(self.device)
        if '_features_dc' in params:
            self._features_dc = params['_features_dc'].to(self.device)
        if '_features_rest' in params:
            self._features_rest = params['_features_rest'].to(self.device)
        if '_opacity' in params:
            self._opacity = params['_opacity'].to(self.device)
    
    def get_colors_precomp(self, viewpoint_camera=None):
        return self.color_activation(self._color)
    
    def get_colors_precomp(self, viewpoint_camera=None):
        shs_view = self.get_features.transpose(1, 2).view(-1, 3, (self.max_sh_degree+1)**2)
        if viewpoint_camera is not None:
            dir_pp = (self.get_xyz - viewpoint_camera.camera_center.repeat(self.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
        else:
            dir_pp_normalized = torch.zeros_like(self._xyz)
        sh2rgb = eval_sh(self.active_sh_degree, shs_view, dir_pp_normalized)
        colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        return colors_precomp

    ##################################################
    def setup_config(self, config):
        self.config = config
        self.max_sh_degree = config.get('sh_degree', 0)

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = fused_point_cloud
        self._features_dc = features[:,:,0:1].transpose(1, 2).contiguous()
        self._features_rest = features[:,:,1:].transpose(1, 2).contiguous()
        self._scaling = scales
        self._rotation = rots
        self._opacity = opacities
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self.active_sh_degree = self.max_sh_degree
    
    ##################################################
    def prune_points(self, valid_points_mask, optimizable_tensors):
        self._xyz = optimizable_tensors['_xyz']

        if '_scaling' in optimizable_tensors:
            self._scaling = optimizable_tensors['_scaling']
        else:
            self._scaling = self._scaling[valid_points_mask]

        if '_rotation' in optimizable_tensors:
            self._rotation = optimizable_tensors['_rotation']
        else:
            self._rotation = self._rotation[valid_points_mask]

        self._opacity = optimizable_tensors.get('_opacity', self._opacity)
        self._features_dc = optimizable_tensors.get('_features_dc', self._features_dc)
        self._features_rest = optimizable_tensors.get('_features_rest', self._features_rest)

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def densification_postfix(self, optimizable_tensors, densify_out):
        self._xyz = optimizable_tensors.get('_xyz', self._xyz)

        if '_scaling' in optimizable_tensors:
            self._scaling = optimizable_tensors['_scaling']
        else:
            self._scaling = torch.cat([self._scaling, densify_out['new_scaling']], dim=0)

        if '_rotation' in optimizable_tensors:
            self._rotation = optimizable_tensors['_rotation']
        else:
            self._rotation = torch.cat([self._rotation, densify_out['new_rotation']], dim=0)

        self._opacity = optimizable_tensors.get('_opacity', self._opacity)
        self._features_dc = optimizable_tensors.get('_features_dc', self._features_dc)
        self._features_rest = optimizable_tensors.get('_features_rest', self._features_rest)

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device='cuda')
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device='cuda')
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device='cuda')

    def prepare_densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device='cuda')
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)

        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense * scene_extent)
    
        if self.config.get('force_scaling_split', False):
            aspect_mask = (torch.max(self.get_scaling, dim=-1).values / self.get_scaling.mean(dim=-1)) > 2.0
            force_mask = torch.max(self.get_scaling, dim=-1).values > self.percent_dense * scene_extent * 1
            force_mask = torch.logical_and(force_mask, aspect_mask)
            selected_pts_mask = torch.logical_or(selected_pts_mask, force_mask)

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means = torch.zeros((stds.size(0), 3),device='cuda')
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)

        return selected_pts_mask, new_xyz
 
    def prepare_split_selected_to_new_xyz(self, selected_pts_mask, new_xyz, N):
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)

        splitout = {
            'new_xyz': new_xyz,
            'new_scaling': new_scaling,
            'new_rotation': new_rotation,
        }
    
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)
        splitout.update({
            'new_features_dc': new_features_dc,
            'new_features_rest': new_features_rest,
            'new_opacity': new_opacity,
        })

        return splitout

    def prepare_densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        
        new_xyz = self._xyz[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        cloneout = {
            'new_xyz': new_xyz,
            'new_scaling': new_scaling,
            'new_rotation': new_rotation,
        }

        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacity = self._opacity[selected_pts_mask]
        cloneout.update({
            'new_features_dc': new_features_dc,
            'new_features_rest': new_features_rest,
            'new_opacity': new_opacity,
        })

        return cloneout

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1


    