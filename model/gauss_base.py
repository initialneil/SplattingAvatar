# The base class for GaussianSplatting Model.
# Contributer(s): Neil Z. Shao
# All rights reserved. Prometheus 2022-2024.
import os
import numpy as np
from pathlib import Path
from plyfile import PlyData, PlyElement
import torch
import torch.nn as nn
from utils.general_utils import strip_symmetric, build_scaling_rotation
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from utils.sh_utils import eval_sh, RGB2SH
from gaussian_renderer import render

def to_abs_path(fn, dir):
    if not os.path.isabs(fn):
        fn = os.path.join(dir, fn)
    return fn

def to_cache_path(dir):
    cache_dir = os.path.join(dir, 'cache')
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir

class GaussianBase(nn.Module):
    def __init__(self, sh_degree=0) -> None:
        super().__init__()
        self.active_sh_degree = sh_degree
        self.max_sh_degree = sh_degree
        self.setup_functions()

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize

    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self.get_rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1
    
    def init_gauss(self, xyz, features_dc, features_extra, opacities, scales, rots, 
                   init_params=False):
        if init_params:
            self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
            self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
            self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
            self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
            self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
            self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))
        else:
            self._xyz = torch.tensor(xyz, dtype=torch.float, device="cuda")
            self._features_dc = torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous()
            self._features_rest = torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous()
            self._opacity = torch.tensor(opacities, dtype=torch.float, device="cuda")
            self._scaling = torch.tensor(scales, dtype=torch.float, device="cuda")
            self._rotation = torch.tensor(rots, dtype=torch.float, device="cuda")

        self.max_radii2D = torch.zeros((self._opacity.shape[0]), device="cuda")
        self.active_sh_degree = self.max_sh_degree
        
    # render
    def render_to_camera(self, viewpoint_cam, pipe, background=None, scaling_modifer=1.0):
        if background == 'white':
            background = torch.tensor([1, 1, 1], dtype=torch.float32, device='cuda')
        elif background == 'black':
            background = torch.tensor([0, 0, 0], dtype=torch.float32, device='cuda')
        else:
            background = torch.rand((3,), dtype=torch.float32, device='cuda')

        out = render(viewpoint_cam, self, pipe, background, scaling_modifer)

        if hasattr(viewpoint_cam, 'original_image'):
            if hasattr(viewpoint_cam, 'gt_alpha_mask'):
                gt_alpha_mask = viewpoint_cam.gt_alpha_mask.cuda()
                gt_image = viewpoint_cam.original_image.cuda()
                gt_image = gt_image * gt_alpha_mask + background[:, None, None] * (1 - gt_alpha_mask)

                out.update({
                    'gt_image': gt_image,
                    'gt_alpha_mask': gt_alpha_mask,
                })
            else:
                gt_image = viewpoint_cam.original_image.cuda()
                out.update({
                    'gt_image': gt_image,
                })

        return out
    
    # save
    def construct_list_of_attributes(self, f_dc, f_rest, scale, rotation):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']

        for i in range(f_dc.shape[1]):
            l.append('f_dc_{}'.format(i))
        for i in range(f_rest.shape[1]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(scale.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(rotation.shape[1]):
            l.append('rot_{}'.format(i))

        return l
    
    # overwrite this function is needed
    # sh 0: f_rest_dims=0
    # sh 3: f_rest_dims=45
    def prepare_to_write(self, f_rest_dims=0):
        xyz = self.get_xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)

        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        if f_rest_dims > xyz.shape[0]:
            zero_dims = f_rest_dims - xyz.shape[0]
            f_rest = np.zeros((xyz.shape[0], zero_dims)).astype(np.float32)
        else:
            f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()

        scale = self._scaling.detach().cpu().numpy()
        rotation = self.get_rotation.detach().cpu().numpy()

        return {
            'xyz': xyz,
            'normals': normals,
            'f_dc': f_dc,
            'f_rest': f_rest,
            'opacities': opacities,
            'scale': scale,
            'rotation': rotation,
        }
    
    def save_ply(self, path, f_rest_dims=0):
        print(f'[3DGS] save_ply to {path}')
        os.makedirs(Path(path).parent, exist_ok=True)

        contents = self.prepare_to_write(f_rest_dims=f_rest_dims)
        xyz = contents['xyz']
        normals = contents['normals']
        f_dc = contents['f_dc']
        f_rest = contents['f_rest']
        opacities = contents['opacities']
        scale = contents['scale']
        rotation = contents['rotation']

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes(f_dc, f_rest, scale, rotation)]
        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    # load
    def load_ply(self, path, init_params=False):
        print(f'[3DGS] load_ply from {path}')
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])
        
        self.init_gauss(xyz, features_dc, features_extra, opacities, scales, rots, 
                        init_params=init_params)

    
