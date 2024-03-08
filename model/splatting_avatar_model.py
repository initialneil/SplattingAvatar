import os
import torch
import torch.nn.functional as thf
import numpy as np
from pathlib import Path
import json
from model import libcore
from simple_phongsurf import PhongSurfacePy3d
from utils.sh_utils import eval_sh, RGB2SH
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from utils.data_utils import sample_bary_on_triangles, retrieve_verts_barycentric
from utils.map import PerVertQuaternion
from utils.graphics_utils import BasicPointCloud
from simple_knn._C import distCUDA2
from pytorch3d.transforms import quaternion_multiply
from .gauss_base import GaussianBase, to_abs_path, to_cache_path

# standard 3dgs
class SplattingAvatarModel(GaussianBase):
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

        # for splatting avatar
        self.register_buffer('sample_fidxs', torch.Tensor(0))
        self.register_buffer('sample_bary', torch.Tensor(0))

        if config is not None:
            self.setup_config(config)

    ##################################################
    @property
    def num_gauss(self):
        return self._xyz.shape[0]

    @property
    def get_xyz_cano(self):
        if self.config.xyz_as_uvd:
            # uv -> self.sample_bary -> self.base_normal --(d)--> xyz
            xyz = self.base_normal_cano * self._xyz[..., -1:]
            return self.base_xyz_cano + xyz
        else:
            return self._xyz

    @property
    def get_xyz(self):
        if self.config.xyz_as_uvd:
            # uv -> self.sample_bary -> self.base_normal --(d)--> xyz
            xyz = self.base_normal * self._xyz[..., -1:]
            return self.base_xyz + xyz
        else:
            return self._xyz

    @property
    def base_normal_cano(self):
        return thf.normalize(retrieve_verts_barycentric(self.cano_norms, self.cano_faces, 
                                                        self.sample_fidxs, self.sample_bary), 
                                                        dim=-1)

    @property
    def base_normal(self):
        return thf.normalize(retrieve_verts_barycentric(self.mesh_norms, self.cano_faces, 
                                                        self.sample_fidxs, self.sample_bary), 
                                                        dim=-1)
    @property
    def base_xyz_cano(self):
        return retrieve_verts_barycentric(self.cano_verts, self.cano_faces, 
                                          self.sample_fidxs, self.sample_bary)
    @property
    def base_xyz(self):
        return retrieve_verts_barycentric(self.mesh_verts, self.cano_faces, 
                                          self.sample_fidxs, self.sample_bary)

    @property
    def get_rotation_cano(self):
        return self.rotation_activation(self._rotation)
        
    @property
    def get_rotation(self):
        return self.rotation_activation(quaternion_multiply(self.base_quat, self._rotation))
    
    @property
    def get_rotation_embed(self):
        return self.rotation_activation(self.base_quat)
    
    @property
    def base_quat(self):
        return torch.einsum('bij,bi->bj', self.tri_quats[self.sample_fidxs], self.sample_bary)
    
    @property
    def get_scaling_cano(self):
        return self.scaling_activation(self._scaling)

    @property
    def get_scaling(self):
        scaling_alter = self._face_scaling[self.sample_fidxs]
        return self.scaling_activation(self._scaling * scaling_alter)

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

        # use _xyz as variables for uvd
        # enabling uvd representation of SplattingAvatar
        self.config.xyz_as_uvd = self.config.get('xyz_as_uvd', True)

    def create_from_pcd(self, pcd : BasicPointCloud):
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().to(self.device)
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().to(self.device))
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().to(self.device)
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().to(self.device)), 0.0000001)
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
    
    def setup_canonical(self, cano_verts, cano_norms, cano_faces):
        self.cano_verts = cano_verts
        self.cano_norms = cano_norms
        self.cano_faces = cano_faces

        # quaternion from cano to pose
        self.quat_helper = PerVertQuaternion(cano_verts, cano_faces).to(self.device)

        # phong surface for triangle walk
        self.phongsurf = PhongSurfacePy3d(cano_verts, cano_faces, cano_norms,
                                          outer_loop=2, inner_loop=50, method='uvd').to(self.device)

    def create_from_canonical(self, cano_mesh):
        cano_verts = cano_mesh['mesh_verts'].float().to(self.device)
        cano_norms = cano_mesh['mesh_norms'].float().to(self.device)
        cano_faces = cano_mesh['mesh_faces'].long().to(self.device)
        self.setup_canonical(cano_verts, cano_norms, cano_faces)
        self.mesh_verts = self.cano_verts
        self.mesh_norms = self.cano_norms
        
        # sample on mesh
        num_samples = self.config.get('num_init_samples', 10000)
        sample_fidxs, sample_bary = sample_bary_on_triangles(cano_faces.shape[0], num_samples)
        self.sample_fidxs = sample_fidxs.to(self.device)
        self.sample_bary = sample_bary.to(self.device)

        sample_verts = retrieve_verts_barycentric(cano_verts, cano_faces, self.sample_fidxs, self.sample_bary)
        sample_norms = retrieve_verts_barycentric(cano_norms, cano_faces, self.sample_fidxs, self.sample_bary)
        sample_norms = thf.normalize(sample_norms, dim=-1)

        pcd = BasicPointCloud(points=sample_verts.detach().cpu().numpy(),
                              normals=sample_norms.detach().cpu().numpy(),
                              colors=torch.full_like(sample_verts, 0.5).float().cpu())
        self.create_from_pcd(pcd)

        # use _xyz as uvd
        if self.config.xyz_as_uvd:
            self._xyz = torch.zeros_like(self._xyz)

    def update_to_posed_mesh(self, posed_mesh=None):
        if posed_mesh is not None:
            self.mesh_verts = posed_mesh['mesh_verts'].float().to(self.device)
            self.mesh_norms = posed_mesh['mesh_norms'].float().to(self.device)

            self.per_vert_quat = self.quat_helper(self.mesh_verts)
            self.tri_quats = self.per_vert_quat[self.cano_faces]

        self._face_scaling = self.quat_helper.calc_face_area_change(self.mesh_verts)

    def update_to_cano_mesh(self):
        cano = {
            'mesh_verts': self.cano_verts,
            'mesh_norms': self.cano_norms,
        }
        self.update_to_posed_mesh(cano)

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

        if self.config.xyz_as_uvd:
            self.sample_fidxs = self.sample_fidxs[valid_points_mask]
            self.sample_bary = self.sample_bary[valid_points_mask]

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

        # mesh embedding
        if self.config.xyz_as_uvd:
            self.sample_fidxs = torch.cat([self.sample_fidxs, densify_out['new_sample_fidxs']], dim=0)
            self.sample_bary = torch.cat([self.sample_bary, densify_out['new_sample_bary']], dim=0)

        # stats
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device='cuda')
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device='cuda')
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device='cuda')

    def prepare_densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device='cuda')
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)

        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(self.get_scaling_cano, dim=1).values > self.percent_dense * scene_extent)

        stds = self.get_scaling_cano[selected_pts_mask].repeat(N,1)
        means = torch.zeros((stds.size(0), 3),device='cuda')
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self.get_rotation_cano[selected_pts_mask]).repeat(N,1,1)
        
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz_cano[selected_pts_mask].repeat(N, 1)
        return selected_pts_mask, new_xyz.detach()
 
    def prepare_split_selected_to_new_xyz(self, selected_pts_mask, new_xyz, N):
        new_scaling = self.scaling_inverse_activation(
            self.get_scaling_cano[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)

        splitout = {
            'new_xyz': new_xyz,
            'new_scaling': new_scaling,
            'new_rotation': new_rotation,
        }

        # fit uvd to new_xyz
        if self.config.xyz_as_uvd:
            # find new embedding point
            fidx = self.sample_fidxs[selected_pts_mask].repeat(N)
            uv = self.sample_bary[selected_pts_mask, :2].repeat(N, 1)
            d = self._xyz[selected_pts_mask, -1:].repeat(N, 1)

            if not self.config.get('skip_triangle_walk', False):
                fidx, uv = self.phongsurf.update_corres_spt(new_xyz, None, fidx, uv)

            bary = torch.concat([uv, 1.0 - uv[:, 0:1] - uv[:, 1:2]], dim=-1)
            new_xyz = torch.concat([torch.zeros_like(uv), d], dim=-1)
        
            splitout.update({
                'new_xyz': new_xyz,
                'new_sample_fidxs': fidx,
                'new_sample_bary': bary,
            })
    
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

        if self.config.xyz_as_uvd:
            cloneout.update({
                'new_sample_fidxs': self.sample_fidxs[selected_pts_mask],
                'new_sample_bary': self.sample_bary[selected_pts_mask],
            })

        return cloneout

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1

    ########################################
    def walking_on_triangles(self):
        if self.config.get('skip_triangle_walk', False):
            return

        fidx = self.sample_fidxs.detach().cpu().numpy().astype(np.int32)
        uv = self.sample_bary[..., :2].detach().cpu().numpy().astype(np.double)
        delta = self._xyz[..., :2].detach().cpu().numpy().astype(np.double)
        fidx, uv = self.phongsurf.triwalk.updateSurfacePoints(fidx, uv, delta)

        self.sample_fidxs = torch.tensor(fidx).long().to(self.device)
        self.sample_bary[..., :2] = torch.tensor(uv).float().to(self.device)
        self.sample_bary[..., 2] = 1.0 - self.sample_bary[..., 0] - self.sample_bary[..., 1]

    ########################################
    def load_from_embedding(self, embed_fn):
        with open(embed_fn, 'r') as fp:
            cc = json.load(fp)

        mesh_fn = cc['cano_mesh']
        if not os.path.isabs(mesh_fn):
            mesh_fn = str(Path(embed_fn).parent / mesh_fn)
        
        cano_mesh = libcore.MeshCpu(mesh_fn)
        cano_verts = torch.tensor(cano_mesh.V).float().to(self.device)
        cano_norms = torch.tensor(cano_mesh.N).float().to(self.device)
        cano_faces = torch.tensor(cano_mesh.F).long().to(self.device)
        self.setup_canonical(cano_verts, cano_norms, cano_faces)

        self._xyz = torch.tensor(cc['_xyz']).float().to(self.device)
        self.sample_fidxs = torch.tensor(cc['sample_fidxs']).long().to(self.device)
        self.sample_bary = torch.tensor(cc['sample_bary']).float().to(self.device)
        
    def save_embedding_json(self, embed_fn):
        obj_fn = embed_fn.replace('.json', '.obj')
        cano_mesh = libcore.MeshCpu()
        cano_mesh.V = self.cano_verts.detach().cpu()
        cano_mesh.N = self.cano_norms.detach().cpu()
        cano_mesh.F = self.cano_faces.detach().cpu()
        cano_mesh.FN = cano_mesh.F
        cano_mesh.save_to_obj(obj_fn)

        embedding = {
            'cano_mesh': Path(obj_fn).name,
            'sample_fidxs': self.sample_fidxs.detach().cpu().tolist(),
            'sample_bary': self.sample_bary.detach().cpu().tolist(),
            '_xyz': self._xyz.detach().cpu().tolist(),
        }

        with open(os.path.join(embed_fn), 'w') as f:
            json.dump(embedding, f)


