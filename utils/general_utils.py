#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import sys
from datetime import datetime
import numpy as np
import random
import cv2

def inverse_sigmoid(x):
    return torch.log(x/(1-x))

def PILtoTorch(pil_image, resolution):
    resized_image_PIL = pil_image.resize(resolution)
    resized_image = torch.tensor(np.array(resized_image_PIL)) / 255.0
    if len(resized_image.shape) == 3:
        return resized_image.permute(2, 0, 1)
    else:
        return resized_image.unsqueeze(dim=-1).permute(2, 0, 1)

def numpyToTorch(image, resolution):
    resized_image_rgb = cv2.resize(image, resolution)
    resized_image_rgb = torch.tensor(resized_image_rgb, dtype=torch.float) / 255.0
    if len(resized_image_rgb.shape) == 3:
        return resized_image_rgb.permute(2, 0, 1)
    else:
        return resized_image_rgb.unsqueeze(dim=-1).permute(2, 0, 1)

def get_expon_lr_func(
    lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000
):
    """
    Copied from Plenoxels

    Continuous learning rate decay function. Adapted from JaxNeRF
    The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
    is log-linearly interpolated elsewhere (equivalent to exponential decay).
    If lr_delay_steps>0 then the learning rate will be scaled by some smooth
    function of lr_delay_mult, such that the initial learning rate is
    lr_init*lr_delay_mult at the beginning of optimization but will be eased back
    to the normal learning rate when steps>lr_delay_steps.
    :param conf: config subtree 'lr' or similar
    :param max_steps: int, the number of steps during optimization.
    :return HoF which takes step as input
    """

    def helper(step):
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            # Disable this parameter
            return 0.0
        if lr_delay_steps > 0:
            # A kind of reverse cosine decay.
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        t = np.clip(step / max_steps, 0, 1)
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return delay_rate * log_lerp

    return helper

def strip_lowerdiag(L):
    uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device="cuda")

    uncertainty[:, 0] = L[:, 0, 0]
    uncertainty[:, 1] = L[:, 0, 1]
    uncertainty[:, 2] = L[:, 0, 2]
    uncertainty[:, 3] = L[:, 1, 1]
    uncertainty[:, 4] = L[:, 1, 2]
    uncertainty[:, 5] = L[:, 2, 2]
    return uncertainty

def strip_symmetric(sym):
    return strip_lowerdiag(sym)

def _build_rotation(r):
    norm = torch.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device='cuda')

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - r*z)
    R[:, 0, 2] = 2 * (x*z + r*y)
    R[:, 1, 0] = 2 * (x*y + r*z)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - r*x)
    R[:, 2, 0] = 2 * (x*z - r*y)
    R[:, 2, 1] = 2 * (y*z + r*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)
    return R

def build_rotation(r):
    if len(r.shape) == 2:
        return _build_rotation(r)
    
    input_shape = r.shape
    R = _build_rotation(r.reshape(-1, 4))
    return R.view(*input_shape[:-1], 3, 3)

def build_scaling_rotation(s, r):
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device="cuda")
    R = build_rotation(r)

    L[:,0,0] = s[:,0]
    L[:,1,1] = s[:,1]
    L[:,2,2] = s[:,2]

    L = R @ L
    return L

def build_scaling_rotation_tet(s, r, tet_J):
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device="cuda")
    R = build_rotation(r)

    L[:,0,0] = s[:,0]
    L[:,1,1] = s[:,1]
    L[:,2,2] = s[:,2]

    L = tet_J @ (R @ L)
    return L

def safe_state(silent):
    old_f = sys.stdout
    class F:
        def __init__(self, silent):
            self.silent = silent

        def write(self, x):
            if not self.silent:
                if x.endswith("\n"):
                    old_f.write(x.replace("\n", " [{}]\n".format(str(datetime.now().strftime("%d/%m %H:%M:%S")))))
                else:
                    old_f.write(x)

        def flush(self):
            old_f.flush()

    sys.stdout = F(silent)

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.set_device(torch.device("cuda:0"))

def build_quat(R):
    tr = torch.eye(3)[None,...].repeat(R.shape[0], 1, 1).to(R)
    w = torch.pow(1 + (tr * R).sum(-1).sum(-1), 0.5)/2
    x = (R[:, 2, 1] - R[:, 1, 2])/4/w
    y = (R[:, 0, 2] - R[:, 2, 0])/4/w
    z = (R[:, 1, 0] - R[:, 0, 1])/4/w
    quat = torch.stack([w, x, y, z], dim=-1)
    return quat

def sample_bary_on_triangles(num_faces, num_samples):
    sample_bary = torch.zeros(num_samples, 3)
    sample_bary[:, 0] = torch.rand(num_samples)
    sample_bary[:, 1] = torch.rand(num_samples) * (1.0 - sample_bary[:, 0])
    sample_bary[:, 2] = 1.0 - sample_bary[:, 0] - sample_bary[:, 1]
    sample_fidxs = torch.randint(0, num_faces, size=(num_samples,))

    # shuffle bary
    indices = torch.argsort(torch.rand_like(sample_bary), dim=-1)
    sample_bary = torch.gather(sample_bary, dim=-1, index=indices)

    return sample_fidxs, sample_bary

def sample_bary_on_tetrahedrons(num_faces, num_samples):
    sample_bary = torch.zeros(num_samples, 4)
    sample_bary[:, 0] = torch.rand(num_samples)
    sample_bary[:, 1] = torch.rand(num_samples) * (1.0 - sample_bary[:, 0])
    sample_bary[:, 2] = torch.rand(num_samples) * (1.0 - sample_bary[:, 0] - sample_bary[:, 1])
    sample_bary[:, 3] = 1.0 - sample_bary[:, 0] - sample_bary[:, 1] - sample_bary[:, 2]
    sample_fidxs = torch.randint(0, num_faces, size=(num_samples,))

    # shuffle bary
    indices = torch.argsort(torch.rand_like(sample_bary), dim=-1)
    sample_bary = torch.gather(sample_bary, dim=-1, index=indices)

    return sample_fidxs, sample_bary

def retrieve_verts_barycentric(vertices, faces, fidxs, barys):
    triangle_verts = vertices[faces].float()

    if len(triangle_verts.shape) == 3:
        sample_verts = torch.einsum('nij,ni->nj', triangle_verts[fidxs], barys)
    elif len(triangle_verts.shape) == 4:
        sample_verts = torch.einsum('bnij,ni->bnj', triangle_verts[:, fidxs, ...], barys)
    else:
        raise NotImplementedError
    
    return sample_verts
