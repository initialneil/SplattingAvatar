import os
import torch
import numpy as np
import json
from collections import namedtuple
from model import libcore
from .map import get_triangles, quaternion_to_rotation_matrix, calc_per_vert_quaternion

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

def retrieve_verts_barycentric(vertices, faces, fidxs, barys):
    triangle_verts = vertices[faces].float()

    if len(triangle_verts.shape) == 3:
        sample_verts = torch.einsum('nij,ni->nj', triangle_verts[fidxs], barys)
    elif len(triangle_verts.shape) == 4:
        sample_verts = torch.einsum('bnij,ni->bnj', triangle_verts[:, fidxs, ...], barys)
    else:
        raise NotImplementedError
    
    return sample_verts

def load_canonical_json(fn):
    with open(fn) as f:
        cano_info = json.load(f)

    mesh_fn = cano_info['mesh_fn']
    mesh = libcore.MeshCpu(os.path.join(os.path.dirname(fn), mesh_fn))

    cano_verts = torch.tensor(mesh.V).float()
    cano_norms = torch.tensor(mesh.N).float()
    cano_faces = torch.tensor(mesh.F).long()
    sample_fidxs = torch.tensor(cano_info['sample_fidxs']).long()
    sample_bary = torch.tensor(cano_info['sample_bary'])

    info = {
        'mesh_fn': mesh_fn,
        'mesh': mesh,
        'cano_verts': cano_verts,
        'cano_norms': cano_norms,
        'cano_faces': cano_faces,
        'sample_fidxs': sample_fidxs,
        'sample_bary': sample_bary,
        'sample_verts': retrieve_verts_barycentric(cano_verts, cano_faces, sample_fidxs, sample_bary),
    }

    if 'smplx_fn' in cano_info:
        info.update({ 'smplx_fn': cano_info['smplx_fn'] })

    if '_xyz' in cano_info:
        info.update({'xyz': torch.tensor(cano_info['_xyz'])})

    if '_sample_bary' in cano_info:
        info.update({'sample_bary': torch.tensor(cano_info['_sample_bary'])})

    if '_rotation' in cano_info:
        info.update({'rotation': torch.tensor(cano_info['_rotation'])})

    CanoInfo = namedtuple('cano', [k for k in info.keys()])
    cano = CanoInfo(**info)
    return cano
    
def retrieve_deformed_sample(cano_verts, cano_faces, sample_fidxs, sample_bary, 
                             mesh_verts, mesh_norms, mesh_faces):
    
    sample_verts = retrieve_verts_barycentric(mesh_verts, mesh_faces, sample_fidxs, sample_bary)
    sample_norms = retrieve_verts_barycentric(mesh_norms, mesh_faces, sample_fidxs, sample_bary)

    per_vert_quat = calc_per_vert_quaternion(cano_verts, cano_faces, mesh_verts)
    triangle_quats = per_vert_quat[cano_faces]
    sample_quats = torch.einsum('bij,bi->bj', triangle_quats[sample_fidxs], sample_bary)
    sample_R = quaternion_to_rotation_matrix(sample_quats)

    # sanity check
    if per_vert_quat.isnan().sum() > 0 or per_vert_quat.isinf().sum() > 0:
        print('[ERROR] per_vert_quat.isnan().sum() > 0 or per_vert_quat.isinf().sum() > 0')
        raise ValueError
    
    return {
        'sample_verts': sample_verts, 
        'sample_norms': sample_norms, 
        'sample_R': sample_R,
    }

def save_guass_rotation_to_ply(fn, xyz, quat, scaling=None, finger_lens=0.005):
    import pytorch3d
    from utils.general_utils import build_scaling_rotation
    if scaling is None:
        R = pytorch3d.transforms.quaternion_to_matrix(quat)
    else:
        R = build_scaling_rotation(scaling, quat)
        finger_lens = 1.0
    cams = []
    for i in range(xyz.shape[0]):
        cam = libcore.Camera()
        cam.c = xyz[i]
        cam.R = R[i].T      # gauss record transpose of R
        cams.append(cam)
    libcore.saveCamerasToPly(fn, cams, finger_lens=finger_lens)

def sort_and_unique(new_xyz, index_batch):
    order_idxs = index_batch.argsort()
    new_xyz = new_xyz[order_idxs]
    index_batch = index_batch[order_idxs]

    unique, idx, counts = torch.unique(index_batch, sorted=True, return_inverse=True, return_counts=True)
    _, ind_sorted = torch.sort(idx, stable=True)
    cum_sum = counts.cumsum(0)
    cum_sum = torch.cat((torch.tensor([0]).to(cum_sum), cum_sum[:-1]))
    first_indicies = ind_sorted[cum_sum]
    new_xyz = new_xyz[first_indicies]
    index_batch = index_batch[first_indicies]

    return new_xyz, index_batch


