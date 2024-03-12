# SMPL-X helper.
# Contributer(s): Neil Z. Shao
# All rights reserved. Prometheus 2022-2024.
import os, copy
import torch
import cv2
import numpy as np
from collections import namedtuple
from .. import libcore
from . import smplx

def get_smplx_model_path(model_type='smplx', fn=None):
    if fn is None:
        return os.path.join(os.path.dirname(__file__), 'smplx_models')
    else:
        return os.path.join(os.path.dirname(__file__), 'smplx_models', model_type, fn)

def create_smplx_model(model_path=None, gender='neutral', model_type='smplx', ext='npz',
                       skip_betas=False, skip_v_template=False, skip_poses=True,
                       **smplx_params):
    if model_path is None:
        model_path = get_smplx_model_path()
    elif not os.path.exists(model_path):
        model_path = get_smplx_model_path(model_type=model_type, fn=model_path)

    if skip_betas:
        if 'betas' in smplx_params:
            smplx_params.pop('betas')

    if skip_v_template:
        if 'v_template' in smplx_params:
            smplx_params.pop('v_template')
            
    if skip_poses:
        keys = [key for key in smplx_params]
        for key in keys:
            if isinstance(smplx_params[key], torch.Tensor):
                if key != 'betas' and key != 'v_template':
                    smplx_params.pop(key)

    smplx_model = smplx.create(model_path, gender=gender, model_type=model_type, ext=ext,
                               **smplx_params)
    return smplx_model

def create_smplx_lite_model(model_path=None, gender='male', model_type='smplx-lite', ext='pkl',
                       **smplx_params):
    if model_path is None:
        model_path = get_smplx_model_path()
    elif not os.path.exists(model_path):
        model_path = get_smplx_model_path(model_type=model_type, fn=model_path)

    model_path = os.path.join(model_path, model_type, 'SMPLX-LITE_{}.{ext}'.format(gender.upper(), ext=ext))

    smplx_model = smplx.create(model_path, gender=gender, model_type=model_type, ext=ext,
                               **smplx_params)
    return smplx_model

def load_regressor(regressor_path):
    if regressor_path.endswith('.npy'):
        X_regressor = torch.tensor(np.load(regressor_path)).float()
    elif regressor_path.endswith('.txt'):
        data = np.loadtxt(regressor_path)
        with open(regressor_path, 'r') as f:
            shape = f.readline().split()[1:]
        reg = np.zeros((int(shape[0]), int(shape[1])))
        for i, j, v in data:
            reg[int(i), int(j)] = v
        X_regressor = torch.tensor(reg).float()
    else:
        import ipdb; ipdb.set_trace()
    return X_regressor

def load_smplx_J_regressor_body25_smplx(fn='J_regressor_body25_smplx.txt'):
    if not os.path.isabs(fn):
        fn = get_smplx_model_path(model_type='', fn=fn)
    return load_regressor(fn)

def load_smplx_J_regressor_body25_smplx_lite(fn='J_regressor_body25_smplx_lite.txt'):
    if not os.path.isabs(fn):
        fn = get_smplx_model_path(model_type='', fn=fn)
    return load_regressor(fn)

def write_J_regressor(fn, J_regressor):
    with open(fn, 'w') as fp:
        fp.write(f'# {J_regressor.shape[0]} {J_regressor.shape[1]}\n')
        for i in range(J_regressor.shape[0]):
            for j in range(J_regressor.shape[1]):
                if J_regressor[i, j] != 0:
                    fp.write(f'{i} {j} {J_regressor[i, j]}\n')

def load_smplx_part_labels(gender='male', model_type='smplx', model_path=None):
    assert gender in ['male', 'female']

    if model_path is None:
        model_path = get_smplx_model_path()

    # https://github.com/Skype-line/X-Avatar#quick-demo
    # load part labels
    import pickle as pkl
    verts_ids = pkl.load(open(os.path.join(model_path, model_type, f'non_watertight_{gender}_vertex_labels.pkl'), 'rb'), 
                         encoding='latin1')
    return verts_ids        

def convert_smplx_to_meshcpu(smplx_model, V=None):
    if V is None:
        V = smplx_model.v_template
    if isinstance(V, torch.Tensor):
        V = V.detach().cpu().numpy()
    
    mesh = libcore.MeshCpu()
    mesh.V = V
    mesh.F = smplx_model.faces.astype(int)
    mesh.update_per_vertex_normals()
    if hasattr(smplx_model, 'tc'):
        mesh.TC = smplx_model.tc
        mesh.FTC = smplx_model.tc_faces
    return mesh

def save_smplx_to_obj(fn, smplx_model, V=None):
    mesh = convert_smplx_to_meshcpu(smplx_model, V=V)
    mesh.save_to_obj(fn)

def write_smplx_objs(smplx_dir, frm_list, smplx_model, out, max_workers=8):
    def _write_smplx_obj(idx):
        frm_idx = frm_list[idx]
        mesh = convert_smplx_to_meshcpu(smplx_model, V=out['vertices'][idx])
        mesh.save_to_obj(os.path.join(smplx_dir, f'smplx_{frm_idx:06d}.obj'))

    num_frames = len(frm_list)
    idxs = [i for i in range(num_frames)]

    import concurrent.futures
    from tqdm import tqdm
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executrer:
        for res in tqdm(executrer.map(_write_smplx_obj, idxs), total=num_frames):
            pass

def convert_smplx_params_cv2gl(smplx_params):
    # create model
    smplx_model = create_smplx_model(**smplx_params)

    # target verts in gl
    out = smplx_model(**smplx_params)
    verts_gl = out['vertices'].detach().clone()
    verts_gl[0, :, 1:3] = -verts_gl[0, :, 1:3]

    # flip R
    rvec = smplx_params['poses'][0, :3].numpy()
    smplx_R = cv2.Rodrigues(rvec)[0]
    smplx_R_gl = copy.deepcopy(smplx_R)
    smplx_R_gl[1:3, :] = -smplx_R_gl[1:3, :]
    rvec_gl = cv2.Rodrigues(smplx_R_gl)[0]

    # calc transl to target
    smplx_params_gl = copy.deepcopy(smplx_params)
    smplx_params_gl['poses'][0, :3] = torch.from_numpy(rvec_gl).view(-1)
    smplx_params_gl['transl'] = torch.zeros_like(smplx_params_gl['transl'])
    out = smplx_model(**smplx_params_gl)
    
    smplx_t_gl = (verts_gl - out['vertices']).mean(dim=-2)
    smplx_params_gl['transl'] = smplx_t_gl

    return smplx_params_gl

##################################################
def load_and_detach(fn, map_location='cpu'):
    smplx_params = torch.load(fn, map_location=map_location)
    for key in smplx_params:
        if isinstance(smplx_params[key], torch.Tensor):
            smplx_params[key] = smplx_params[key].detach()

            if len(smplx_params[key].shape) == 1:
                smplx_params[key] = smplx_params[key].unsqueeze(0)

    if 'use_pca' not in smplx_params:
        smplx_params['use_pca'] = True
    if 'flat_hand_mean' not in smplx_params:
        smplx_params['flat_hand_mean'] = True
    if 'num_betas' not in smplx_params and 'betas' in smplx_params:
        smplx_params['num_betas'] = smplx_params['betas'].shape[-1]
    if 'num_expression_coeffs' not in smplx_params and 'expression' in smplx_params:
        smplx_params['num_expression_coeffs'] = smplx_params['expression'].shape[-1]
    if 'gender' not in smplx_params:
        smplx_params['gender'] = 'male'
    if 'model_type' not in smplx_params:
        smplx_params['model_type'] = 'smplx'

    return smplx_params
