# InstantAvatar/PeopleSnapshot data Reader.
# Contributer(s): Neil Z. Shao
# All rights reserved. Prometheus 2022-2024.
import os
import cv2
import json
from copy import deepcopy
import torch
import numpy as np
from scene.dataset_readers import convert_to_scene_cameras
from model import libcore
from model.smplx_utils import smplx_utils
import pytorch3d.structures.meshes as py3d_meshes

def read_instant_avatar_frameset(dat_dir, frm_idx, cam, extension='.png'):
    image_path = os.path.join(dat_dir, f'images/image_{frm_idx:04d}.png')
    mask_path = os.path.join(dat_dir, f'masks/mask_{frm_idx:04d}.png')
    
    image = cv2.imread(image_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
    image = np.concatenate([image, mask[:, :, None]], axis=-1)

    color_frames = libcore.DataVec()
    color_frames.cams = [cam]
    color_frames.frames = [image]
    color_frames.images_path = [image_path]
    return color_frames

class InstantAvatarDataset(torch.utils.data.Dataset):
    def __init__(self, config, split='train', frm_list=None):
        self.config = config
        self.split = split

        self.dat_dir = config.dat_dir
        self.cameras_extent = config.get('cameras_extent', 1.0)

        self.load_config_file()
        self.load_camera_file()
        self.load_pose_file()
        self.num_frames = len(self.frm_list)
        print(f'[InstantAvatarDataset] num_frames = {self.num_frames}')

    ##################################################
    # load config.json
    def load_config_file(self):
        if not os.path.exists(os.path.join(self.dat_dir, 'config.json')):
            raise NotImplementedError
        
        with open(os.path.join(self.dat_dir, 'config.json'), 'r') as fp:
            contents = json.load(fp)

        self.start_idx = contents[self.split]['start']
        self.end_idx = contents[self.split]['end']
        self.step = contents[self.split]['skip']
        self.frm_list = [i for i in range(self.start_idx, self.end_idx+1, self.step)]

        self.smpl_config = {
            'model_type': 'smpl',
            'gender': contents['gender'],
        }

    # load cameras.npz
    def load_camera_file(self):
        if not os.path.exists(os.path.join(self.dat_dir, 'cameras.npz')):
            raise NotImplementedError

        contents = np.load(os.path.join(self.dat_dir, 'cameras.npz'))
        K = contents["intrinsic"]
        c2w = np.linalg.inv(contents["extrinsic"])
        height = contents["height"]
        width = contents["width"]
        w2c = np.linalg.inv(c2w)

        R = w2c[:3,:3]
        T = w2c[:3, 3]

        cam = libcore.Camera()
        cam.h, cam.w = height, width
        cam.set_K(K)
        cam.R = R
        cam.setTranslation(T)
        # print(cam)
        self.cam = cam

    # load poses.npz
    def load_pose_file(self):
        if not os.path.exists(os.path.join(self.dat_dir, 'poses.npz')):
            raise NotImplementedError
        
        smpl_params = dict(np.load(os.path.join(self.dat_dir, 'poses.npz')))

        if "thetas" in smpl_params:
            smpl_params["body_pose"] = smpl_params["thetas"][..., 3:]
            smpl_params["global_orient"] = smpl_params["thetas"][..., :3]

        self.smpl_params = {
            "betas": torch.tensor(smpl_params["betas"].astype(np.float32).reshape(1, 10)),
            "body_pose": torch.tensor(smpl_params["body_pose"].astype(np.float32))[self.frm_list],
            "global_orient": torch.tensor(smpl_params["global_orient"].astype(np.float32))[self.frm_list],
            "transl": torch.tensor(smpl_params["transl"].astype(np.float32))[self.frm_list],
        }

        # cano mesh
        self.smpl_model = smplx_utils.create_smplx_model(**self.smpl_config)
        with torch.no_grad():
            out = self.smpl_model(**self.smpl_params)

        # verts_normals_padded is not updated except for the first batch
        # so init with only one batch of verts and use update_padded later
        self.mesh_py3d = py3d_meshes.Meshes(out['vertices'][:1], 
                                            torch.tensor(self.smpl_model.faces[None, ...].astype(int)))
        self.cano_mesh = {
            'mesh_verts': self.mesh_py3d.verts_packed().detach().clone(),
            'mesh_norms': self.mesh_py3d.verts_normals_packed().detach().clone(),
            'mesh_faces': self.mesh_py3d.faces_packed().detach().clone(),
        }

        # load refined
        refine_fn = os.path.join(self.dat_dir, f"poses/anim_nerf_{self.split}.npz")
        if os.path.exists(refine_fn):
            print(f'[InstantAvatar] use refined smpl: {refine_fn}')
            split_smpl_params = np.load(refine_fn)
            refined_keys = [k for k in split_smpl_params if k != 'betas']
            self.smpl_params['betas'] = torch.tensor(split_smpl_params['betas']).float()
            for key in refined_keys:
                self.smpl_params[key] = torch.tensor(split_smpl_params[key]).float()
            
        self.smpl_model = smplx_utils.create_smplx_model(**self.smpl_config)
        with torch.no_grad():
            out = self.smpl_model(**self.smpl_params)
        self.smpl_verts = out['vertices']

        # # verts_normals_padded is not updated except for the first batch
        # # so init with only one batch of verts and use update_padded later
        # self.mesh_py3d = py3d_meshes.Meshes(self.smpl_verts[:1], 
        #                                     torch.tensor(self.smpl_model.faces[None, ...].astype(int)))

    ##################################################
    def __len__(self):
        return len(self.frm_list)

    def __getitem__(self, idx):
        if idx is None:
            idx = torch.randint(0, len(self.frm_list), (1,)).item()

        frm_idx = self.frm_list[idx]

        ##########
        color_frames = read_instant_avatar_frameset(self.dat_dir, frm_idx, self.cam)
        scene_cameras = convert_to_scene_cameras(color_frames, self.config)
        
        batch = {
            'idx': idx,
            'frm_idx': frm_idx,
            'color_frames': color_frames,
            'scene_cameras': scene_cameras,
            'cameras_extent': self.cameras_extent,
        }

        ##########
        batch['mesh_info'] = self.get_smpl_mesh(idx)
        return batch

    def get_smpl_mesh(self, idx):
        frame_mesh = self.mesh_py3d.update_padded(self.smpl_verts[idx:idx+1])
        return {
            'mesh_verts': frame_mesh.verts_packed(),
            'mesh_norms': frame_mesh.verts_normals_packed(),
            'mesh_faces': frame_mesh.faces_packed(),
        }

    

    