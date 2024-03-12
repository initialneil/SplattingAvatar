# IMavatar data Reader.
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
from model.imavatar.flame import FLAME
import pytorch3d.structures.meshes as py3d_meshes

def read_imavatar_frameset(dat_dir, frame_info, intrinsics, extension='.png'):
    w2c = np.array(frame_info['world_mat'])
    w2c = np.concatenate([w2c, np.array([[0, 0, 0, 1]])], axis=0)

    # standard R, t in OpenCV coordinate
    R = w2c[:3,:3]
    T = w2c[:3, 3]
    R[1:, :] = -R[1:, :]
    T[1:] = -T[1:]

    # Note:
    # R is stored transposed (R.T) due to 'glm''s column-major storage in CUDA code

    # dirty fix
    file_path = frame_info['file_path']
    file_path = file_path.replace('/image/', '/images/')

    image_path = os.path.abspath(os.path.join(dat_dir, file_path + extension)).replace('\\', '/')
    if not os.path.exists(image_path):
        image_path = image_path.replace('/images/', '/image/')

    image_name = os.path.basename(image_path)
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

    cam = libcore.Camera()
    cam.h, cam.w = image.shape[:2]
    cam.fx = abs(cam.w * intrinsics[0])
    cam.fy = abs(cam.h * intrinsics[1])
    cam.cx = abs(cam.w * intrinsics[2])
    cam.cy = abs(cam.h * intrinsics[3])
    cam.R = R
    cam.setTranslation(T)
    # print(cam)

    color_frames = libcore.DataVec()
    color_frames.cams = [cam]
    color_frames.frames = [image]
    color_frames.images_path = [image_path]
    return color_frames

class IMavatarDataset(torch.utils.data.Dataset):
    def __init__(self, config, split='train', frm_list=None):
        self.config = config
        self.split = split

        self.dat_dir = config.dat_dir
        self.cameras_extent = config.get('cameras_extent', 1.0)
        self.num_for_train = config.get('num_for_train', -350)

        self.load_flame_json()
        self.num_frames = len(self.frm_list)
        print(f'[IMavatarDataset] num_frames = {self.num_frames}')

    ##################################################
    # load flame_params.json
    def load_flame_json(self):
        if not os.path.exists(os.path.join(self.dat_dir, 'flame_params.json')):
            raise NotImplementedError
        
        with open(os.path.join(self.dat_dir, 'flame_params.json')) as fp:
            contents = json.load(fp)
            self.intrinsics = contents['intrinsics']
            self.shape_params = torch.tensor(contents['shape_params']).unsqueeze(0)

            if self.split == 'train':
                self.frames_info = contents['frames'][:self.num_for_train]
            else:
                self.frames_info = contents['frames'][self.num_for_train:]

            self.frm_list = []
            self.flame_params = []
            for frame in self.frames_info:
                frm_idx = os.path.basename(frame['file_path'])
                self.frm_list.append(frm_idx)
                self.flame_params.append({
                    'full_pose': torch.tensor(frame['pose']).unsqueeze(0),
                    'expression_params': torch.tensor(frame['expression']).unsqueeze(0),
                })

        """
        - This is the IMAvatar/DECA version of FLAME
        - Originally from: https://github.com/zhengyuf/IMavatar/tree/main/code/flame
        - What's changed from normal FLAME:
          - There's a `factor=4` in the `flame.py`, making the output mesh 4 times larger
          - The input `full_pose` is [Nx15], which is a combination of different pose components
          - In a standard FLAME model, there is `pose_params`[Nx6], `neck_pose`[Nx3], `eye_pose`[Nx6].
            To convert to `full_pose`:
            ```
            # [3] global orient
            # [3] neck
            # [3] jaw
            # [6] eye
            full_pose = torch.concat([pose_params[:, :3], neck_pose, pose_params[:, 3:], eye_pose], dim=-1)
            ```
        """
        self.flame = FLAME('model/imavatar/FLAME2020/generic_model.pkl', 
                           'model/imavatar/FLAME2020/landmark_embedding.npy',
                           n_shape=100,
                           n_exp=50,
                           shape_params=self.shape_params,
                           canonical_expression=None,
                           canonical_pose=None)
        self.mesh_py3d = py3d_meshes.Meshes(self.flame.v_template[None, ...].float(), 
                                            torch.from_numpy(self.flame.faces[None, ...].astype(int)))

    ##################################################
    def __len__(self):
        return len(self.frm_list)

    def __getitem__(self, idx):
        if idx is None:
            idx = torch.randint(0, len(self.frm_list), (1,)).item()

        frm_idx = int(idx)

        ##########
        color_frames = read_imavatar_frameset(self.dat_dir, self.frames_info[idx], self.intrinsics)
        scene_cameras = convert_to_scene_cameras(color_frames, self.config)
        
        batch = {
            'idx': idx,
            'frm_idx': frm_idx,
            'color_frames': color_frames,
            'scene_cameras': scene_cameras,
            'cameras_extent': self.cameras_extent,
        }

        ##########
        batch['mesh_info'] = self.get_flame_mesh(idx)
        return batch

    def get_flame_mesh(self, idx):
        with torch.no_grad():
            flame_params = self.get_flame_params(idx)
            vertices, _, _ = self.flame(flame_params['expression_params'], flame_params['full_pose'])

            frame_mesh = self.mesh_py3d.update_padded(vertices)

        return {
            'mesh_verts': frame_mesh.verts_packed(),
            'mesh_norms': frame_mesh.verts_normals_packed(),
            'mesh_faces': frame_mesh.faces_packed(),
        }

    def get_flame_params(self, idx):
        return {
            'n_shape': 100,
            'n_exp': 50,
            'shape_params': self.shape_params,
            **self.flame_params[idx],
        }
    

    