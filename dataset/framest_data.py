# Frameset Reader.
# Contributer(s): Neil Z. Shao
# All rights reserved. Prometheus 2022-2024.
import os
import torch
from scene.dataset_readers import convert_to_scene_cameras
from utils.graphics_utils import BasicPointCloud
from model import libcore
# from prometheus.bone_deformer import smplx_utils

def read_frame_list(fn):
    frm_list = []
    with open(fn) as f:
        for line in f.readlines():
            line = line.replace('\n', '')
            if len(line) > 0:
                frm_list.append(line)
    return frm_list

class FramesetDataset(torch.utils.data.Dataset):
    def __init__(self, config, split='train', frm_list=None):
        self.config = config
        self.split = split

        if frm_list is None:
            split_fn = config.get(f'split_{split}_fn', 'train.txt')
            self.frm_list = read_frame_list(os.path.join(config.dat_dir, split_fn))
        else:
            self.frm_list = frm_list
        self.num_frames = len(self.frm_list)
        print(f'[FramesetDataset][{self.split}] num_frames = {self.num_frames}')

        self.dat_dir = config.dat_dir
        self.verbose_timer = config.get('verbose_timer', False)
        self.frameset_type = config.get('frameset_type', 'color_frames')
        self.smplx_forward_transl = config.get('smplx_forward_transl', False)
        self.cameras_extent = config.get('cameras_extent', 2.0)

        self.load_all_smplx(config.get('all_smplx_fn', None))

        self.cache_buffer = libcore.CacheBuffer(max_size=config.get('cache_size', 20))

    def load_all_smplx(self, fn=None):
        if fn is None:
            fn = os.path.join(self.config.dat_dir, 'all_smplx.pt')
        elif not os.path.isabs(fn):
            fn = os.path.join(self.config.yaml_dir, fn)

        if os.path.exists(fn):
            smplx_params = torch.load(fn, map_location='cpu')

            idxs = [int(idx) for idx in self.frm_list]
            for k in smplx_params:
                if isinstance(smplx_params[k], torch.Tensor):
                    smplx_params[k] = smplx_params[k].detach().clone().cpu()
                    if smplx_params[k].shape[0] > 1:
                        smplx_params[k] = smplx_params[k][idxs]
            self.smplx_params = smplx_params

            if 'smplx_forward_transl' in smplx_params:
                self.smplx_forward_transl = smplx_params['smplx_forward_transl']
        else:
            self.smplx_params = None

    def __len__(self):
        return len(self.frm_list)

    def __getitem__(self, idx=None):
        if idx is None:
            idx = torch.randint(0, len(self.frm_list), (1,)).item()

        frm_idx = int(self.frm_list[idx])
        if self.cache_buffer.haskey(frm_idx):
            return self.cache_buffer.get(frm_idx)

        batch = {
            'idx': idx,
            'frm_idx': frm_idx,
        }

        ##########
        if self.verbose_timer:
            libcore.startCpuTimer('[FramesetDataset] readPromethInfo')

        datavec_dir = os.path.join(self.dat_dir, f'{frm_idx:06d}', self.frameset_type)
        color_frames = libcore.loadDataVecFromFolder(datavec_dir, with_frames=False)
        if self.config.get('select_focal_lt', 0) > 0:
            t_focal = self.config.select_focal_lt
            cam_select = [i for i in range(len(color_frames.cams)) if color_frames.cams[i].fx < t_focal]
            color_frames = color_frames.toSubSet(cam_select)
        color_frames.load_images_parallel(max_workers=4)
        scene_cameras = convert_to_scene_cameras(color_frames, self.config)
        
        batch.update({
            'color_frames': color_frames,
            'scene_cameras': scene_cameras,
            'cameras_extent': self.cameras_extent,
        })

        if self.verbose_timer:
            libcore.stopCpuTimer('[FramesetDataset] cameraList_from_camInfos')

        # # load smplx if needed
        # smplx_params = self.load_smplx_params(idx)
        # if smplx_params is not None:
        #     batch.update({
        #         'smplx_params': smplx_params,
        #     })

        self.cache_buffer.set(frm_idx, batch)
        return batch

    # def load_smplx_params(self, idx):
    #     if self.smplx_params is not None:
    #         smplx_params = {}
    #         for key in self.smplx_params:
    #             if not isinstance(self.smplx_params[key], torch.Tensor):
    #                 smplx_params[key] = self.smplx_params[key]
    #             else:
    #                 if key == 'betas':
    #                     smplx_params[key] = self.smplx_params[key].detach().clone()
    #                 else:
    #                     smplx_params[key] = self.smplx_params[key][idx].detach().clone()
    #                     if len(smplx_params[key].shape) == 1:
    #                         smplx_params[key] = smplx_params[key].unsqueeze(0)
    #         return smplx_params

    #     frm_idx = int(self.frm_list[idx])

    #     smplx_type = self.config.get('smplx_type', None)
    #     if smplx_type is None:
    #         return None

    #     if smplx_type == 'nvdiffsmplx_out':
    #         smplx_fn = os.path.join(self.config.dat_dir, f'nvdiffsmplx_out/{frm_idx:06d}', 'smplx.pt')
    #     elif smplx_type == 'smplx-refined':
    #         smplx_fn = os.path.join(self.config.dat_dir, f'{frm_idx:06d}', 'smplx-refined.pt')
    #     elif smplx_type == 'smplx_params_list':
    #         smplx_fn = os.path.join(self.config.yaml_dir, f'smplx_params_list/smplx-{frm_idx:06d}.pt')
    #     else:
    #         smplx_fn = os.path.join(self.config.dat_dir, f'{smplx_type}/smplx-{frm_idx:06d}.pt')

    #     return smplx_utils.load_and_detach(smplx_fn) if os.path.exists(smplx_fn) else None
    
    # def sample_pcd(self):
    #     batch = self.__getitem__(0)
    #     smplx_model = smplx_utils.create_smplx_model(**batch['smplx_params'])
    #     out = smplx_model(**batch['smplx_params'])
    #     pcd = BasicPointCloud(out['vertices'][0], 
    #                           torch.full_like(out['vertices'][0], 0.1),
    #                           torch.zeros_like(out['vertices'][0]))
    #     return pcd
