# Dataset helper for Frameset.
# Contributer(s): Neil Z. Shao
# All rights reserved. Prometheus 2022-2024.
import random
import torch
from tqdm import tqdm

def make_frameset_data(dataset_config, split='train', frm_list=None):
    if not dataset_config.get('resolution', False):
        dataset_config.resolution = 1

    if not dataset_config.get('data_device', False):
        dataset_config.data_device = 'cpu'

    if dataset_config.frameset_type == 'frameset' or dataset_config.frameset_type == 'color_frames':
        from .framest_data import FramesetDataset
        frameset = FramesetDataset(dataset_config, split=split, frm_list=frm_list)
    elif dataset_config.frameset_type == 'imavatar':
        from .imavatar_data import IMavatarDataset
        frameset = IMavatarDataset(dataset_config, split=split, frm_list=frm_list)
    elif dataset_config.frameset_type == 'instant_avatar':
        from .instant_avatar_reader import InstantAvatarDataset
        frameset = InstantAvatarDataset(dataset_config, split=split, frm_list=frm_list)
    else:
        raise NotImplementedError

    return frameset

def frameset_collate_fn(batches):
    return batches

def make_dataloader(frameset, shuffle=False, batch_size=1, 
                    num_workers=1, prefetch_factor=2,
                    persistent_workers=True):
    dataloader = torch.utils.data.DataLoader(frameset, shuffle=shuffle, 
                                             num_workers=num_workers, prefetch_factor=prefetch_factor,
                                             batch_size=batch_size, 
                                             persistent_workers=persistent_workers,
                                             collate_fn=frameset_collate_fn)
    return dataloader


class BatchSampler:
    def __init__(self, dataset, sample_mode='cdf', base_psnr=32.0, 
                 bulk_mode='rand_batch') -> None:
        self.dataset = dataset
        self.N = dataset.__len__()
        
        self.sample_mode = sample_mode
        self.base_psnr = base_psnr
        self.psnr_stats = torch.full((self.N,), base_psnr, dtype=torch.float)
        self.bulk_mode = bulk_mode

    def update_stats(self, idx_list, bulk_stats):
        bulk_psnr = bulk_stats['psnr'] / (bulk_stats['count'] + 1e-5)
        valid_idx = bulk_stats['count'] > 0
        self.psnr_stats[idx_list[valid_idx]] = bulk_psnr[valid_idx]
        return bulk_psnr

    def sample_cdf(self, bulk_size):
        x = self.base_psnr - self.psnr_stats
        x[x < 0] = 0
        x[x > 20] = 20

        x[1:-1] = (x[:-2] + x[1:-1] + x[2:]) / 3.0

        pdf = 1.0 - torch.exp(-x / 4.0) + 1e-3
        pdf = pdf / torch.sum(pdf, -1, keepdim=True)
        cdf = torch.cumsum(pdf, -1)

        # u = torch.linspace(0., 1., steps=bulk_size).unsqueeze(0).repeat(cdf.shape[0], 1)
        # u = torch.linspace(0., 1., steps=bulk_size)
        u = torch.rand((bulk_size,)).sort(dim=-1)[0]
        
        inds = torch.searchsorted(cdf, u, right=True)
        return inds.unique()

    def sample_small_batch(self, bulk_size=10):
        if bulk_size >= self.N:
            idx_list = torch.tensor([k for k in range(self.N)]).int()
        else:
            if self.sample_mode == 'cdf':
                idx_list = self.sample_cdf(bulk_size)
            else:
                idx_list = torch.randint(0, self.N, (bulk_size,))

        bulk_stats = {
            'count': torch.zeros((len(idx_list),)),
            'psnr': torch.zeros((len(idx_list),)),
        }

        return idx_list, bulk_stats
    
    def load_small_batch(self, bulk_size=10):
        idx_list, bulk_stats = self.sample_small_batch(bulk_size=bulk_size)

        batch_list = []
        for idx in tqdm(idx_list):
            batch = self.dataset.__getitem__(idx)
            batch_list.append(batch)
        
        self.bulk = {
            'idx_list': idx_list,
            'batch_list': batch_list,
            'bulk_stats': bulk_stats,
        }
        return self.bulk

    def next(self):
        if self.bulk_mode.startswith('rand_batch'):
            idx_list = self.bulk['idx_list']
            batch_list = self.bulk['batch_list']

            if self.bulk_mode == 'rand_batch_rand_cam':
                bulk_i = random.randint(0, len(batch_list)-1)
                idx = idx_list[bulk_i]
                viewpoint_cams = batch_list[bulk_i]['scene_cameras']
                cam_i = random.randint(0, len(viewpoint_cams)-1)
                viewpoint_cam = viewpoint_cams[cam_i].cuda()
            else:
                if 'current' not in self.bulk:
                    bulk_i = random.randint(0, len(batch_list)-1)
                    idx = idx_list[bulk_i]
                    viewpoint_stack = batch_list[bulk_i]['scene_cameras'].copy()
                    self.bulk['current'] = {
                        'bulk_i': bulk_i,
                        'idx': idx,
                        'viewpoint_stack': viewpoint_stack,
                    }

                bulk_i = self.bulk['current']['bulk_i']
                idx = self.bulk['current']['idx']
                viewpoint_stack = self.bulk['current']['viewpoint_stack']
                viewpoint_cam = viewpoint_stack.pop(random.randint(0, len(viewpoint_stack)-1)).cuda()

                if len(viewpoint_stack) == 0:
                    self.bulk.pop('current')

        return {
            'bulk_i': bulk_i,
            'idx': idx,
            'viewpoint_cam': viewpoint_cam,
        }
