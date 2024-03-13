import os
import cv2
import numpy as np
from pathlib import Path
from random import randint
from argparse import ArgumentParser
from scene.dataset_readers import convert_to_scene_cameras
from gaussian_renderer import network_gui
from datetime import datetime
from tqdm import tqdm
from omegaconf import OmegaConf
from model.std_gauss_model import StandardGaussModel
from model.std_gauss_optim import StandardGaussOptimizer
from utils.sh_utils import SH2RGB
from utils.graphics_utils import BasicPointCloud
from model import libcore

##################################################
def load_frameset(frm_dir, thresh_focal=None):
    color_frames = libcore.loadDataVecFromFolder(frm_dir, with_frames=False)
    if thresh_focal is not None:
        cam_select = [i for i in range(len(color_frames.cams)) if color_frames.cams[i].fx < thresh_focal]
        color_frames = color_frames.toSubSet(cam_select)
    color_frames.load_images_parallel(4)
    return color_frames

def sample_pcs(cams, num_pts):
    cam_pos = np.stack([cam.c for cam in cams])
    center = cam_pos.mean(axis=0)
    radius = np.linalg.norm(cam_pos - center, axis=1).mean()
    center[1] /= 2

    # We create random points inside the bounds of the synthetic Blender scenes
    xyz = (np.random.random((num_pts, 3)) - 0.5) * radius + center
    shs = np.random.random((num_pts, 3)) / 255.0
    pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))
    return pcd
##################################################

if __name__ == '__main__':
    parser = ArgumentParser(description="test tetgen")
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--dat_dir', type=str, required=True)
    parser.add_argument('--configs', type=lambda s: [i for i in s.split(';')], 
                        required=True, help='path to config file')
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--init_gs', type=str, default=None, help="initial 3dgs ply")
    args, extras = parser.parse_known_args()

    # output dir
    if args.model_path is None:
        model_path = f"output-splatting/{datetime.now().strftime('@%Y%m%d-%H%M%S')}"
    else:
        model_path = args.model_path
        
    if not os.path.isabs(model_path):
        model_path = os.path.join(args.dat_dir, model_path)
    os.makedirs(model_path, exist_ok=True)

    # load model and training config
    config = libcore.load_from_config(args.configs, cli_args=extras)
    OmegaConf.save(config, os.path.join(model_path, 'config.yaml'))
    libcore.set_seed(config.get('seed', 9061))

    ##################################################
    config.dataset.dat_dir = args.dat_dir
    config.cache_dir = os.path.join(args.dat_dir, f'cache_{Path(args.configs[0]).stem}')

    ##################################################
    color_frames = load_frameset(f'{args.dat_dir}/color_frames')
    if config.get('dataset', None):
        if config.dataset.get('resolution', None):
            resolution = config.dataset.resolution
            for i in range(color_frames.size):
                cam = color_frames.cams[i]
                img = color_frames.frames[i]
                if 0:
                    cam.scaleIntrinsics(cam.w // resolution, cam.h // resolution)
                    img = cv2.resize(img, (cam.w, cam.h))
                else:
                    for j in range(int(np.log(resolution) // np.log(2))):
                        img = cv2.pyrDown(img)
                    cam.scaleIntrinsics(img.shape[1], img.shape[0])

                color_frames.cams[i] = cam
                color_frames.frames[i] = img

    scene_cameras = convert_to_scene_cameras(color_frames)

    ##################################################
    pipe = config.pipe
    gs_model = StandardGaussModel(config.model, verbose=True)

    if args.init_gs is not None:
        if not os.path.isabs(args.init_gs):
            args.init_gs = os.path.join(args.dat_dir, args.init_gs)
        gs_model.load_ply(args.init_gs)
    else:
        pcd = sample_pcs(color_frames.cams, 10000)
        gs_model.create_from_pcd(pcd, 2.0)
    
    gs_optim = StandardGaussOptimizer(gs_model, config.optim)

    ##################################################
    if args.ip != 'none':
        network_gui.init(args.ip, args.port)
    viewpoint_stack = None

    total_iteration = config.optim.total_iteration
    save_every_iter = config.optim.get('save_every_iter', 10000)

    pbar = tqdm(range(1, total_iteration+1))
    for iteration in pbar:
        gs_optim.update_learning_rate(iteration)

        # pick a random camera to train
        if not viewpoint_stack:
            viewpoint_stack = scene_cameras.copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1)).cuda()
        gt_image = viewpoint_cam.original_image.cuda()

        # send one image to gui (optional)
        if args.ip != 'none':
            network_gui.render_to_network(gs_model, pipe, args.dat_dir, gt_image=gt_image)

        # render
        render_pkg = gs_model.render_to_camera(viewpoint_cam, pipe)
        image = render_pkg['render']
        gt_image = render_pkg['gt_image']
        gt_alpha_mask = render_pkg['gt_alpha_mask']

        # loss
        loss = gs_optim.collect_loss(gt_image, image, gt_alpha_mask=gt_alpha_mask)
        loss['loss'].backward()

        gs_optim.adaptive_density_control(render_pkg, iteration)

        gs_optim.step()
        gs_optim.zero_grad(set_to_none=True)

        pbar.set_postfix({
            '#gauss': gs_model.num_gauss,
            'loss': loss['loss'].item(),
            'psnr': loss['psnr_full'],
        })

        # save
        if save_every_iter > 0 and iteration % save_every_iter == 0:
            pc_dir = gs_optim.save_checkpoint(model_path, iteration)
            libcore.write_tensor_image(os.path.join(pc_dir, 'gt_image.jpg'), gt_image, rgb2bgr=True)
            libcore.write_tensor_image(os.path.join(pc_dir, 'render.jpg'), image, rgb2bgr=True)

    # training finished. hold on
    if args.ip != 'none':
        while network_gui.conn is not None:
            network_gui.render_to_network(gs_model, pipe, args.dat_dir)

    print('[done]')

