import os
from pathlib import Path
from random import randint
from argparse import ArgumentParser
from gaussian_renderer import network_gui
from datetime import datetime
from tqdm import tqdm
from omegaconf import OmegaConf
from model.splatting_avatar_model import SplattingAvatarModel
from model.splatting_avatar_optim import SplattingAvatarOptimizer
from model.loss_base import run_testing
from dataset.dataset_helper import make_frameset_data, make_dataloader
from model import libcore

if __name__ == '__main__':
    parser = ArgumentParser(description='SplattingAvatar Training')
    parser.add_argument('--ip', type=str, default='127.0.0.1')
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--dat_dir', type=str, required=True)
    parser.add_argument('--configs', type=lambda s: [i for i in s.split(';')], 
                        required=True, help='path to config file')
    parser.add_argument('--model_path', type=str, default=None)
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
    frameset_train = make_frameset_data(config.dataset, split='train')
    frameset_test = make_frameset_data(config.dataset, split='test')
    dataloader = make_dataloader(frameset_train, shuffle=True)

    # first frame as canonical
    first_batch = frameset_train.__getitem__(0)
    cano_mesh = first_batch['mesh_info']

    ##################################################
    pipe = config.pipe
    gs_model = SplattingAvatarModel(config.model, verbose=True)
    gs_model.create_from_canonical(cano_mesh)

    gs_optim = SplattingAvatarOptimizer(gs_model, config.optim)

    ##################################################
    if args.ip != 'none':
        network_gui.init(args.ip, args.port)
        verify = args.dat_dir
    else:
        verify = None

    data_iterator = iter(dataloader)

    total_iteration = config.optim.total_iteration
    save_every_iter = config.optim.get('save_every_iter', 10000)
    testing_iterations = config.optim.get('testing_iterations', [total_iteration])

    pbar = tqdm(range(1, total_iteration+1))
    for iteration in pbar:        
        gs_optim.update_learning_rate(iteration)

        try:
            batches = next(data_iterator)
        except:
            data_iterator = iter(dataloader)
            batches = next(data_iterator)

        batch = batches[0]
        frm_idx = batch['frm_idx']
        scene_cameras = batch['scene_cameras']

        # update to current posed mesh
        gs_model.update_to_posed_mesh(batch['mesh_info'])
            
        # there should be only one camera
        viewpoint_cam = scene_cameras[0].cuda()
        gt_image = viewpoint_cam.original_image
        
        # send one image to gui (optional)
        if args.ip != 'none':
            network_gui.render_to_network(gs_model, pipe, verify, gt_image=gt_image)

        # render
        render_pkg = gs_model.render_to_camera(viewpoint_cam, pipe)
        image = render_pkg['render']
        gt_image = render_pkg['gt_image']
        gt_alpha_mask = render_pkg['gt_alpha_mask']

        # ### debug ###
        # from model import libcore
        # libcore.write_tensor_image(os.path.join('e:/dummy/gt_image.jpg'), gt_image, rgb2bgr=True)
        # libcore.write_tensor_image(os.path.join('e:/dummy/render.jpg'), image, rgb2bgr=True)

        # loss
        loss = gs_optim.collect_loss(gt_image, image, gt_alpha_mask=gt_alpha_mask)
        loss['loss'].backward()

        # densify and prune
        gs_optim.adaptive_density_control(render_pkg, iteration)

        gs_optim.step()
        gs_optim.zero_grad(set_to_none=True)

        pbar.set_postfix({
            '#gauss': gs_model.num_gauss,
            'loss': loss['loss'].item(),
            'psnr': loss['psnr_full'],
        })

        # walking on triangles
        gs_optim.update_trangle_walk(iteration)

        # report testing
        if iteration in testing_iterations:
            run_testing(pipe, frameset_test, gs_model, model_path, iteration, verify=verify)

        # save
        if iteration % save_every_iter == 0:
            pc_dir = gs_optim.save_checkpoint(model_path, iteration)
            libcore.write_tensor_image(os.path.join(pc_dir, 'gt_image.jpg'), gt_image, rgb2bgr=True)
            libcore.write_tensor_image(os.path.join(pc_dir, 'render.jpg'), image, rgb2bgr=True)

    ##################################################
    # training finished. hold on
    while network_gui.conn is not None:
        network_gui.render_to_network(gs_model, pipe, args.dat_dir)

    print('[done]')

