import os
from argparse import ArgumentParser
from model.splatting_avatar_model import SplattingAvatarModel
from model.loss_base import run_testing
from dataset.dataset_helper import make_frameset_data, make_dataloader
from gaussian_renderer import network_gui
from model import libcore

if __name__ == '__main__':
    parser = ArgumentParser(description='SplattingAvatar Evaluation')
    parser.add_argument('--ip', type=str, default='127.0.0.1')
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--dat_dir', type=str, required=True)
    parser.add_argument('--configs', type=lambda s: [i for i in s.split(';')], 
                        required=True, help='path to config file')
    parser.add_argument('--pc_dir', type=str, default=None)
    args, extras = parser.parse_known_args()

    # load model and training config
    config = libcore.load_from_config(args.configs, cli_args=extras)

    ##################################################
    config.dataset.dat_dir = args.dat_dir
    frameset_test = make_frameset_data(config.dataset, split='test')

    ##################################################
    pipe = config.pipe
    
    gs_model = SplattingAvatarModel(config.model, verbose=True)
    ply_fn = os.path.join(args.pc_dir, 'point_cloud.ply')
    gs_model.load_ply(ply_fn)
    embed_fn = os.path.join(args.pc_dir, 'embedding.json')
    gs_model.load_from_embedding(embed_fn)

    ##################################################
    if args.ip != 'none':
        network_gui.init(args.ip, args.port)
        verify = args.dat_dir
    else:
        verify = None

    ##################################################
    run_testing(pipe, frameset_test, gs_model, args.pc_dir, verify=verify)

    print('[done]')

