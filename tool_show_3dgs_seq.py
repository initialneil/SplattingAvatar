import os
from argparse import ArgumentParser
from arguments import PipelineParams
from gaussian_renderer import network_gui
from omegaconf import OmegaConf
from model.std_gauss_model import StandardGaussModel
from tqdm import tqdm

if __name__ == '__main__':
    parser = ArgumentParser(description="test tetgen")
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--ply_dir', type=str, required=True)
    parser.add_argument('--sh_degree', type=int, default=0)
    args, extras = parser.parse_known_args()
    pipe = pp.extract(args)

    ##################################################
    config = OmegaConf.create({
        'sh_degree': args.sh_degree,
    })

    pipe.compute_cov3D_python = False
    pipe.convert_SHs_python = False
    gs_model = StandardGaussModel(config)

    ply_fns = [fn for fn in os.listdir(args.ply_dir) if fn.endswith('.ply')]
    ply_fns = ply_fns   #[:10]

    params_all = []
    for fn in tqdm(ply_fns):
        gs_model.load_ply(os.path.join(args.ply_dir, fn))
        params_all.append(gs_model.get_params())

    ##################################################
    curr_idx = 0
    frm_i = 0
    play_fps = 10

    network_gui.init(args.ip, args.port)

    while True:
        frm_i = int(curr_idx / play_fps)
        gs_model.set_params(params_all[frm_i])

        # send one image to gui (optional)
        do_training = network_gui.render_to_network(gs_model, pipe, 'show_3dgs')
        while not do_training:
            do_training = network_gui.render_to_network(gs_model, pipe, 'show_3dgs')

        curr_idx = (curr_idx + 1) % len(ply_fns * play_fps)

    print('[done]')

