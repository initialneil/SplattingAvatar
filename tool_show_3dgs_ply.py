from argparse import ArgumentParser
from gaussian_renderer import network_gui
from omegaconf import OmegaConf
from model.std_gauss_model import StandardGaussModel

if __name__ == '__main__':
    parser = ArgumentParser(description="test tetgen")
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--ply_fn', type=str, required=True)
    parser.add_argument('--sh_degree', type=int, default=0)
    args, extras = parser.parse_known_args()

    ##################################################
    config = OmegaConf.create({
        'sh_degree': args.sh_degree,
    })
    gs_model = StandardGaussModel(config)
    gs_model.load_ply(args.ply_fn)

    pipe = OmegaConf.create({
        'compute_cov3D_python': False,
        'convert_SHs_python': False,
        'debug': False,
    })

    ##################################################
    network_gui.init(args.ip, args.port)

    while True:
        # send one image to gui (optional)
        network_gui.render_to_network(gs_model, pipe, 'show_3dgs')

    print('[done]')

