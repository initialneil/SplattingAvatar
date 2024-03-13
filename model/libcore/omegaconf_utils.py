# Omegaconf utils.
# Mostly from https://github.com/bennyguo/instant-nsr-pl/blob/main/utils/misc.py
# Contributer(s): Neil Z. Shao
# All rights reserved. Prometheus 2022-2024.
import os
from omegaconf import OmegaConf
from packaging import version
from pathlib import Path

# ============ Register OmegaConf Recolvers ============= #
OmegaConf.register_new_resolver('calc_exp_lr_decay_rate', lambda factor, n: factor**(1./n))
OmegaConf.register_new_resolver('add', lambda a, b: a + b)
OmegaConf.register_new_resolver('sub', lambda a, b: a - b)
OmegaConf.register_new_resolver('mul', lambda a, b: a * b)
OmegaConf.register_new_resolver('div', lambda a, b: a / b)
OmegaConf.register_new_resolver('idiv', lambda a, b: a // b)
OmegaConf.register_new_resolver('basename', lambda p: os.path.basename(p))
# ======================================================= #


def prompt(question):
    inp = input(f"{question} (y/n)").lower().strip()
    if inp and inp == 'y':
        return True
    if inp and inp == 'n':
        return False
    return prompt(question)


def load_config(*yaml_files, cli_args=[]):
    yaml_confs = [OmegaConf.load(f) for f in yaml_files]
    cli_conf = OmegaConf.from_cli(cli_args)
    conf = OmegaConf.merge(*yaml_confs, cli_conf)
    OmegaConf.resolve(conf)
    # conf.yaml_confs = yaml_confs
    conf.yaml_dir = str(Path(yaml_files[0]).parent).replace('\\', '/')
    # conf.yaml_dirs = [str(Path(fn).parent).replace('\\', '/') for fn in yaml_files]
    return conf

def load_from_config(yaml_files, dat_dir=None, cli_args=[]):
    if dat_dir is not None:
        for i in range(len(yaml_files)):
            if not os.path.isabs(yaml_files[i]):
                yaml_files[i] = os.path.join(dat_dir, yaml_files[i])
    config = load_config(*yaml_files, cli_args=cli_args)
    return config

def config_to_primitive(config, resolve=True):
    return OmegaConf.to_container(config, resolve=resolve)


def dump_config(path, config):
    with open(path, 'w') as fp:
        OmegaConf.save(config=config, f=fp)

def get_rank():
    # SLURM_PROCID can be set even if SLURM is not managing the multiprocessing,
    # therefore LOCAL_RANK needs to be checked first
    rank_keys = ("RANK", "LOCAL_RANK", "SLURM_PROCID", "JSM_NAMESPACE_RANK")
    for key in rank_keys:
        rank = os.environ.get(key)
        if rank is not None:
            return int(rank)
    return 0


def parse_version(ver):
    return version.parse(ver)


