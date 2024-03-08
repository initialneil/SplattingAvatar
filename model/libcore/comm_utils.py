# Common utils.
# Contributer(s): Neil Z. Shao
# All rights reserved. Prometheus 2022-2024.

########################################
def read_list_from_txt(fn):
    frm_list = []
    with open(fn) as f:
        for line in f.readlines():
            line = line.replace('\n', '')
            if len(line) > 0:
                frm_list.append(line)
    return frm_list

########################################
from collections import namedtuple

def to_namedtuple(class_name, contents):
    ClassT = namedtuple(class_name, [k for k in contents.keys()])
    item = ClassT(**contents)
    return item

########################################
import concurrent.futures

def parallel_foreach(_func, _args_list, max_workers=8):
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executrer:
        res_list = []
        for res in executrer.map(_func, _args_list):
            res_list.append(res)
    return res_list

########################################
from datetime import datetime

# @20240117-191934
def datetime_strftime(format='@%Y%m%d-%H%M%S'):
    return datetime.now().strftime(format)

########################################
# cache with max size
class CacheBuffer:
    def __init__(self, max_size=20) -> None:
        self.cache = {}
        self.max_size = max_size

    def __repr__(self) -> str:
        return self.cache

    def set(self, key, value):
        if key not in self.cache:
            self.make_space()

        self.cache[key] = {
            'value': value,
            'timestamp': datetime_strftime(),
        }

    def get(self, key, default=None):
        if key in self.cache:
            self.cache[key]['timestamp'] = datetime_strftime()
            return self.cache[key]['value']
        return None

    def haskey(self, key):
        return key in self.cache

    def make_space(self):
        if len(self.cache) > 0 and len(self.cache) >= self.max_size:
            items = sorted(self.cache.items(), key=lambda item: item[1]['timestamp'])
            last_key = items[0][0]
            self.cache.pop(last_key)

########################################
def set_seed(seed=0):
    try:
        import torch
        torch.manual_seed(seed)
    except:
        pass

    try:
        import random
        random.seed(seed)
    except:
        pass

    try:
        import numpy as np
        np.random.seed(seed)
    except:
        pass

