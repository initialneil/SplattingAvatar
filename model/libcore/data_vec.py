# DataVec for data vector.
# Contributer(s): Neil Z. Shao
# All rights reserved. Prometheus 2022-2024.
from .camera import *
from .ply_utils import saveCamerasToPly
import json
import cv2
import os
from copy import deepcopy

def _load_image_file(img_fn):
    img = cv2.imread(img_fn, cv2.IMREAD_UNCHANGED)

    # depth
    if (img.dtype == np.dtype('uint16')):
        img = img.astype(float) / 10000.0

    return img

class DataVec:
    def __init__(self) -> None:
        self.frames = []
        self.cams = []
        self.cam_models = []
        self.image_formats = []
        self.sns = []
        self.frm_idx = -1
        self.images_path = []

    def __repr__(self) -> str:
        return '[DataVec] size = %d' % (self.size)

    @property
    def size(self):
        return len(self.cams)

    def loadFromFile(self, fn):
        rigs = []
        try:
            with open(fn, 'r') as f:
                value = json.load(f)
                for i in range(0, len(value['rigs'])):
                    rig = Rig()
                    rig.loadFromJson(value['rigs'][i])
                    rigs.append(rig)
            
            self.sns = []
            self.cams = []
            self.image_formats = []
            for i in range(len(rigs)):
                self.sns.append(rigs[i].info)
                self.cams.append(rigs[i].cams[0])
                self.image_formats.append(rigs[i].image_format)
            
            return True
        except:
            return False

    def saveToFile(self, fn):
        value = {}
        value['rigs'] = []
        for i in range(self.size):
            rig = Rig()
            rig.cams.append(self.cams[i])
            rig.info = self.sns[i] if i < len(self.sns) else ''
            rig.image_format = self.image_formats[i] if i < len(self.image_formats) else ''
            value['rigs'].append(rig.saveToJson())

        with open(fn, 'w', encoding='utf-8') as f:
            json.dump(value, f, ensure_ascii=False, indent=4)

    def copyInfo(self, other):
        self.cams = other.cams
        self.frm_idx = other.frm_idx
        self.cam_models = other.cam_models
        self.image_formats = other.image_formats
        self.sns = other.sns

    def copyData(self, other):
        self.copyInfo(other)
        self.frames = deepcopy(other.frames)
        self.cams = other.cams

    def clone(self):
        clone_vec = DataVec()
        clone_vec.copyData(self)
        return clone_vec

    def toColorSet(self):
        sub_idxs = [i for i, value in enumerate(self.image_formats) if value == 'RGB']
        sub_vec = DataVec()
        sub_vec.frm_idx = self.frm_idx
        sub_vec.cams = [self.cams[i] for i in sub_idxs]
        sub_vec.frames = [self.frames[i] for i in sub_idxs]
        sub_vec.cam_models = [self.cam_models[i] for i in sub_idxs] if len(self.cam_models) > 0 else []
        sub_vec.image_formats = [self.image_formats[i] for i in sub_idxs]
        sub_vec.sns = [self.sns[i] for i in sub_idxs]
        return sub_vec, sub_idxs

    def toSubSet(self, sub_idxs):
        sub_vec = DataVec()
        sub_vec.frm_idx = sub_vec.frm_idx

        if len(self.cams) == self.size:
            sub_vec.cams = [self.cams[i] for i in sub_idxs]
        if len(self.frames) == self.size:
            sub_vec.frames = [self.frames[i] for i in sub_idxs]
        if len(self.images_path) == self.size:
            sub_vec.images_path = [self.images_path[i] for i in sub_idxs]
        if len(self.cam_models) == self.size:
            sub_vec.cam_models = [self.cam_models[i] for i in sub_idxs] if len(self.cam_models) > 0 else []
        if len(self.image_formats) == self.size:
            sub_vec.image_formats = [self.image_formats[i] for i in sub_idxs]
        if len(self.sns) == self.size:
            sub_vec.sns = [self.sns[i] for i in sub_idxs]
            
        return sub_vec
    
    def load_images(self):
        self.frames = []
        print('[loadDataVecFromFolder] loading ', end='')
        for i in range(0, self.size):
            print('.', end='')
            img_fn = self.images_path[i]
            img = _load_image_file(img_fn)
            self.frames.append(img)
        print('[done]')

    def load_images_parallel(self, max_workers):
        self.frames = []
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executrer:
            for res in executrer.map(_load_image_file, self.images_path):
                self.frames.append(res)

def loadDataVecFromFolder(dir, with_frames=True, max_workers=4):
    data_vec = DataVec()

    # load cameras
    data_vec.loadFromFile(dir + "/cameras.json")

    data_vec.frames = []
    data_vec.images_path = []

    for i in range(0, data_vec.size):
        img_fn = dir + '/%02d.png' % i
        data_vec.images_path.append(img_fn)

    # load images
    if with_frames:
        if max_workers <= 0:
            data_vec.load_images()
        else:
            data_vec.load_images_parallel(max_workers)
        
    return data_vec

def saveDataVecToFolder(dir, data_vec, with_frames=True):
    os.makedirs(dir, exist_ok=True)
    if not os.path.exists(dir):
        print('[saveDataVecToFolder][ERROR] make dir failed: %s' % dir)
        return

    # save cameras
    data_vec.saveToFile(dir + "/cameras.json")

    # save images
    if with_frames:
        print('[saveDataVecToFolder] saving ', end='')
        for i in range(0, len(data_vec.frames)):
            print('.', end='')
            img = data_vec.frames[i]

            # depth
            if (img.dtype == np.float32 or img.dtype == np.float64):
                img = (img * 10000.0).astype(np.uint16)

            img_fn = dir + '/%02d.png' % i
            cv2.imwrite(img_fn, img)

        print('[done]')

    saveCamerasToPly(dir + '/cams.ply', data_vec.cams)
