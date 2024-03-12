# The standard camera class.
# Contributer(s): Neil Z. Shao
# All rights reserved. Prometheus 2022-2024.
import json
import numpy as np
import copy
from .json_utils import *
from .transform import convertRtFromCV2GL

class Camera:
    def __init__(self):
        # camera sn
        self.info = ''
        
        # width height
        self.w = 0; self.h = 0
        
        # intrinsics
        self.fx = 0; self.fy = 0; self.cx = 0; self.cy = 0; self.fov = 0
        
        # distort (skip for now)
        
        # extrinsics
        # R: rotation matrix
        # c: camera center in world coordinates
        # t: t = -R * c; c = -R' * t
        self.R = np.eye(3)
        self.c = np.zeros(3)

    def __repr__(self) -> str:
        return '[Camera] %dx%d' % (self.w, self.h) + \
            '\n[Camera] fx = %.2f, fy = %.2f, cx = %.2f, cy = %.2f' % (
                self.fx, self.fy, self.cx, self.cy) + \
            '\n[Camera] R = %.6f, %.6f, %.6f' % (
                self.R[0, 0], self.R[0, 1], self.R[0, 2]) + \
            '\n[Camera]     %.6f, %.6f, %.6f' % (
                self.R[1, 0], self.R[1, 1], self.R[1, 2]) + \
            '\n[Camera]     %.6f, %.6f, %.6f' % (
                self.R[2, 0], self.R[2, 1], self.R[2, 2]) + \
            '\n[Camera] c = %.6f, %.6f, %.6f' % (
                self.c[0], self.c[1], self.c[2])

    @property
    def t(self):
        return -self.R.dot(self.c)

    def setTranslation(self, t):
        self.c = -self.R.transpose().dot(t)

    @property
    def K(self):
        return np.array([
            [self.fx, 0, self.cx], 
            [0, self.fy, self.cy], 
            [0, 0, 1]])
    
    @property
    def K_homo(self):
        return np.array([
            [self.fx, 0, self.cx, 0], 
            [0, self.fy, self.cy, 0], 
            [0, 0, 1, 0],
            [0, 0, 0, 1]])

    def set_K(self, K, w=0, h=0):
        self.fx = K[0, 0].item()
        self.fy = K[1, 1].item()
        self.cx = K[0, 2].item()
        self.cy = K[1, 2].item()
        if w > 0:
            self.w = w
        if h > 0:
            self.h = h

    def set_c2w(self, c2w):
        self.R = c2w[:3, :3].T
        self.c = c2w[:, 3]

    def set_Rt(self, Rt):
        for i in range(3):
            for j in range(3):
                self.R[i, j] = Rt[i, j]

        t = np.array([0.0, 0.0, 0.0])
        for i in range(3):
            t[i] = Rt[i, 3]
        self.setTranslation(t)

    @property
    def Rt_w2c(self):
        Rt = np.eye(4)
        for i in range(3):
            for j in range(3):
                Rt[i, j] = self.R[i, j]
            Rt[i, 3] = self.t[i]
        return Rt

    @property
    def sz(self):
        return [int(self.w), int(self.h)]

    # scale intrinsics
    def scaleIntrinsics(self, width, height):
        width = int(width)
        height = int(height)

        if width <= 0 or height <= 0:
            return
        if self.w == width and self.h == height:
            return

        scale_x = float(width) / self.w
        scale_y = float(height) / self.h
        self.fx *= scale_x
        self.fy *= scale_y

        dx = self.cx - self.w / 2.0
        dy = self.cy - self.h / 2.0
        self.cx = width / 2.0 + dx * scale_x
        self.cy = height / 2.0 + dy * scale_y

        self.w = width
        self.h = height

    def scaleIntrinsicsBaseWidth(self, base_width):
        scale = float(base_width) / self.w
        if scale == 0:
            return

        work_w = self.w * scale
        work_h = self.h * scale
        self.scaleIntrinsics(work_w, work_h)

    # clone
    def clone(self):
        return copy.deepcopy(self)

    # copy
    def copyFrom(self, other):
        self.fx = other.fx
        self.fy = other.fy
        self.cx = other.cx
        self.cy = other.cy
        self.w = other.w
        self.h = other.h
        self.info = other.info
        self.R = copy.deepcopy(other.R)
        self.c = copy.deepcopy(other.c)

    # convert Rt to OpenGL coordinates
    def toOpenGL(self):
        cam_gl = copy.deepcopy(self)
        cam_gl.R, t_gl = convertRtFromCV2GL(self.R, self.t)
        cam_gl.setTranslation(t_gl)
        return cam_gl

    # io
    def saveToJson(self):
        cam_value = dict()
        cam_value['info'] = self.info
        cam_value['fx'] = float(self.fx)
        cam_value['fy'] = float(self.fy)
        cam_value['cx'] = float(self.cx)
        cam_value['cy'] = float(self.cy)
        cam_value['w'] = int(self.w)
        cam_value['h'] = int(self.h)
        cam_value['R'] = self.R.tolist()
        cam_value['c'] = self.c.tolist()
        return cam_value

    def loadFromJson(self, value):
        self.info = value['info']
        self.fx = value['fx']
        self.fy = value['fy']
        self.cx = value['cx']
        self.cy = value['cy']
        self.w = value['w']
        self.h = value['h']
        self.R = readMatrixFromJson(value['R'])
        self.c = readVectorFromJson(value['c'])

    def saveToFile(self, fn):
        val = self.saveToJson()
        import json
        with open(fn, 'w') as fp:
            json.dump(val, fp)

    def loadFromFile(self, fn):
        with open(fn, 'r') as fp:
            val = json.load(fp)
        self.loadFromJson(val)

class Rig:
    def __init__(self):
        self.cams = []
        self.info = ''
        self.image_format = 'RGB'

    def saveToJson(self):
        value = dict()
        value['info'] = self.info
        value['image_format'] = self.image_format

        cams_json = []
        for i in range(len(self.cams)):
            # cam_value = self.cams[i].saveToJson()
            cam_value = dict()
            cam_value['info'] = self.cams[i].info
            cam_value['fx'] = float(self.cams[i].fx)
            cam_value['fy'] = float(self.cams[i].fy)
            cam_value['cx'] = float(self.cams[i].cx)
            cam_value['cy'] = float(self.cams[i].cy)
            cam_value['w'] = int(self.cams[i].w)
            cam_value['h'] = int(self.cams[i].h)
            cam_value['R'] = self.cams[i].R.tolist()
            cam_value['c'] = self.cams[i].c.tolist()
            cams_json.append(cam_value)
        value['cameras'] = cams_json
        return value

    def loadFromJson(self, value):
        self.info = value['info']
        self.image_format = value['image_format']
        
        self.cams = []
        cams_json = value['cameras']
        for i in range(0, len(cams_json)):
            cam = Camera()
            cam.loadFromJson(cams_json[i])
            self.cams.append(cam)
        return True

    def saveToFile(self, fn):
        value = self.saveToJson()
        with open(fn, 'w') as f:
            json.dump(value, f)
        return

    def loadFromFile(self, fn):
        with open(fn, 'r') as f:
            value = json.load(f)
            return self.loadFromJson(value)

# crop image boundary and update camera
def crop_camera_image_boundary(cam, img, top, bottom, left, right):
    crop_cam = copy.deepcopy(cam)
    crop_img = copy.deepcopy(img[top:cam.h-bottom, left:cam.w-right])

    crop_cam.w = cam.w - right - left
    crop_cam.h = cam.h - top - bottom
    crop_cam.cx = cam.cx - left
    crop_cam.cy = cam.cy - top

    return crop_cam, crop_img
