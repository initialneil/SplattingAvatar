#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import sys
import torch
from PIL import Image
from typing import NamedTuple
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from utils.camera_utils import loadCam
from utils.graphics_utils import BasicPointCloud
from tqdm import tqdm
import cv2
import random

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int

class CameraInfoExt(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}

def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        # image = Image.open(image_path)
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                              image_path=image_path, image_name=image_name, width=width, height=height)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T

    if 'red' not in vertices:
        shs = np.random.random((positions.shape[0], 3)) / 255.0
        colors = SH2RGB(shs)
    else:
        colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0

    if 'nx' not in vertices:
        normals=np.zeros((positions.shape[0], 3))
    else:
        normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def readColmapSceneInfo(path, images, eval, llffhold=8):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = "images" if images == None else images
    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(path, reading_dir))
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png"):
    cam_infos = []

    from model import libcore
    cam_dir = os.path.join(os.path.join(path, 'cameras'))
    os.makedirs(cam_dir, exist_ok=True)

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        for idx, frame in tqdm(enumerate(frames)):
            if '.png' in frame["file_path"]:
                cam_name = os.path.join(path, frame["file_path"])
            else:
                cam_name = os.path.join(path, frame["file_path"] + extension)

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            # c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = w2c[:3,:3]
            _R = np.transpose(R)  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            # image_path = os.path.join(path, cam_name)
            # image_name = Path(cam_name).stem
            # image = Image.open(image_path)

            # im_data = np.array(image.convert("RGBA"))

            # bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            # norm_data = im_data / 255.0
            # arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            # image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

            image_path = os.path.join(path, cam_name)
            image_name = os.path.basename(image_path)
            image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)

            h, w = image.shape[:2]
            fovy = focal2fov(fov2focal(fovx, w), h)
            FovY = fovy 
            FovX = fovx
            cam_info = CameraInfo(uid=idx, R=_R, T=T, FovY=FovY, FovX=FovX, image=image,
                                  image_path=image_path, image_name=image_name, width=w, height=h)

            cam_infos.append(cam_info)

            # dump
            fn = os.path.join(cam_dir, f'{idx:05d}.json')
            if not os.path.exists(fn):
                cam = libcore.Camera()
                cam.w = w
                cam.h = h
                cam.fx = fov2focal(fovx, w)
                cam.fy = fov2focal(fovy, h)
                cam.cx = w * 0.5
                cam.cy = h * 0.5
                cam.R = R
                cam.setTranslation(T)
                cam.saveToFile(fn)
                libcore.saveCamerasToPly(fn.replace('.json', '.ply'), [cam])
            
    return cam_infos

def framesetToCameraInfo(idx, cam, image, image_path=''):
    R = cam.R
    c = cam.c[:, None]
    
    R_inv = np.linalg.inv(R)
    c2w = np.concatenate([np.concatenate([R_inv, c], axis=1), np.array([[0, 0, 0, 1]])])
    w2c = np.linalg.inv(c2w)
    
    R = np.transpose(w2c)[:3, :3]
    T = w2c[:3, 3]
    
    h, w = image.shape[:2]
    return CameraInfoExt(uid=idx, R=R, T=T, image=image,
                         image_path=image_path, image_name=image_path, width=w, height=h,
                         fx=cam.fx, fy=cam.fy, cx=cam.cx, cy=cam.cy)

def readNerfSyntheticInfo(path, white_background, eval, extension=".png", with_ply=True):
    if not eval:
        # print("Reading Training Transforms")
        train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension)
        # train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []
        nerf_normalization = getNerfppNorm(train_cam_infos)
    else:
        train_cam_infos = []
        # print("Reading Test Transforms")
        test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", white_background, extension)
        nerf_normalization = getNerfppNorm(test_cam_infos)

    if with_ply:
        ply_path = os.path.join(path, "points3d.ply")
        if not os.path.exists(ply_path):
            # Since this data set has no colmap data, we start with random points
            num_pts = 100_000
            print(f"Generating random point cloud ({num_pts})...")
            
            # We create random points inside the bounds of the synthetic Blender scenes
            xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
            shs = np.random.random((num_pts, 3)) / 255.0
            pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

            storePly(ply_path, xyz, SH2RGB(shs) * 255)
        try:
            pcd = fetchPly(ply_path)
        except:
            pcd = None
    else:
        pcd = None
        ply_path = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

##################################################
def make_scene_camera(idx, cam, img, image_path='', config=None):
    if config is None:
        from collections import namedtuple
        config = namedtuple('scene_camera', ['resolution', 'data_device'])(1, 'cuda')
        resolution_scale = 1.0
    else:
        resolution_scale = config.get('resolution_scale', 1.0)

    cam_info = framesetToCameraInfo(idx, cam, img, image_path=image_path)
    scene_camera = loadCam(config, idx, cam_info, resolution_scale)
    return scene_camera

def convert_to_scene_cameras(color_frames, config=None):
    scene_cameras = []
    for idx in range(color_frames.size):
        image_path = color_frames.images_path[idx]
        img = cv2.cvtColor(color_frames.frames[idx], cv2.COLOR_BGRA2RGBA)
        camera = make_scene_camera(idx, color_frames.cams[idx], img, image_path, config)
        scene_cameras.append(camera)
    return scene_cameras

##################################################
def readCamerasFromPrometh(path, extension=".png", mini_batch=0):
    from model import libcore
    color_frames = libcore.loadDataVecFromFolder(path, with_frames=False)

    if not os.path.exists(os.path.join(path, 'cams.ply')):
        libcore.saveCamerasToPly(os.path.join(path, 'cams.ply'), color_frames.cams)

    if mini_batch > 0 and mini_batch < color_frames.size:
        mini_idxs = torch.randint(0, color_frames.size, (mini_batch,))
        color_frames = color_frames.toSubSet(mini_idxs)
    color_frames.load_images_parallel(max_workers=4)

    cam_infos = []
    for idx in range(color_frames.size):
        image_path = color_frames.images_path[idx]
        img = cv2.cvtColor(color_frames.frames[idx], cv2.COLOR_BGRA2RGBA)
        cam_infos.append(framesetToCameraInfo(idx, color_frames.cams[idx], img, image_path=image_path))
                
    return cam_infos

def readPromethInfo(path, eval, extension=".png", mini_batch=0, with_ply=False):
    # print("Reading Training Transforms")
    train_cam_infos = readCamerasFromPrometh(path, extension, mini_batch=mini_batch)
    
    if not eval:
        # train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []
    else:
        # print("Reading Test Transforms")
        test_cam_infos = readCamerasFromPrometh(path, extension, mini_batch=mini_batch)

    nerf_normalization = getNerfppNorm(train_cam_infos)

    if with_ply:
        ply_path = os.path.join(path, "points3d.ply")
        if not os.path.exists(ply_path):
            # Since this data set has no colmap data, we start with random points
            num_pts = 100_000
            print(f"Generating random point cloud ({num_pts})...")
            
            # We create random points inside the bounds of the synthetic Blender scenes
            xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
            shs = np.random.random((num_pts, 3)) / 255.0
            pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

            storePly(ply_path, xyz, SH2RGB(shs) * 255)
        try:
            pcd = fetchPly(ply_path)
        except:
            pcd = None
    else:
        pcd = None
        ply_path = None

    # temp_point = pcd.points
    # temp_point[:, 0] = - temp_point[:, 0]
    # temp_point = - temp_point
    # pcd = pcd._replace(points = temp_point)
    
    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

##################################################
def readCamerasFromTHUman40(subject, frame_id, cams, mini_batch=0):
    from model import libcore
    color_frames, color_masks = libcore.DataVec(), libcore.DataVec()
    color_frames.cams = []
    color_frames.images_path = []
    color_masks.images_path = []
    for cam_id in range(0, len(cams)):
        cam_sn = 'cam%02d' % cam_id
        img_fpath = os.path.join(subject, 'images/%s/%08d.jpg' % (cam_sn, frame_id))
        msk_fpath = os.path.join(subject, 'masks/%s/%08d.jpg' % (cam_sn, frame_id))
        if os.path.exists(img_fpath) and os.path.exists(msk_fpath):
            color_frames.cams.append(cams[cam_id])
            color_frames.images_path.append(img_fpath)
            color_masks.images_path.append(msk_fpath)
        color_masks.cams = color_frames.cams

    if mini_batch > 0 and mini_batch < color_frames.size:
        mini_idxs = torch.randint(0, color_frames.size, (mini_batch,))
        color_frames = color_frames.toSubSet(mini_idxs)
        color_masks = color_masks.toSubSet(mini_idxs)

    color_frames.load_images_parallel(max_workers=4)
    color_masks.load_images_parallel(max_workers=4)

    cam_infos = []
    for idx in range(color_frames.size):
        image_path = color_frames.images_path[idx]
        img = np.concatenate([color_frames.frames[idx], color_masks.frames[idx][:, :, None]], axis=-1)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
        cam_infos.append(framesetToCameraInfo(idx, color_frames.cams[idx], img, image_path=image_path))
                
    return cam_infos

def readTHUman40Info(path, frm_idx, cams, eval, mini_batch=0, with_ply=False):
    # print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTHUman40(path, frm_idx, cams, mini_batch=mini_batch)
    
    if not eval:
        # train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []
    else:
        # print("Reading Test Transforms")
        test_cam_infos = readCamerasFromTHUman40(path, frm_idx, cams, mini_batch=mini_batch)

    nerf_normalization = getNerfppNorm(train_cam_infos)

    if with_ply:
        ply_path = os.path.join(path, "points3d.ply")
        if not os.path.exists(ply_path):
            # Since this data set has no colmap data, we start with random points
            num_pts = 100_000
            print(f"Generating random point cloud ({num_pts})...")
            
            # We create random points inside the bounds of the synthetic Blender scenes
            xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
            shs = np.random.random((num_pts, 3)) / 255.0
            pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

            storePly(ply_path, xyz, SH2RGB(shs) * 255)
        try:
            pcd = fetchPly(ply_path)
        except:
            pcd = None
    else:
        pcd = None
        ply_path = None

    # temp_point = pcd.points
    # temp_point[:, 0] = - temp_point[:, 0]
    # temp_point = - temp_point
    # pcd = pcd._replace(points = temp_point)
    
    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

##################################################
def readCamerasFromActorsHQ(subject, frame_id, cams, mini_batch=0):
    from model import libcore
    color_frames, color_masks = libcore.DataVec(), libcore.DataVec()
    color_frames.cams = []
    color_frames.images_path = []
    color_masks.images_path = []
    for cam_id in range(0, len(cams)):
        cam_sn = 'Cam%03d' % (cam_id + 1)
        img_fpath = os.path.join(subject, 'rgbs/%s/%s_rgb%06d.jpg' % (cam_sn, cam_sn, frame_id))
        msk_fpath = os.path.join(subject, 'masks/%s/%s_mask%06d.png' % (cam_sn, cam_sn, frame_id))
        if os.path.exists(img_fpath) and os.path.exists(msk_fpath):
            color_frames.cams.append(cams[cam_id])
            color_frames.images_path.append(img_fpath)
            color_masks.images_path.append(msk_fpath)
        color_masks.cams = color_frames.cams

    if mini_batch > 0 and mini_batch < color_frames.size:
        mini_idxs = torch.randint(0, color_frames.size, (mini_batch,))
        color_frames = color_frames.toSubSet(mini_idxs)
        color_masks = color_masks.toSubSet(mini_idxs)

    color_frames.load_images_parallel(max_workers=4)
    color_masks.load_images_parallel(max_workers=4)

    cam_infos = []
    for idx in range(color_frames.size):
        image_path = color_frames.images_path[idx]
        color = color_frames.frames[idx]
        mask = color_masks.frames[idx]
        img = np.concatenate([color, mask[:, :, None]], axis=-1)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
        cam_infos.append(framesetToCameraInfo(idx, color_frames.cams[idx], img, image_path=image_path))
                
    return cam_infos

def readActorsHQInfo(path, frm_idx, cams, eval, mini_batch=0, with_ply=False):
    # print("Reading Training Transforms")
    train_cam_infos = readCamerasFromActorsHQ(path, frm_idx, cams, mini_batch=mini_batch)
    
    if not eval:
        # train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []
    else:
        # print("Reading Test Transforms")
        test_cam_infos = readCamerasFromTHUman40(path, frm_idx, cams, mini_batch=mini_batch)

    nerf_normalization = getNerfppNorm(train_cam_infos)

    if with_ply:
        ply_path = os.path.join(path, "points3d.ply")
        if not os.path.exists(ply_path):
            # Since this data set has no colmap data, we start with random points
            num_pts = 100_000
            print(f"Generating random point cloud ({num_pts})...")
            
            # We create random points inside the bounds of the synthetic Blender scenes
            xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
            shs = np.random.random((num_pts, 3)) / 255.0
            pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

            storePly(ply_path, xyz, SH2RGB(shs) * 255)
        try:
            pcd = fetchPly(ply_path)
        except:
            pcd = None
    else:
        pcd = None
        ply_path = None

    # temp_point = pcd.points
    # temp_point[:, 0] = - temp_point[:, 0]
    # temp_point = - temp_point
    # pcd = pcd._replace(points = temp_point)
    
    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def readActorsHQCameras(source_path, scale='4x'):
    import cv2
    import csv
    from model import libcore
    with open(os.path.join(source_path, scale, 'calibration.csv'), 'r') as fp:
        csv_reader = csv.DictReader(fp)
        line_count = 0
        cams = []
        for row in csv_reader:
            if line_count == 0:
                print(f'Column names are {", ".join(row)}')
                line_count += 1

            rvec = np.array([float(row['rx']), float(row['ry']), float(row['rz'])])
            c = np.array([float(row['tx']), float(row['ty']), float(row['tz'])])
            
            cam = libcore.Camera()
            cam.R = cv2.Rodrigues(rvec)[0].T
            cam.c = c
            cam.w = int(row['w'])
            cam.h = int(row['h'])
            cam.fx = float(row['fx']) * cam.w
            cam.fy = float(row['fy']) * cam.h
            cam.cx = float(row['px']) * cam.w
            cam.cy = float(row['py']) * cam.h
            cams.append(cam)

            line_count += 1
        print(f'Processed {line_count} lines.')
        libcore.saveCamerasToPly(os.path.join(source_path, 'cams.ply'), cams)
    return cams

"""
import cv2
import numpy as np
from scipy.spatial.transform import Rotation
from actorshq.dataset.camera_data import CameraData, read_calibration_csv
cameras = read_calibration_csv("/mnt/e/Datasets/ActorsHQ/humanrf/data/Actor02/Sequence1/4x/calibration.csv")
print(np.linalg.inv(cameras[0].extrinsic_matrix_cam2world()))
"""
def readActorsHQInfoFull(source_path, frm_idx, eval, scale='4x', with_ply=False):
    cams = readActorsHQCameras(source_path, scale=scale)
    scene_info = readActorsHQInfo(os.path.join(source_path, scale), frm_idx, cams, eval, with_ply=with_ply)
    return scene_info

##################################################
def readCamerasFromIMavatar(path, jsonfile, eval=False, full_data=False, extension='.png', max_frames=0):
    cam_infos = []

    with open(os.path.join(path, jsonfile)) as json_file:
        contents = json.load(json_file)
        intrinsics = contents['intrinsics']

    if full_data:
        frames = contents['frames']
    else:
        if eval:
            frames = contents['frames'][-350:]
            # frames = contents['frames'][-10:]
        else:
            frames = contents['frames'][:-350]
            # frames = contents['frames'][-350:]
            # frames = contents['frames'][:10]

    if max_frames > 0 and len(frames) > max_frames:
        frames = random.choices(frames, k=max_frames)

    for idx, frame in tqdm(enumerate(frames), total=len(frames)):
        w2c = np.array(frame['world_mat'])
        w2c = np.concatenate([w2c, np.array([[0, 0, 0, 1]])], axis=0)
        c2w = np.linalg.inv(w2c)

        R = w2c[:3,:3]
        T = w2c[:3, 3]
        R[1:, :] = -R[1:, :]
        T[1:] = -T[1:]

        # R is stored transposed due to 'glm' in CUDA code
        _R = np.transpose(R)  
        T = T

        # dirty fix
        file_path = frame['file_path']
        file_path = file_path.replace('/image/', '/images/')

        image_path = os.path.abspath(os.path.join(path, file_path + extension)).replace('\\', '/')
        if not os.path.exists(image_path):
            image_path = image_path.replace('/images/', '/image/')

        image_name = os.path.basename(image_path)
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)

        h, w = image.shape[:2]
        fx = abs(w * intrinsics[0])
        fy = abs(h * intrinsics[1])
        cx = abs(w * intrinsics[2])
        cy = abs(h * intrinsics[3])
        # FovX = focal2fov(fx, w)
        # FovY = focal2fov(fy, h)
        # cam_info = CameraInfo(uid=idx, R=_R, T=T, FovY=FovY, FovX=FovX, image=image,
        #                       image_path=image_path, image_name=image_name, width=w, height=h)

        cam_info = CameraInfoExt(uid=idx, R=_R, T=T, image=image,
                                    image_path=image_path, image_name=image_path, width=w, height=h,
                                    fx=fx, fy=fy, cx=cx, cy=cy)

        cam_infos.append(cam_info)

        # dump
        if not os.path.exists(os.path.join(path, 'cam.json')):
            from model import libcore
            cam = libcore.Camera()
            cam.w = w
            cam.h = h
            cam.fx = fx
            cam.fy = fy
            cam.cx = cx
            cam.cy = cy
            cam.R = R
            cam.setTranslation(T)
            cam.saveToFile(os.path.join(path, 'cam.json'))
            libcore.saveCamerasToPly(os.path.join(path, 'cam.ply'), [cam])
        
    return cam_infos

def readIMavatarInfo(path, eval, full_data=False, with_ply=False, extension='.png', max_frames=3000):
    json_fn = 'flame_params.json' if os.path.exists(os.path.join(path, 'flame_params.json')) else 'flame_params_nha.json'

    if not eval:
        train_cam_infos = readCamerasFromIMavatar(path, json_fn, eval=eval, full_data=full_data,
                                                  extension=extension, max_frames=max_frames)
        test_cam_infos = []
        nerf_normalization = getNerfppNorm(train_cam_infos)
    else:
        train_cam_infos = []
        test_cam_infos = readCamerasFromIMavatar(path, json_fn, eval=eval, full_data=full_data, 
                                                 extension=extension, max_frames=max_frames)
        nerf_normalization = getNerfppNorm(test_cam_infos)

    if with_ply:
        ply_path = os.path.join(path, "points3d.ply")
        if not os.path.exists(ply_path):
            # Since this data set has no colmap data, we start with random points
            num_pts = 100_000
            print(f"Generating random point cloud ({num_pts})...")
            
            # We create random points inside the bounds of the synthetic Blender scenes
            xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
            shs = np.random.random((num_pts, 3)) / 255.0
            pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

            storePly(ply_path, xyz, SH2RGB(shs) * 255)
        try:
            pcd = fetchPly(ply_path)
        except:
            pcd = None
    else:
        pcd = None
        ply_path = None

    # temp_point = pcd.points
    # temp_point[:, 0] = - temp_point[:, 0]
    # temp_point = - temp_point
    # pcd = pcd._replace(points = temp_point)
    
    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

##################################################
def readCamerasFromInstantAvatar(path, npz_fn, config, eval=False, extension='.png', max_frames=0):
    cam_infos = []

    contents = np.load(os.path.join(path, npz_fn))
    K = contents["intrinsic"]
    c2w = np.linalg.inv(contents["extrinsic"])
    height = contents["height"]
    width = contents["width"]
    w2c = np.linalg.inv(c2w)

    if eval:
        if 'frm_list' in config['test']:
            frm_list = config['test']['frm_list']
        else:
            frm_list = [k for k in range(config['test']['start'], config['test']['end']+1, config['test']['skip'])]
    else:
        if 'frm_list' in config['train']:
            frm_list = config['train']['frm_list']
        else:
            frm_list = [k for k in range(config['train']['start'], config['train']['end']+1, config['train']['skip'])]

    if max_frames > 0 and len(frm_list) > max_frames:
        frm_list = random.choices(frm_list, k=max_frames)

    for idx in tqdm(frm_list):
        if isinstance(idx, int):
            image_path = os.path.join(path, 'images', f'image_{idx:04d}.png')
            mask_path = os.path.join(path, 'masks', f'mask_{idx:04d}.png')
            image_name = Path(image_path).stem.split('_')[1]
        else:
            image_path = os.path.join(path, 'images', f'{idx}.png')
            mask_path = os.path.join(path, 'masks', f'{idx}.png')
            image_name = Path(image_path).stem

        R = w2c[:3,:3]
        T = w2c[:3, 3]

        # R is stored transposed due to 'glm' in CUDA code
        _R = np.transpose(R)  
        T = T

        # dirty fix
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        # image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)

        image = np.concatenate([image, mask[:, :, None]], axis=-1)

        h, w = image.shape[:2]
        fx = K[0, 0]
        fy = K[1, 1]
        cx = K[0, 2]
        cy = K[1, 2]
        # FovX = focal2fov(fx, w)
        # FovY = focal2fov(fy, h)
        # cam_info = CameraInfo(uid=idx, R=_R, T=T, FovY=FovY, FovX=FovX, image=image,
        #                       image_path=image_path, image_name=image_name, width=w, height=h)
        cam_info = CameraInfoExt(uid=idx, R=_R, T=T, image=image,
                                    image_path=image_path, image_name=image_name, width=w, height=h,
                                    fx=fx, fy=fy, cx=cx, cy=cy)

        cam_infos.append(cam_info)
        
    return cam_infos

def readInstantAvatarInfo(path, config, eval, with_ply=False, extension='.png', max_frames=3000):
    if not eval:
        train_cam_infos = readCamerasFromInstantAvatar(path, 'cameras.npz', config, eval=eval, extension=extension, max_frames=max_frames)
        test_cam_infos = []
        nerf_normalization = getNerfppNorm(train_cam_infos)
    else:
        train_cam_infos = []
        test_cam_infos = readCamerasFromInstantAvatar(path, 'cameras.npz', config, eval=eval, extension=extension, max_frames=max_frames)
        nerf_normalization = getNerfppNorm(test_cam_infos)

    if with_ply:
        ply_path = os.path.join(path, "points3d.ply")
        if not os.path.exists(ply_path):
            # Since this data set has no colmap data, we start with random points
            num_pts = 100_000
            print(f"Generating random point cloud ({num_pts})...")
            
            # We create random points inside the bounds of the synthetic Blender scenes
            xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
            shs = np.random.random((num_pts, 3)) / 255.0
            pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

            storePly(ply_path, xyz, SH2RGB(shs) * 255)
        try:
            pcd = fetchPly(ply_path)
        except:
            pcd = None
    else:
        pcd = None
        ply_path = None

    # temp_point = pcd.points
    # temp_point[:, 0] = - temp_point[:, 0]
    # temp_point = - temp_point
    # pcd = pcd._replace(points = temp_point)
    
    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

##################################################
sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender": readNerfSyntheticInfo,
    "Prometh": readPromethInfo,
    "THUman4.0": readTHUman40Info,
    "ActorsHQ": readActorsHQInfoFull,
}