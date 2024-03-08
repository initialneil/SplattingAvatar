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

from scene.cameras import Camera
import numpy as np
from utils.general_utils import PILtoTorch, numpyToTorch
from utils.graphics_utils import fov2focal

WARNED = False

def loadCam(args, id, cam_info, resolution_scale, verbose_timer=False):
    if isinstance(cam_info.image, np.ndarray):
        orig_h, orig_w = cam_info.image.shape[:2]
    else:
        orig_w, orig_h = cam_info.image.size

    if args.resolution in [1, 2, 4, 8]:
        resolution = round(orig_w/(resolution_scale * args.resolution)), round(orig_h/(resolution_scale * args.resolution))
    else:  # should be a type that converts to float
        if args.resolution == -1:
            if orig_w > 1600:
                global WARNED
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                        "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    WARNED = True
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))

    if verbose_timer:
        from model import libcore
        libcore.startCpuTimer('[loadCam] resize')

    if isinstance(cam_info.image, np.ndarray):
        resized_image_rgb = numpyToTorch(cam_info.image, resolution)
    else:
        # PIL is very slow
        # resized_image_rgb = PILtoTorch(cam_info.image, resolution)
        # use cv2 instead
        image = np.array(cam_info.image)
        resized_image_rgb = numpyToTorch(image, resolution)

    if verbose_timer:
        libcore.stopCpuTimer('[loadCam] resize')

    gt_image = resized_image_rgb[:3, ...]
    loaded_mask = None

    if resized_image_rgb.shape[0] == 4:
        loaded_mask = resized_image_rgb[3:4, ...]

    if hasattr(cam_info, 'FovX'):
        return Camera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T, 
                      FoVx=cam_info.FovX, FoVy=cam_info.FovY, 
                      image=gt_image, gt_alpha_mask=loaded_mask,
                      image_name=cam_info.image_name, uid=id, data_device=args.data_device)
    else:
        return Camera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T, 
                      w=cam_info.width, h=cam_info.height, fx=cam_info.fx, fy=cam_info.fy, cx=cam_info.cx, cy=cam_info.cy,
                      image=gt_image, gt_alpha_mask=loaded_mask,
                      image_name=cam_info.image_name, uid=id, data_device=args.data_device)

def cameraList_from_camInfos(cam_infos, resolution_scale, args):
    camera_list = []

    verbose_timer = False
    if verbose_timer:
        from model import libcore

    # func_args = []
    # for id, cam_info in enumerate(cam_infos):
    #     params = { 'args': args, 'id': id, 'cam_info': cam_info, 'resolution_scale': resolution_scale }
    #     func_args.append(params)

    # import concurrent.futures
    # with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executrer:
    #     for res in executrer.map(lambda params: loadCam(**params), func_args):
    #         camera_list.append(res)

    for id, c in enumerate(cam_infos):
        if verbose_timer:
            libcore.startCpuTimer(loadCam)
        camera_list.append(loadCam(args, id, c, resolution_scale, verbose_timer=verbose_timer))
        if verbose_timer:
            libcore.stopCpuTimer(loadCam)

    return camera_list

def camera_to_JSON(id, camera : Camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    if hasattr(camera, 'FovX'):
        camera_entry = {
            'id' : id,
            'img_name' : camera.image_name,
            'width' : camera.width,
            'height' : camera.height,
            'position': pos.tolist(),
            'rotation': serializable_array_2d,
            'fy' : fov2focal(camera.FovY, camera.height),
            'fx' : fov2focal(camera.FovX, camera.width)
        }
    else:
        camera_entry = {
            'id' : id,
            'img_name' : camera.image_name,
            'width' : camera.width,
            'height' : camera.height,
            'position': pos.tolist(),
            'rotation': serializable_array_2d,
            'fy' : camera.fy,
            'fx' : camera.fx,
            'cy' : camera.cy,
            'cx' : camera.cx,
        }
    return camera_entry
