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

import torch
import traceback
import socket
import json
import numpy as np
from scene.cameras import MiniCam

host = "127.0.0.1"
port = 6009

conn = None
addr = None

listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

def init(wish_host, wish_port):
    global host, port, listener
    host = wish_host
    port = wish_port
    listener.bind((host, port))
    listener.listen()
    listener.settimeout(0)

def try_connect():
    global conn, addr, listener
    try:
        conn, addr = listener.accept()
        print(f"\nConnected by {addr}")
        conn.settimeout(None)
    except Exception as inst:
        pass
            
def read():
    global conn
    messageLength = conn.recv(4)
    messageLength = int.from_bytes(messageLength, 'little')
    message = conn.recv(messageLength)
    return json.loads(message.decode("utf-8"))

def send(message_bytes, verify):
    global conn
    if message_bytes != None:
        conn.sendall(message_bytes)
    conn.sendall(len(verify).to_bytes(4, 'little'))
    conn.sendall(bytes(verify, 'ascii'))

def receive():
    message = read()

    width = message["resolution_x"]
    height = message["resolution_y"]

    if width != 0 and height != 0:
        try:
            do_training = bool(message["train"])
            fovy = message["fov_y"]
            fovx = message["fov_x"]
            znear = message["z_near"]
            zfar = message["z_far"]
            do_shs_python = bool(message["shs_python"])
            do_rot_scale_python = bool(message["rot_scale_python"])
            keep_alive = bool(message["keep_alive"])
            scaling_modifier = message["scaling_modifier"]
            world_view_transform = torch.reshape(torch.tensor(message["view_matrix"]), (4, 4)).cuda()
            world_view_transform[:,1] = -world_view_transform[:,1]
            world_view_transform[:,2] = -world_view_transform[:,2]
            full_proj_transform = torch.reshape(torch.tensor(message["view_projection_matrix"]), (4, 4)).cuda()
            full_proj_transform[:,1] = -full_proj_transform[:,1]
            custom_cam = MiniCam(width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform)
        except Exception as e:
            print("")
            traceback.print_exc()
            raise e
        return custom_cam, do_training, do_shs_python, do_rot_scale_python, keep_alive, scaling_modifier
    else:
        return None, None, None, None, None, None

########## routine ##########
def send_image_to_network(image, verify):
    global conn
    if conn == None:
        try_connect()
    if conn != None:
        try:
            custom_cam, do_training, do_shs_python, _, _, scaling_modifer = receive()

            net_image = torch.zeros((3, custom_cam.image_height, custom_cam.image_width))
            if image.shape[1] > net_image.shape[1] or image.shape[2] > net_image.shape[2]:
                step = max(image.shape[1] / net_image.shape[1], image.shape[2] / net_image.shape[2])
                step = max(int(np.ceil(step)), 1)
                image = image[:3, ::step, ::step]
            top = (net_image.shape[1] - image.shape[1]) // 2
            left = (net_image.shape[2] - image.shape[2]) // 2
            net_image[:, top:top+image.shape[1], left:left+image.shape[2]] = image

            net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
            send(net_image_bytes, verify)
        except Exception as e: 
            conn = None

def render_to_network(model, pipe, verify, gt_image=None):
    do_training = True
    global conn

    if conn == None:
        try_connect()
    if conn != None:
        try:
            custom_cam, do_training, do_shs_python, _, _, scaling_modifer = receive()

            with torch.no_grad():
                net_image = model.render_to_camera(custom_cam, pipe, background='white',
                                                    scaling_modifer=scaling_modifer)["render"]
                
            if gt_image is not None:
                step = int(max(max(gt_image.shape[1] / 200, gt_image.shape[2] / 200), 1))
                img = gt_image[:, ::step, ::step]
                net_image[:, :img.shape[1], :img.shape[2]] = img

            net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
            send(net_image_bytes, verify)

        except Exception as e: 
            conn = None
    
    return do_training
