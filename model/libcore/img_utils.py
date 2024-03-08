# Image processing utils.
# Contributer(s): Neil Z. Shao
# All rights reserved. Prometheus 2022-2024.
import numpy as np
import cv2
import copy

def readFloat2FromPng(fn, scale=10000.0, shift=32768.0):
    value_map = cv2.imread(fn, cv2.IMREAD_UNCHANGED)
    value_map = (value_map.astype(float) - shift) / scale
    h = int(value_map.shape[0])
    w = int(value_map.shape[1] / 2)
    wF = w * 2

    map2f = np.stack([value_map[0:h, 0:w], value_map[0:h, w:wF]], axis=2)
    return map2f

def writeFloat2ToPng(fn, value_map, scale=10000.0, shift=32768.0):
    value_map = (value_map * scale + shift).astype(np.uint16)
    h = int(value_map.shape[0])
    w = int(value_map.shape[1])

    map2f = np.concatenate([value_map[:, :, 0], value_map[:, :, 1]], axis=1)
    cv2.imwrite(fn, map2f)

def readFloat4FromPng(fn):
    value_map = cv2.imread(fn, cv2.IMREAD_UNCHANGED)
    value_map = (value_map.astype(float) - 32768.0) / 10000.0
    h = int(value_map.shape[0] / 2)
    w = int(value_map.shape[1] / 2)
    hF = h * 2
    wF = w * 2

    map4f = np.stack([value_map[0:h, 0:w], value_map[h:hF, 0:w], value_map[0:h, w:wF], value_map[h:hF, w:wF]], axis=2)
    return map4f

def writeNMapToUchar3(nmap_fn, nmap):
    nmap_clr = (nmap * 127 + 128).astype(np.uint8)
    cv2.imwrite(nmap_fn, nmap_clr)
    
def detachToNumpy(img):
    if len(img.shape) == 4:
        img = img.squeeze(0)
    return (img.detach().cpu() * 255.0).numpy().astype(np.uint8)

# normalize to
def colorizeWeightsMap(weights, colormap=cv2.COLORMAP_JET, 
                       min_val=None, max_val=None, 
                       to_rgb=False):
    if min_val is None:
        min_val = weights.min()
    if max_val is None:
        max_val = weights.max()

    vals = (weights - min_val) / (max_val - min_val)
    vals = (vals.clip(0, 1) * 255).astype(np.uint8)
    canvas = cv2.applyColorMap(vals, colormap=colormap)
    if to_rgb:
        return canvas[..., [2, 1, 0]]
    else:
        return canvas

# display
def cvshow(img, max_width=1920, title='image'):
    # tensor or ndarray
    if not isinstance(img, np.ndarray):
        img = img.detach().float().cpu().squeeze().numpy()

    if img.dtype == float:
        if len(img.shape) == 2:
            img = colorizeWeightsMap(img)
        else:
            img = (img * 255.0).clip(0, 255).astype(np.uint8)
    

    if max_width > 0 and img.shape[1] > max_width:
        scale = float(max_width) / img.shape[1]
        img = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)

    cv2.imshow(title, img)
    cv2.waitKey(100)

# tensor to image
def write_tensor_image(fn, tensor, rgb2bgr=False):
    if len(tensor.shape) == 3:
        if tensor.shape[0] == 3 or tensor.shape[0] == 4:
            tensor = tensor.permute([1, 2, 0])

    if rgb2bgr:
        if tensor.shape[2] == 3:
            tensor = tensor[:, :, [2, 1, 0]]
        else:
            tensor = tensor[:, :, [2, 1, 0, 3]]
    
    cv2.imwrite(fn, (tensor.clamp(0, 1) * 255).detach().cpu().numpy().astype(np.uint8))

# draw points
def draw_pixel_points(img, pixels, radius=3, color=None, thickness=0, 
                      fontFace=0, fontScale=1.0,
                      text_start_number=None):
    canvas = copy.deepcopy(img)
    for i in range(pixels.shape[0]):
        if color is not None:
            clr = color
        else:
            clr = (np.random.rand(3) * 256).astype(int).tolist()
        cv2.circle(canvas, pixels[i].astype(int), radius, clr, thickness=thickness)
        if text_start_number is not None:
            cv2.putText(canvas, str(i + text_start_number), (pixels[i] + 10).astype(int), 
                        fontFace, fontScale, clr)
    return canvas
    
# draw pairs
def draw_pixel_pairs(img, pxls0, pxls1, pxls0_color=[0, 0, 255], pxls1_color=[255, 0, 0], 
                     line_color=[255, 255, 255], thickness=1):
    canvas = copy.deepcopy(img)
    for i in range(pxls0.shape[0]):
        cv2.circle(canvas, pxls0[i].astype(int), 1, pxls0_color, thickness=thickness)
        cv2.circle(canvas, pxls1[i].astype(int), 1, pxls1_color, thickness=thickness)
        cv2.line(canvas, pxls0[i].astype(int), pxls1[i].astype(int), line_color, thickness=thickness)
    return canvas
    
