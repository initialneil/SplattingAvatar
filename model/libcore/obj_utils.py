# OBJ utilities.
# Contributer(s): Neil Z. Shao
# All rights reserved. Prometheus 2022-2024.
from .transform import *

# save camera depth to obj
def saveCameraDepthToObj(fn, cam, depth):
    with open(fn, 'w') as f:
        for y in range(0, cam.h):
            for x in range(0, cam.w):
                if (depth[y][x] < 0.1):
                    continue

                d = depth[y][x]
                pt = unprojectPixelToPoint(cam, x, y, d, with_Rt=True)
                f.write('v %f %f %f\n' % (pt[0], pt[1], pt[2]))

        
