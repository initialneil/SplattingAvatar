# PLY utilities.
# Contributer(s): Neil Z. Shao
# All rights reserved. Prometheus 2022-2024.
from cv2 import polylines
from .transform import *
import os
import cv2

try:
    import igl
except ImportError:
    print('[Warning] import igl failed.')
    print('[Warning] Please install igl with conda install -c conda-forge igl')
    print('[Warning] https://github.com/libigl/libigl-python-bindings')

class PlyWriter:   
    def __init__(self) -> None:
        # vertex
        self.verts = []
        self.norms = []
        self.colors = []

        # edge index list
        self.edge_idxs = []
        self.edge_colors = []

        # triangle index list
        self.tri_idxs = []

        # write options
        self.with_verts = False
        self.with_norms = False
        self.with_color = False
        self.with_edge = False
        self.with_triangle = False

    # add
    def addVertex(self, vt):
        self.verts.append(vt)
        self.with_verts = True

    def addNormal(self, nm):
        self.norms.append(nm)
        self.with_norms = True

    def addColor(self, clr):
        self.colors.append(clr)
        self.with_color = True

    def addTriangle(self, idx0, idx1, idx2, tri_clr=None):
        self.tri_idxs.append([idx0, idx1, idx2])
        self.with_triangle = True

    def addEdge(self, vt0, vt1, edge_clr=None):
        vt_idx = len(self.verts)
        self.addVertex(vt0)
        self.addVertex(vt1)
        self.addEdgeByIdx(vt_idx, vt_idx + 1, edge_clr)
        self.with_edge = True

    def addEdgeByIdx(self, idx0, idx1, edge_clr=None):
        self.edge_idxs.append(np.array([idx0, idx1]).astype(int))
        self.edge_colors.append(edge_clr)
        self.with_edge = True

    def addBox(self, origin, box_sz, R=None, edge_clr=None):
        edge_x = np.array([box_sz[0], 0, 0])
        edge_y = np.array([0, box_sz[1], 0])
        edge_z = np.array([0, 0, box_sz[2]])
        
        self.addEdge(origin, origin + edge_x, edge_clr)
        self.addEdge(origin, origin + edge_y, edge_clr)
        self.addEdge(origin, origin + edge_z, edge_clr)
        self.addEdge(origin + edge_x, origin + edge_x + edge_y, edge_clr)
        self.addEdge(origin + edge_x, origin + edge_x + edge_z, edge_clr)
        self.addEdge(origin + edge_y, origin + edge_y + edge_x, edge_clr)
        self.addEdge(origin + edge_y, origin + edge_y + edge_z, edge_clr)
        self.addEdge(origin + edge_z, origin + edge_z + edge_x, edge_clr)
        self.addEdge(origin + edge_z, origin + edge_z + edge_y, edge_clr)
        self.addEdge(origin + edge_x + edge_y, origin + edge_x + edge_y + edge_z, edge_clr)
        self.addEdge(origin + edge_x + edge_z, origin + edge_x + edge_y + edge_z, edge_clr)
        self.addEdge(origin + edge_y + edge_z, origin + edge_x + edge_y + edge_z, edge_clr)

    def addRotatedBox(self, box_sz, origin=None, center=None, R=None, edge_clr=None):
        if origin is None and center is None:
            print('[addRotatedBox] either origin or center is needed')
            return
        if R is None:
            R = np.eye(3)

        edge_x = R[0, :] * box_sz[0]
        edge_y = R[1, :] * box_sz[1]
        edge_z = R[2, :] * box_sz[2]

        if origin is None:
            origin = center - edge_x / 2.0 - edge_y / 2.0 - edge_z / 2.0
        
        self.addEdge(origin, origin + edge_x, edge_clr)
        self.addEdge(origin, origin + edge_y, edge_clr)
        self.addEdge(origin, origin + edge_z, edge_clr)
        self.addEdge(origin + edge_x, origin + edge_x + edge_y, edge_clr)
        self.addEdge(origin + edge_x, origin + edge_x + edge_z, edge_clr)
        self.addEdge(origin + edge_y, origin + edge_y + edge_x, edge_clr)
        self.addEdge(origin + edge_y, origin + edge_y + edge_z, edge_clr)
        self.addEdge(origin + edge_z, origin + edge_z + edge_x, edge_clr)
        self.addEdge(origin + edge_z, origin + edge_z + edge_y, edge_clr)
        self.addEdge(origin + edge_x + edge_y, origin + edge_x + edge_y + edge_z, edge_clr)
        self.addEdge(origin + edge_x + edge_z, origin + edge_x + edge_y + edge_z, edge_clr)
        self.addEdge(origin + edge_y + edge_z, origin + edge_x + edge_y + edge_z, edge_clr)

    def writeToPly(self, fn, format='binary'):
        with open(fn, 'wb') as f:
            f.write('ply\n'.encode('ascii'))

            if format == 'ascii':
                f.write('format ascii 1.0\n'.encode('ascii'))
            else:
                f.write('format binary_little_endian 1.0\n'.encode('ascii'))

            # vertex header
            if self.with_verts:
                f.write(('element vertex %d\n' % len(self.verts)).encode('ascii'))
                f.write('property float x\n'.encode('ascii'))
                f.write('property float y\n'.encode('ascii'))
                f.write('property float z\n'.encode('ascii'))

                if self.with_norms:
                    f.write('property float nx\n'.encode('ascii'))
                    f.write('property float ny\n'.encode('ascii'))
                    f.write('property float nz\n'.encode('ascii'))

                if self.with_color:
                    f.write('property uchar red\n'.encode('ascii'))
                    f.write('property uchar green\n'.encode('ascii'))
                    f.write('property uchar blue\n'.encode('ascii'))
            
            # edge header
            if self.with_edge:
                f.write(('element edge %d\n' % len(self.edge_idxs)).encode('ascii'))
                f.write('property int vertex1\n'.encode('ascii'))
                f.write('property int vertex2\n'.encode('ascii'))

                f.write('property uchar red\n'.encode('ascii'))
                f.write('property uchar green\n'.encode('ascii'))
                f.write('property uchar blue\n'.encode('ascii'))

            # triangle header
            if self.with_triangle:
                f.write(('element face %d\n' % len(self.tri_idxs)).encode('ascii'))
                f.write('property list uchar int vertex_indices\n'.encode('ascii'))

                # f.write('property uchar red\n'.encode('ascii'))
                # f.write('property uchar green\n'.encode('ascii'))
                # f.write('property uchar blue\n'.encode('ascii'))

            f.write('end_header\n'.encode('ascii'))

            # bytes io
            if format != 'ascii':
                from io import BytesIO
                bytes_io = BytesIO()

            # vertex
            if self.with_verts:
                for i in range(0, len(self.verts)):
                    vt = self.verts[i]
                    if format == 'ascii':
                        f.write(('%f %f %f ' % (vt[0], vt[1], vt[2])).encode('ascii'))
                    else:
                        bytes_io.write(np.array([vt[0], vt[1], vt[2]]).astype(np.float32).tobytes())

                    if self.with_norms:
                        nm = self.norms[i]
                        if format == 'ascii':
                            f.write(('%f %f %f ' % (nm[0], nm[1], nm[2])).encode('ascii'))
                        else:
                            bytes_io.write(np.array([nm[0], nm[1], nm[2]]).astype(np.float32).tobytes())

                    if self.with_color:
                        clr = self.colors[i]
                        if format == 'ascii':
                            f.write(('%d %d %d ' % (clr[0], clr[1], clr[2])).encode('ascii'))
                        else:
                            bytes_io.write(np.array([clr[0], clr[1], clr[2]]).astype(np.uint8).tobytes())

                    if format == 'ascii':
                        f.write(('\n').encode('ascii'))

            # edge
            if self.with_edge:
                for i in range(len(self.edge_idxs)):
                    idx0 = self.edge_idxs[i][0]
                    idx1 = self.edge_idxs[i][1]
                    r, g, b = 255, 255, 255
                    if (len(self.edge_colors) > i and self.edge_colors[i] is not None):
                        r, g, b = self.edge_colors[i]
                    if format == 'ascii':
                        f.write(('%d %d %d %d %d\n' % (idx0, idx1, r, g, b)).encode('ascii'))
                    else:
                        bytes_io.write(np.array([idx0, idx1]).astype(np.int32).tobytes())
                        bytes_io.write(np.array([r, g, b]).astype(np.uint8).tobytes())

            # triangle
            if self.with_triangle:
                for i in range(len(self.tri_idxs)):
                    idx0 = self.tri_idxs[i][0]
                    idx1 = self.tri_idxs[i][1]
                    idx2 = self.tri_idxs[i][2]
                    if format == 'ascii':
                        f.write(('%d %d %d %d\n' % (3, idx0, idx1, idx2)).encode('ascii'))
                    else:
                        bytes_io.write(np.array([3]).astype(np.uint8).tobytes())
                        bytes_io.write(np.array([idx0, idx1, idx2]).astype(np.int32).tobytes())

            if format != 'ascii':
                f.write(bytes_io.getbuffer())
                bytes_io.close()


# save camera depth to obj
def saveCameraDepthToPly(fn, cam, depth, color=None, base_width=None):
    if base_width is not None:
        cam = cam.clone()
        cam.scaleIntrinsicsBaseWidth(base_width)
        color = cv2.resize(color, dsize=cam.sz, interpolation=cv2.INTER_CUBIC)
        depth = cv2.resize(depth, dsize=cam.sz, interpolation=cv2.INTER_NEAREST)

    # vertex, normal map
    print('[saveCameraDepthToPly] calc vmap, nmap')
    vmap, nmap = calcVNMap(cam, depth)

    saveVNmapToPly(fn, vmap, nmap, color, tag='saveCameraDepthToPly')

def saveVNmapToPly(fn, vmap, nmap, color, tag='saveVNmapToPly'):
    print('[%s] prepare ply writer' % tag)
    ply_writer = PlyWriter()
    for y in range(0, vmap.shape[0]):
        for x in range(0, vmap.shape[1]):
            if (isnan(vmap[y, x][0]) or isnan(nmap[y, x][0])):
                continue

            ply_writer.addVertex(vmap[y, x])
            ply_writer.addNormal(nmap[y, x])

            if color is not None:
                clr = color[y, x].astype(np.int8)
                clr = np.array([clr[2], clr[1], clr[0]], dtype=np.uint8)
                ply_writer.addColor(clr)

    print('[%s] write ply' % tag)
    ply_writer.writeToPly(fn)

def saveCamerasToPly(fn, cams, finger_lens=0.1):
    ply_writer = PlyWriter()
    for i in range(len(cams)):
        c = cams[i].c
        ply_writer.addVertex(c)
        ply_writer.addColor(np.array([255, 255, 255]))

        finger_clrs = np.array([
            [255, 0, 0],
            [0, 255, 0],
            [0, 0, 255]])

        for j in range(0, 3):
            pt = c + cams[i].R[j, :] * finger_lens
            ply_writer.addVertex(pt)
            ply_writer.addColor(finger_clrs[j, :])

            idx0 = i * 4
            idx1 = idx0 + 1 + j
            ply_writer.addEdgeByIdx(idx0, idx1)
    
    # print('[saveCamerasToPly] write ply')
    ply_writer.writeToPly(fn)

def saveLinePointsToPly(fn, pts, nms=None, clr_mask=None):
    if pts is None or pts.shape[0] == 0:
        return
    
    ply_writer = PlyWriter()
    ply_writer.addVertex(pts[0])

    for i in range(1, pts.shape[0]):
        ply_writer.addVertex(pts[i])
        ply_writer.addEdgeByIdx(i - 1, i)

    if nms is not None:
        for i in range(0, nms.shape[0]):
            ply_writer.addNormal(nms[i])

    if clr_mask is not None:
        for i in range(clr_mask.shape[0]):
            if clr_mask[i]:
                ply_writer.addColor(np.array([255, 255, 255]))
            else:
                ply_writer.addColor(np.array([0, 0, 255]))

    # print('[saveCamerasToPly] write ply')
    ply_writer.writeToPly(fn)

def savePointsToPly(fn, verts, norms=None, clr_mask=None, colors=None):
    if verts is None or verts.shape[0] == 0:
        return
    
    if len(verts.shape) == 3:
        verts = verts[0]

    try:
        import torch
        if isinstance(verts, torch.Tensor):
            verts = verts.detach().cpu()
    except:
        pass
    
    ply_writer = PlyWriter()
    for i in range(verts.shape[0]):
        ply_writer.addVertex(verts[i])
        if norms is not None and norms.shape == verts.shape:
            ply_writer.addNormal(norms[i])

    if clr_mask is not None:
        for i in range(clr_mask.shape[0]):
            if clr_mask[i]:
                ply_writer.addColor(np.array([255, 255, 255]))
            else:
                ply_writer.addColor(np.array([0, 0, 255]))

    if colors is not None:
        for i in range(colors.shape[0]):
            ply_writer.addColor(colors[i])

    # print('[saveCamerasToPly] write ply')
    ply_writer.writeToPly(fn)

def savePairedPointsToPly(fn, pts0, pts1, nml0=None, nml1=None):
    if pts0 is None or pts1 is None or pts0.shape[0] == 0 or pts0.shape[0] != pts1.shape[0]:
        return
    
    ply_writer = PlyWriter()

    for i in range(pts0.shape[0]):
        ply_writer.addVertex(pts0[i])
        ply_writer.addVertex(pts1[i])
        ply_writer.addEdgeByIdx(i * 2, i * 2 + 1)

        ply_writer.addColor(np.array([255, 0, 0]))
        ply_writer.addColor(np.array([0, 0, 255]))

        if nml0 is not None and nml1 is not None:
            ply_writer.addNormal(nml0[i])
            ply_writer.addNormal(nml1[i])

    # print('[saveCamerasToPly] write ply')
    ply_writer.writeToPly(fn)
