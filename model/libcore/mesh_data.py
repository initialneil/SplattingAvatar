# The mesh class.
# Contributer(s): Neil Z. Shao
# All rights reserved. Prometheus 2022-2024.
import igl
import numpy as np

class MeshCpu:
    """
    - V: vertices
    - F: faces of verts
    - N: normals
    - FN: faces of normals
    - TC: texture coordinates
    - FTC: faces of TC
    - VC: vertex colors
    """
    def __init__(self, mesh_fn=None, atlas_fn=None, use_original_N=False):
        self.V = None
        self.F = None
        self.N = None
        self.FN = None

        # texture coordinates
        self.TC = None
        self.FTC = None
        self.mtl_fn = None
        self.VC = None

        # edge normals
        # we use FE different from igl
        # igl's edge index is the opposite edge w.r.t. the vertex
        # we record the edge starting from the vertex
        self.EN = None
        self.FE = None

        if mesh_fn is not None:
            if mesh_fn.endswith('.obj'):
                self.read_obj(mesh_fn)
            elif mesh_fn.endswith('.ply'):
                self.read_ply(mesh_fn)
            else:
                print(f'skip unknown format: {mesh_fn}')

        if not use_original_N and self.V is not None and self.F is not None:
            self.update_per_vertex_normals()
        if self.N is None and self.V is not None and self.F is not None:
            self.update_per_vertex_normals()

        if self.TC is not None and self.TC.shape[1] == 3:
            print('[MeshCpu] remove 3rd column of TC')
            self.TC = self.TC[:, :2]

        self.atlas = None
        if atlas_fn is not None:
            import cv2
            self.atlas = cv2.imread(atlas_fn)

    def __repr__(self):
        return \
            '[MeshCpu] %d vertices, %d faces' % (
                self.V.shape[0] if self.V is not None else 0,
                self.F.shape[0] if self.F is not None else 0
            ) + \
            '\n[MeshCpu] %d normals, %d tcs' % (
                self.N.shape[0] if self.N is not None else 0,
                self.TC.shape[0] if self.TC is not None else 0
            )
    
    def update_per_vertex_normals(self):
        self.N = igl.per_vertex_normals(self.V, self.F)
        self.FN = self.F

    def update_edge_normals(self):
        face_normals = igl.per_face_normals(self.V, self.F, np.zeros_like(self.V))
        self.EN, E, EMAP = igl.per_edge_normals(self.V, self.F, 0, face_normals)

        self.FE = np.zeros_like(self.F)
        for i in range(self.F.shape[0]):
            for j in range(3):
                # EMAP(i + j * F.rows(), 0) is the opposite edge w.r.t. the vertex
                # we record the edge starting from the vertex
                self.FE[i, (j + 1) % 3] = EMAP[i + j * self.F.shape[0]]

    def read_obj(self, obj_fn):
        mesh = igl.read_obj(obj_fn)
        self.V, self.TC, self.N, self.F, self.FTC, self.FN = mesh
        if self.TC.shape[0] == 0:
            self.TC = None
            self.FTC = None
        if self.N.shape[0] == 0:
            self.N = None
            self.FN = None

        # VC
        if self.V is not None and self.V.shape[1] == 6:
            self.V, self.VC = self.V[:, :3], self.V[:, 3:6].astype(np.uint8)

    def read_ply(self, ply_fn):
        self.V, self.F = igl.read_triangle_mesh(ply_fn)
        self.N = None
        self.FN = None
        self.TC = None
        self.FTC = None

    def save_to_obj(self, obj_fn, mtl_fn=None, atlas_fn=None):
        import os
        if atlas_fn is not None and self.atlas is not None:
            if os.path.isabs(atlas_fn):
                fn = atlas_fn
            else:
                fn = os.path.join(os.path.dirname(obj_fn), atlas_fn)
            import cv2
            cv2.imwrite(fn, self.atlas)

            if mtl_fn is None:
                mtl_fn = self.mtl_fn
            if mtl_fn is not None:
                if os.path.isabs(mtl_fn):
                    fn = mtl_fn
                else:
                    fn = os.path.join(os.path.dirname(obj_fn), mtl_fn)
                write_mtl_fn(fn, atlas_fn)

        save_to_obj(obj_fn, self.V, self.F, N=self.N, FN=self.FN, 
            TC=self.TC, FTC=self.FTC, 
            VC=self.VC, mtl_fn=mtl_fn)

    def save_to_ply(self, ply_fn):
        from .ply_utils import PlyWriter
        ply_writer = PlyWriter()
        ply_writer.with_verts = True
        ply_writer.verts = self.V
        ply_writer.with_triangle = True
        ply_writer.tri_idxs = self.F

        with_normal = self.N is not None
        if with_normal:
            ply_writer.with_norms = True
            ply_writer.norms = self.N

        ply_writer.writeToPly(ply_fn)

    # flipYZ
    def flipYZ(self):
        self.V[:, 1] = -self.V[:, 1]
        self.V[:, 2] = -self.V[:, 2]

        if self.N is not None:
            self.N[:, 1] = -self.N[:, 1]
            self.N[:, 2] = -self.N[:, 2]

    # flip if not in opengl coordinates
    def flipToOpenGL(self):
        c = self.V[:, 1].mean()
        if (c < 0):
            self.flipYZ()
            
    # flip if not in opencv coordinates
    def flipToOpenCV(self):
        c = self.V[:, 1].mean()
        if (c > 0):
            self.flipYZ()

def write_mtl_fn(mtl_fn, atlas_fn):
    with open(mtl_fn, 'w') as f:
        f.write('newmtl initialShadingGroup\n')
        f.write('illum 4\n')
        f.write('Kd 0.00 0.00 0.00\n')
        f.write('Ka 0.00 0.00 0.00\n')
        f.write('Tf 1.00 1.00 1.00\n')
        f.write(f'map_Kd {atlas_fn}\n')
        f.write('Ni 1.00\n')
      
def save_to_obj(obj_fn, V, F=None, N=None, FN=None, TC=None, FTC=None, VC=None, mtl_fn=None):
    with open(obj_fn, 'w') as f:
        if FN is None and N is not None:
            if V.shape == N.shape and F is not None:
                FN = F

        with_f = F is not None
        with_normal = ((N is not None) & (FN is not None))
        with_tc = ((TC is not None) and (FTC is not None) and (FTC.shape == F.shape))
        with_mtl = (mtl_fn is not None)

        f.write('# %d vertices\n' % (V.shape[0]))

        if with_normal:
            f.write('# %d normals\n' % (N.shape[0]))
        else:
            f.write('# %d normals\n' % (0))

        if with_tc:
            f.write('# %d texture coordinates\n' % (TC.shape[0]))
        else:
            f.write('# %d texture coordinates\n' % (0))
        
        if with_f:
            f.write('# %d triangles\n' % (F.shape[0]))
        else:
            f.write('# %d triangles\n' % (0))

        if with_mtl:
            f.write('mtllib %s\n' % (mtl_fn))
            f.write('g default\n')

        # v
        if VC is not None and VC.shape == V.shape:
            for i in range(V.shape[0]):
                f.write('v %f %f %f %d %d %d\n' % (V[i, 0], V[i, 1], V[i, 2], VC[i, 0], VC[i, 1], VC[i, 2]))
        else:
            for i in range(V.shape[0]):
                f.write('v %f %f %f\n' % (V[i, 0], V[i, 1], V[i, 2]))

        # vn
        if with_normal:
            for i in range(N.shape[0]):
                f.write('vn %f %f %f\n' % (N[i, 0], N[i, 1], N[i, 2]))

        # vt
        if with_tc:
            for i in range(TC.shape[0]):
                f.write('vt %f %f\n' % (TC[i, 0], TC[i, 1]))

        # # mtl
        # if with_mtl:
        #     f.write('s off\n')
        #     f.write('g _Mesh\n')
        #     f.write('usemtl initialShadingGroup\n')

        # f
        if with_f:
            if (not with_normal) and (not with_tc):
                for i in range(F.shape[0]):
                    f.write('f %d %d %d\n' % (F[i, 0] + 1, F[i, 1] + 1, F[i, 2] + 1))

            elif (with_normal) and (not with_tc):
                for i in range(F.shape[0]):
                    f.write('f %d//%d %d//%d %d//%d\n' % (
                        F[i, 0] + 1, FN[i, 0] + 1, F[i, 1] + 1, FN[i, 1] + 1, F[i, 2] + 1, FN[i, 2] + 1))

            elif (not with_normal) and (with_tc):
                for i in range(F.shape[0]):
                    f.write('f %d/%d %d/%d %d/%d\n' % (
                        F[i, 0] + 1, FTC[i, 0] + 1, F[i, 1] + 1, FTC[i, 1] + 1, F[i, 2] + 1, FTC[i, 2] + 1))

            elif (with_normal) and (with_tc):
                for i in range(F.shape[0]):
                    f.write('f %d/%d/%d %d/%d/%d %d/%d/%d\n' % (
                        F[i, 0] + 1, FTC[i, 0] + 1, FN[i, 0] + 1, 
                        F[i, 1] + 1, FTC[i, 1] + 1, FN[i, 1] + 1, 
                        F[i, 2] + 1, FTC[i, 2] + 1, FN[i, 2] + 1))

##################################################
try:
    import torch
    _with_torch = True
except:
    _with_torch = False

if _with_torch:
    def sample_bary_on_triangles(num_faces, num_samples):
        sample_bary = torch.zeros(num_samples, 3)
        sample_bary[:, 0] = torch.rand(num_samples)
        sample_bary[:, 1] = torch.rand(num_samples) * (1.0 - sample_bary[:, 0])
        sample_bary[:, 2] = 1.0 - sample_bary[:, 0] - sample_bary[:, 1]
        sample_fidxs = torch.randint(0, num_faces, size=(num_samples,))
        return sample_fidxs, sample_bary

    def retrieve_verts_barycentric(vertices, faces, fidxs, barys):
        triangle_verts = vertices[faces].float()

        if len(triangle_verts.shape) == 3:
            sample_verts = torch.einsum('nij,ni->nj', triangle_verts[fidxs], barys)
        elif len(triangle_verts.shape) == 4:
            sample_verts = torch.einsum('bnij,ni->bnj', triangle_verts[:, fidxs, ...], barys)
        else:
            raise NotImplementedError
        
        return sample_verts

##################################################
def convert_tetgen_to_meshcpu(fn, tet_verts, tet_faces):
    if isinstance(tet_verts, torch.Tensor):
        tet_verts = tet_verts.cpu()
    if isinstance(tet_faces, torch.Tensor):
        tet_faces = tet_faces.cpu()

    mesh = MeshCpu()
    mesh.V = tet_verts
    if isinstance(tet_faces, np.ndarray):
        mesh.F = np.stack([
            tet_faces[:, 0], tet_faces[:, 1], tet_faces[:, 2],
            tet_faces[:, 2], tet_faces[:, 1], tet_faces[:, 3],
            tet_faces[:, 3], tet_faces[:, 1], tet_faces[:, 0],
            tet_faces[:, 0], tet_faces[:, 2], tet_faces[:, 3],
        ]).transpose().reshape(-1, 3)
    
    mesh.save_to_obj(fn)
