import taichi as ti
from datatypes import vec3f
from constants import *
import numpy as np
import math
import trimesh as tm
from typing import Optional

@ti.dataclass
class Material:
    albedo: vec3f # type: ignore
    emissive: vec3f # type: ignore
    shininess: ti.f32 # type: ignore
    roughness: ti.f32
    bsdf_type: ti.i32 # type: ignore

def create_lambert(albedo, emissive=(0, 0, 0)):
    return Material(
        albedo=ti.Vector(list(albedo)),
        emissive=ti.Vector(list(emissive)),
        shininess=0.0,
        roughness=0.0,
        bsdf_type=BSDF_LAMBERT
    )

def create_phong(albedo, shininess, emissive=(0, 0, 0)):
    return Material(
        albedo=ti.Vector(list(albedo)),
        emissive=ti.Vector(list(emissive)),
        shininess=shininess,
        roughness=0.0,
        bsdf_type=BSDF_PHONG
    )

def create_ggx(albedo, roughness, emissive=(0, 0, 0)):
    return Material(
        albedo=ti.Vector(list(albedo)),
        emissive=ti.Vector(list(emissive)),
        shininess=0.0,
        roughness=roughness,
        bsdf_type=BSDF_GGX
    )

@ti.dataclass
class Sphere:
    center: vec3f # type: ignore
    radius: ti.f32 # type: ignore
    material_id: ti.i32 # type: ignore

@ti.dataclass
class Plane:
    point: vec3f # type: ignore
    normal: vec3f # type: ignore
    material_id: ti.i32 # type: ignore

@ti.dataclass
class Triangle:
    v0: vec3f #type:ignore
    v1: vec3f #type:ignore
    v2: vec3f #type:ignore
    normal: vec3f  # optional precomputed #type:ignore
    n0: vec3f #type:ignore
    n1: vec3f #type:ignore
    n2: vec3f #type:ignore
    material_id: ti.i32 #type:ignore

@ti.dataclass
class PointLight:
    position: vec3f # type: ignore
    color: vec3f # type: ignore

@ti.dataclass
class BVHNode:
    is_leaf: ti.i32 # type: ignore
    left_or_start: ti.i32 # type: ignore
    right_or_count: ti.i32 # type: ignore
    aabb_min: vec3f # type: ignore
    aabb_max: vec3f # type: ignore
    depth: ti.i32 # type: ignore

@ti.dataclass
class BVHInfo:
    triangle_offset: ti.i32 # type: ignore
    num_nodes: ti.i32 # type: ignore
    transform: ti.types.matrix(4, 4, ti.f32) # type: ignore
    inv_transform: ti.types.matrix(4, 4, ti.f32) # type: ignore

#upload_free_triangles(self.free_triangles, free_tri_offset, Nt, triangle_ids)
@ti.kernel
def upload_free_triangles(
    free_triangles: ti.template(),   # type: ignore
    offset: ti.i32, count: ti.i32,  # type: ignore 
    triangle_ids: ti.types.ndarray(), # type: ignore
):
    for i in range(count):
        j = offset + i
        free_triangles[j] = triangle_ids[i]

@ti.kernel
def upload_triangles(
    triangles: ti.template(),   # type: ignore
    offset: ti.i32, count: ti.i32,  # type: ignore 
    v0: ti.types.ndarray(), # type: ignore
    v1: ti.types.ndarray(), # type: ignore
    v2: ti.types.ndarray(), # type: ignore
    n: ti.types.ndarray(), # type: ignore
    n0: ti.types.ndarray(), # type: ignore
    n1: ti.types.ndarray(), # type: ignore
    n2: ti.types.ndarray(), # type: ignore
    material_id: ti.types.ndarray() # type: ignore
):
    for i in range(count):
        j = offset + i
        triangles.v0[j] = ti.Vector([v0[i, 0], v0[i, 1], v0[i, 2]], dt=ti.f32)
        triangles.v1[j] = ti.Vector([v1[i, 0], v1[i, 1], v1[i, 2]], dt=ti.f32)
        triangles.v2[j] = ti.Vector([v2[i, 0], v2[i, 1], v2[i, 2]], dt=ti.f32)
        triangles.normal[j] = ti.Vector([n[i, 0], n[i, 1], n[i, 2]], dt=ti.f32)
        triangles.n0[j] = ti.Vector([n0[i, 0], n0[i, 1], n0[i, 2]], dt=ti.f32)
        triangles.n1[j] = ti.Vector([n1[i, 0], n1[i, 1], n1[i, 2]], dt=ti.f32)
        triangles.n2[j] = ti.Vector([n2[i, 0], n2[i, 1], n2[i, 2]], dt=ti.f32)
        triangles.material_id[j] = material_id[i]

@ti.kernel
def upload_bvh(
    bvhs: ti.template(),   # type: ignore
    bvh_id: ti.i32,  # type: ignore
    node_count: ti.i32, # type: ignore
    is_leaf: ti.types.ndarray(), # type: ignore
    left_or_start: ti.types.ndarray(), # type: ignore
    right_or_count: ti.types.ndarray(), # type: ignore
    aabb_min: ti.types.ndarray(), # type: ignore
    aabb_max: ti.types.ndarray(), # type: ignore
    depth: ti.types.ndarray(), # type: ignore
):
    for i in range(node_count):
        bvhs.is_leaf[bvh_id, i] = is_leaf[i]
        bvhs.left_or_start[bvh_id, i] = left_or_start[i]
        bvhs.right_or_count[bvh_id, i] = right_or_count[i]
        bvhs.aabb_min[bvh_id, i] = ti.Vector([aabb_min[i, 0], aabb_min[i, 1], aabb_min[i, 2]], dt=ti.f32)
        bvhs.aabb_max[bvh_id, i] = ti.Vector([aabb_max[i, 0], aabb_max[i, 1], aabb_max[i, 2]], dt=ti.f32)
        bvhs.depth[bvh_id, i] = depth[i]


class Scene:
    def __init__(self):
        # existing
        self.num_spheres = ti.field(dtype=ti.i32, shape=())
        self.spheres = Sphere.field(shape=MAX_SPHERES)
        self.num_triangles = ti.field(dtype=ti.i32, shape=())
        self.triangles = Triangle.field(shape=MAX_TRIANGLES)
        self.num_free_triangles = ti.field(dtype=ti.i32, shape=())
        self.free_triangles = ti.field(dtype=ti.i32, shape=(MAX_TRIANGLES))

        self.num_planes = ti.field(dtype=ti.i32, shape=())
        self.planes = Plane.field(shape=MAX_PLANES)
        self.materials = Material.field(shape=MAX_MATERIALS)
        self.num_materials = ti.field(dtype=ti.i32, shape=())
        self.num_light_spheres = ti.field(dtype=ti.i32, shape=())
        self.light_spheres_id = ti.field(dtype=ti.i32, shape=MAX_SPHERES)

        self.num_light_triangles = ti.field(dtype=ti.i32, shape=())
        self.light_triangles_id = ti.field(dtype=ti.i32, shape=MAX_TRIANGLES)

        self.bvhs = BVHNode.field(shape=(MAX_BVHS, MAX_BVH_NODES))
        self.bvh_infos = BVHInfo.field(shape=(MAX_BVHS))
        self.num_bvhs = ti.field(dtype=ti.i32, shape=())
        self.num_nodes = ti.field(dtype=ti.i32, shape=(MAX_BVH_NODES))

        self.ground_color = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.horizon_color = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.sky_color = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.ground_color[None] =  1* ti.Vector([0.2, 0.1, 0.1])
        self.horizon_color[None] =  1* ti.Vector([0.6, 0.5, 0.5])
        self.sky_color[None] = 1*  ti.Vector([0.3, 0.5, 0.8])

        self.sun_direction = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.sun_color = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.sun_size = ti.field(dtype=ti.f32, shape=())

        # Default values
        self.sun_direction[None] = vec3f(0.0, 1.0, 0.0).normalized()  # overhead
        self.sun_color[None] = 1 * vec3f(5.0, 4.5, 3.0)  # warm bright sun
        self.sun_size[None] = 0.01  # sharpness of falloff (lower = smaller sun)
    
    # def add_material(self, albedo, emissive, shininess):
    #     material_id = self.num_materials[None]
    #     self.num_materials[None] += 1
    #     bsdf_type = BSDF_PHONG if shininess > EPS else BSDF_LAMBERT
    #     self.materials[material_id] =  Material(albedo=ti.Vector(list(albedo)), 
    #                                             emissive=ti.Vector(list(emissive)), 
    #                                             shininess=shininess,
    #                                             bsdf_type=bsdf_type)
    #     return material_id
    
    def add_material(self, mat: Material):
        mat_id = self.num_materials[None]
        self.materials[mat_id] = mat
        self.num_materials[None] += 1
        return mat_id

    def add_sphere(self, center:np.ndarray, radius:float, material_id:int):
        idx = self.num_spheres[None]
        if idx >= MAX_SPHERES:
            return -1
        self.num_spheres[None] += 1
        self.spheres[idx] = Sphere(ti.Vector(list(center)), radius, material_id)
        if self.materials[material_id].emissive.norm() > 0.0:
            light_idx = self.num_light_spheres[None]
            self.num_light_spheres[None] += 1
            self.light_spheres_id[light_idx] = idx
        return idx
        
    def add_plane(self, point:np.ndarray, normal:np.ndarray, material_id:int):
        idx = self.num_planes[None]
        if idx >= MAX_PLANES:
            return -1
        self.num_planes[None] += 1
        self.planes[idx] =  Plane(point=ti.Vector(list(point)), normal=ti.Vector(list(normal)), material_id=material_id)
        if self.materials[material_id].emissive.norm() > 0.0:
            print("[Warning] Plane lights are not supported.")
        return idx
    
    def add_triangle(self, v0:np.ndarray, v1:np.ndarray, v2:np.ndarray, material_id:int,
                     n0:Optional[np.ndarray]=None, n1:Optional[np.ndarray]=None, n2:Optional[np.ndarray]=None):
        idx = self.num_triangles[None]
        if idx >= MAX_TRIANGLES:
            return -1
        self.num_triangles[None] += 1

        v0 = np.asanyarray(v0)
        v1 = np.asanyarray(v1)
        v2 = np.asanyarray(v2)
        normal = np.cross(v1-v0, v2-v1)
        normal = normal / np.linalg.norm(normal)
        n0 = normal if n0 is None else np.asanyarray(n0)
        n1 = normal if n1 is None else np.asanyarray(n1)
        n2 = normal if n2 is None else np.asanyarray(n2)
        if normal.dot(n0 + n1 + n2) < 0:
            normal = -normal
       
        self.triangles[idx] =  Triangle(
            v0=ti.Vector(list(v0)),
            v1=ti.Vector(list(v1)), 
            v2=ti.Vector(list(v2)), 
            normal=ti.Vector(list(normal)),
            n0=ti.Vector(list(n0)),
            n1=ti.Vector(list(n1)),
            n2=ti.Vector(list(n2)),
            material_id=material_id)
        
        free_idx = self.num_free_triangles[None]
        self.free_triangles[free_idx] = idx
        self.num_free_triangles[None] += 1

        if self.materials[material_id].emissive.norm() > 0.0:
            light_idx = self.num_light_triangles[None]
            self.light_triangles_id[light_idx] = idx
            self.num_light_triangles[None] += 1
        return idx

    def add_quad(self, position, scale, axis, angle, material_id):
        position = np.array(position)
        scale = np.array(scale)
        axis = np.array(axis)
        def rotate(axis, angle, v):
            s = math.sin(angle*0.5)
            u = axis * s
            w = math.cos(angle*0.5)
            vu = (v * u).sum()
            uu = (u * u).sum()
            return 2 * vu * u + (w**2 - uu) * v + 2 * w * np.cross(u, v)

        fl = np.array([-0.5*scale[1], 0, -0.5*scale[0]])
        fr = np.array([ 0.5*scale[1], 0, -0.5*scale[0]])
        bl = np.array([-0.5*scale[1], 0,  0.5*scale[0]])
        br = np.array([ 0.5*scale[1], 0,  0.5*scale[0]])
        fl = rotate(axis, angle, fl) + position
        fr = rotate(axis, angle, fr) + position
        bl = rotate(axis, angle, bl) + position
        br = rotate(axis, angle, br) + position
        id1 = self.add_triangle(fl, fr, br, material_id)
        id2 = self.add_triangle(br, bl, fl, material_id)
        return id1, id2
    
    def add_mesh(self, mesh:tm.Trimesh, material_id:int):
        Nt = len(mesh.faces)
        tri_offset = self.num_triangles[None]
        free_tri_offset = self.num_free_triangles[None]
 
        if tri_offset + Nt >= MAX_TRIANGLES:
            print("[Scene] Not enough triangle slots!")
            return -1
        
        self.num_triangles[None] += Nt
        self.num_free_triangles[None] += Nt

        v0 = np.ascontiguousarray(mesh.triangles[:, 0])
        v1 = np.ascontiguousarray(mesh.triangles[:, 1])
        v2 = np.ascontiguousarray(mesh.triangles[:, 2])
        n = mesh.face_normals
        face_vertex_normals = mesh.vertex_normals[mesh.faces]
        n0 = np.ascontiguousarray(face_vertex_normals[:, 0])
        n1 = np.ascontiguousarray(face_vertex_normals[:, 1])
        n2 = np.ascontiguousarray(face_vertex_normals[:, 2])
        mid = np.full((Nt,), material_id, dtype=np.int32)
        triangle_ids = np.arange(tri_offset, tri_offset + Nt, dtype=np.int32)

        upload_triangles(self.triangles, tri_offset, Nt, v0, v1, v2, n, n0, n1, n2, mid)
        upload_free_triangles(self.free_triangles, free_tri_offset, Nt, triangle_ids)


    def add_mesh_bvh(self, mesh:tm.Trimesh, bvh_dict:dict, material_id:int, transform:np.ndarray=None):
        bvh_id = self.num_bvhs[None]
        Nt = len(mesh.faces)
        tri_offset = self.num_triangles[None]

        if self.num_bvhs[None] >= MAX_BVHS:
            print("[Scene] Not enough BVH slots!")
            return -1
 
        if tri_offset + Nt >= MAX_TRIANGLES:
            print("[Scene] Not enough triangle slots!")
            return -1
        

        is_leaf, left_or_start, right_or_count, aabb_min, aabb_max, depth, max_leaf_size, binned, torder = bvh_dict.values()
        Nn = len(is_leaf)

        if Nn > MAX_BVH_NODES:
            print("[Scene] Not enough bvh node slots!")
            return -1

        self.num_bvhs[None] += 1
        self.num_triangles[None] += Nt
        self.num_nodes[bvh_id] += Nn

        Nt = len(mesh.faces)
        v0 = mesh.triangles[:, 0]
        v1 = mesh.triangles[:, 1]
        v2 = mesh.triangles[:, 2]
        n = mesh.face_normals
        face_vertex_normals = mesh.vertex_normals[mesh.faces]
        n0 = face_vertex_normals[:, 0]
        n1 = face_vertex_normals[:, 1]
        n2 = face_vertex_normals[:, 2]
        mid = np.full((Nt,), material_id, dtype=np.int32)

        upload_triangles(self.triangles, tri_offset, Nt, v0[torder], v1[torder], v2[torder], n[torder], 
                         n0[torder], n1[torder], n2[torder], mid[torder])


        upload_bvh(self.bvhs, bvh_id, Nn, is_leaf, left_or_start, right_or_count, aabb_min, aabb_max, depth)
        self.bvh_infos.triangle_offset[bvh_id] = tri_offset
        self.bvh_infos.num_nodes[bvh_id] = Nn
        if transform is None:
            transform = np.identity(4, dtype=np.float32)
        inv_transform = np.linalg.inv(transform)
        self.bvh_infos.transform[bvh_id] = ti.Matrix(transform.tolist())
        self.bvh_infos.inv_transform[bvh_id] = ti.Matrix(inv_transform.tolist())

        return bvh_id

@ti.func
def environment_color(view_dir: vec3f, scene:ti.template()) -> vec3f: # type: ignore
    ground_color = scene.ground_color[None]
    horizon_color = scene.horizon_color[None]
    sky_color = scene.sky_color[None]
    sun_dir = scene.sun_direction[None]
    sun_col = scene.sun_color[None]
    sun_size = scene.sun_size[None]

    t = 0.5 * (view_dir.y + 1.0)  # remap from [-1, 1] to [0, 1]
    t = ti.min(ti.max(t, 0.0), 1.0)  # clamp

    # Blend top and bottom through horizon
    base = vec3f(0.0,0.0,0.0)
    if t < 0.5:
        # from ground to horizon
        base = (1.0 - t * 2.0) * ground_color + (t * 2.0) * horizon_color
    else:
        # from horizon to sky
        base = (1.0 - (t - 0.5) * 2.0) * horizon_color + ((t - 0.5) * 2.0) * sky_color

    cos_angle = view_dir.dot(sun_dir)
    glow = ti.exp((cos_angle - 1.0) / ti.max(sun_size, 1e-4))  # sharper peak = smaller sun
    sun = sun_col * glow
    return base + sun
