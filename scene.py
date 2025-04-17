import taichi as ti
from datatypes import vec3f
from constants import *
import numpy as np
import math

@ti.dataclass
class Material:
    albedo: vec3f # type: ignore
    emissive: vec3f # type: ignore
    shininess: ti.f32 # type: ignore



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
    material_id: ti.i32 #type:ignore

@ti.dataclass
class PointLight:
    position: vec3f # type: ignore
    color: vec3f # type: ignore

class Scene:
    def __init__(self):
        # existing
        self.num_spheres = ti.field(dtype=ti.i32, shape=())
        self.spheres = Sphere.field(shape=MAX_SPHERES)
        self.num_triangles = ti.field(dtype=ti.i32, shape=())
        self.triangles = Triangle.field(shape=MAX_TRIANGLES)
        self.num_planes = ti.field(dtype=ti.i32, shape=())
        self.planes = Plane.field(shape=MAX_PLANES)
        self.materials = Material.field(shape=MAX_MATERIALS)
        self.num_materials = ti.field(dtype=ti.i32, shape=())
        self.num_light_spheres = ti.field(dtype=ti.i32, shape=())
        self.light_spheres_id = ti.field(dtype=ti.i32, shape=MAX_SPHERES)

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
    
    def add_material(self, albedo, emissive, shininess):
        material_id = self.num_materials[None]
        self.num_materials[None] += 1
        self.materials[material_id] =  Material(albedo=ti.Vector(list(albedo)), 
                                                emissive=ti.Vector(list(emissive)), 
                                                shininess=shininess)
        return material_id

    def add_sphere(self, center:np.ndarray, radius:float, material_id:int):
        idx = self.num_spheres[None]
        self.num_spheres[None] += 1
        self.spheres[idx] = Sphere(ti.Vector(list(center)), radius, material_id)
        if self.materials[material_id].emissive.norm() > 0.0:
            light_idx = self.num_light_spheres[None]
            self.num_light_spheres[None] += 1
            self.light_spheres_id[light_idx] = idx
        return idx
        
    def add_plane(self, point:np.ndarray, normal:np.ndarray, material_id:int):
        idx = self.num_planes[None]
        self.num_planes[None] += 1
        self.planes[idx] =  Plane(point=ti.Vector(list(point)), normal=ti.Vector(list(normal)), material_id=material_id)
        if self.materials[material_id].emissive.norm() > 0.0:
            print("[Warning] Plane lights are not supported.")
        return idx
    
    def add_triangle(self, v0:np.ndarray, v1:np.ndarray, v2:np.ndarray, material_id:int):
        v0 = np.asanyarray(v0)
        v1 = np.asanyarray(v1)
        v2 = np.asanyarray(v2)
        idx = self.num_triangles[None]
        self.num_triangles[None] += 1
        normal = np.cross(v1-v0, v2-v1)
        normal = normal / np.linalg.norm(normal)
        self.triangles[idx] =  Triangle(
            v0=ti.Vector(list(v0)),
            v1=ti.Vector(list(v1)), 
            v2=ti.Vector(list(v2)), 
            normal=ti.Vector(list(normal)),
            material_id=material_id)
        if self.materials[material_id].emissive.norm() > 0.0:
            print("[Warning] Triangle lights are not supported.")
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
