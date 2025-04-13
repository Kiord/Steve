import taichi as ti
from datatypes import vec3f
from constants import *

@ti.dataclass
class Material:
    diffuse: vec3f # type: ignore
    specular: vec3f # type: ignore
    shininess: ti.f32 # type: ignore
    emissive: vec3f # type: ignore



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
class PointLight:
    position: vec3f # type: ignore
    color: vec3f # type: ignore

class Scene:
    def __init__(self):
        # existing
        self.num_spheres = ti.field(dtype=ti.i32, shape=())
        self.spheres = Sphere.field(shape=MAX_SPHERES)
        self.num_planes = ti.field(dtype=ti.i32, shape=())
        self.planes = Plane.field(shape=MAX_PLANES)
        self.materials = Material.field(shape=MAX_MATERIALS)
        self.light = PointLight.field(shape=())