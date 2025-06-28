import taichi as ti
from datatypes import vec3f
from utils import LightSample, sample_sphere_solid_angle
from ray import visibility
from bsdf import bsdf_eval
from constants import *


@ti.func
def emitter_sample(emitter_type: ti.i32, scene: ti.template(), inter, sampler) -> LightSample:
    sample = LightSample(direction=vec3f(0.0), contrib=vec3f(0.0), pdf=0.0)
    if emitter_type == EMITTER_SPHERE:
        sample = sample_sphere_emitter(scene, inter, sampler)
    elif emitter_type == EMITTER_TRIANGLE:
        sample = sample_triangle_emitter(scene, inter, sampler)
    return sample

# -------------------------------------------------------------------------
#  SPHERE EMITTER IMPLEMENTATION
# -------------------------------------------------------------------------

@ti.func
def sample_sphere_emitter(scene: ti.template(), inter, sampler) -> LightSample:
    sample = LightSample(direction=vec3f(0.0), contrib=vec3f(0.0), pdf=0.0)

    num = scene.num_light_spheres[None]
    has_spheres = num > 0

    u = sampler.next()
    sphere_idx = int(u * num) if has_spheres else 0
    light_id = scene.light_spheres_id[sphere_idx]
    sphere = scene.spheres[light_id]
    mat = scene.materials[sphere.material_id]

    sls = sample_sphere_solid_angle(inter.point, sphere, sampler)
    hit_pos = sls.point
    dir = hit_pos - inter.point
    dist = dir.norm()
    dir = dir / ti.max(dist, EPS)

    pdf = sls.pdf / ti.max(num, 1)

    mat_surface = scene.materials[inter.material_id]
    bsdf_val = bsdf_eval(mat_surface, inter.normal, -inter.ray.direction, dir)
    cos_theta = max(0.0, inter.normal.dot(dir))
    vis = visibility(scene, inter.point + EPS * inter.normal, hit_pos)

    contrib = vis * mat.emissive * bsdf_val * cos_theta / pdf

    if has_spheres:
        sample.direction = dir
        sample.contrib = contrib
        sample.pdf = pdf

    return sample

# -------------------------------------------------------------------------
#  TRIANGLE EMITTER IMPLEMENTATION
# -------------------------------------------------------------------------

@ti.func
def sample_triangle_emitter(scene: ti.template(), inter, sampler) -> LightSample:
    sample = LightSample(direction=vec3f(0.0), contrib=vec3f(0.0), pdf=0.0)

    num = scene.num_light_triangles[None]
    has_triangles = num > 0

    u = sampler.next()
    tri_idx = int(u * num) if has_triangles else 0
    tri_id = scene.light_triangles_id[tri_idx]
    tri = scene.triangles[tri_id]
    mat = scene.materials[tri.material_id]

    is_emissive = mat.emissive.norm() > 0.0
    active = has_triangles and is_emissive

    u1 = sampler.next()
    u2 = sampler.next()
    sqrt_u1 = ti.sqrt(u1)
    b0 = 1.0 - sqrt_u1
    b1 = sqrt_u1 * (1.0 - u2)
    b2 = sqrt_u1 * u2

    hit_pos = b0 * tri.v0 + b1 * tri.v1 + b2 * tri.v2
    normal = tri.normal

    dir = hit_pos - inter.point
    dist = dir.norm()
    dir = dir / ti.max(dist, EPS)

    edge1 = tri.v1 - tri.v0
    edge2 = tri.v2 - tri.v0
    area = 0.5 * edge1.cross(edge2).norm()
    pdf = 1.0 / (area * ti.max(num, 1))

    mat_surface = scene.materials[inter.material_id]
    bsdf_val = bsdf_eval(mat_surface, inter.normal, -inter.ray.direction, dir)
    cos_theta = max(0.0, inter.normal.dot(dir))
    vis = visibility(scene, inter.point + EPS * inter.normal, hit_pos)

    contrib = vis * mat.emissive * bsdf_val * cos_theta / pdf

    if active:
        sample.direction = dir
        sample.contrib = contrib
        sample.pdf = pdf

    return sample
