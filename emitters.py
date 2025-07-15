import taichi as ti
from datatypes import vec3f
from utils import sample_sphere_solid_angle, pdf_solid_angle_sphere
from ray import visibility
from bsdf import bsdf_eval
from constants import *

@ti.dataclass
class LightSample:
    direction: vec3f# type: ignore
    contrib: vec3f# type: ignore
    pdf: ti.f32# type: ignore

@ti.dataclass
class EmitterSample():
    primitive_id : ti.i32 # type: ignore
    type: ti.i32 #type: ignore
    pdf: ti.f32 # type: ignore

@ti.func
def sample_emissive_primitive(scene: ti.template(), sampler) -> EmitterSample:
    u = sampler.next() * scene.total_emissive_area[None]
    accum = 0.0
    emitter_sample = EmitterSample(0, EMITTER_NONE, 1.0)

    for i in range(scene.num_emissive_primitives[None]):
        entry = scene.emissive_primitives[i]
        accum += entry.area
        if u < accum:
            emitter_sample.primitive_id = entry.index
            emitter_sample.type = entry.type
            emitter_sample.pdf = entry.area / scene.total_emissive_area[None]
            break

    return emitter_sample

@ti.func
def pdf_solid_angle(scene: ti.template(), inter) -> ti.f32:
    pdf = 0.0
    wi = -inter.ray.direction
    if inter.primitive_type == PRIMITIVE_TRIANGLE:
        pdf_area = 1.0 / scene.total_emissive_area[None]
        cos_light = abs(wi.dot(inter.normal))
        pdf = (pdf_area * inter.t * inter.t) / max(cos_light, EPS)

    elif inter.primitive_type == PRIMITIVE_SPHERE:
        sphere = scene.spheres[inter.primitive_id]
        viewer = inter.ray.origin
        pdf = pdf_solid_angle_sphere(sphere, viewer) * sphere.area / scene.total_emissive_area[None]

    return pdf

@ti.func
def emitter_sample(scene: ti.template(), inter, sampler) -> LightSample:
    emitter_sample = sample_emissive_primitive(scene, sampler)

    sample = LightSample(direction=vec3f(0.0), contrib=vec3f(0.0), pdf=0.0)
    if emitter_sample.type == EMITTER_SPHERE:
        sample = sample_sphere_emitter(scene, inter, emitter_sample, sampler)
    elif emitter_sample.type == EMITTER_TRIANGLE:
        sample = sample_triangle_emitter(scene, inter, emitter_sample, sampler)
    else:
        sample = LightSample(direction=vec3f(0.0), contrib=vec3f(0.0), pdf=0.0)
    return sample

# -------------------------------------------------------------------------
#  SPHERE EMITTER IMPLEMENTATION
# -------------------------------------------------------------------------

@ti.func
def sample_sphere_emitter(scene: ti.template(), inter, emitter_sample, sampler) -> LightSample:
    
    sphere = scene.spheres[emitter_sample.primitive_id]
    mat = scene.materials[sphere.material_id]

    sls = sample_sphere_solid_angle(inter.point, sphere, sampler)
    hit_pos = sls.point
    dir = hit_pos - inter.point
    dist = dir.norm()
    dir = dir / ti.max(dist, EPS)
    
    pdf_solid_angle = emitter_sample.pdf * sls.pdf
    #pdf = sls.pdf / ti.max(num, 1)

    mat_surface = scene.materials[inter.material_id]
    bsdf_val = bsdf_eval(mat_surface, inter.normal, -inter.ray.direction, dir)
    cos_theta = max(0.0, inter.normal.dot(dir))
    vis = visibility(scene, inter.point + EPS * inter.normal, hit_pos)

    contrib = vis * mat.emissive * bsdf_val * cos_theta / pdf_solid_angle

    return LightSample(direction=dir, contrib=contrib, pdf=pdf_solid_angle)

# -------------------------------------------------------------------------
#  TRIANGLE EMITTER IMPLEMENTATION
# -------------------------------------------------------------------------

@ti.func
def sample_triangle_emitter(scene: ti.template(), inter, emitter_sample, sampler) -> LightSample:
    
    tri = scene.triangles[emitter_sample.primitive_id]
    mat = scene.materials[tri.material_id]

    u1 = sampler.next()
    u2 = sampler.next()
    sqrt_u1 = ti.sqrt(u1)
    b0 = 1.0 - sqrt_u1
    b1 = sqrt_u1 * (1.0 - u2)
    b2 = sqrt_u1 * u2

    hit_pos = b0 * tri.v0 + b1 * tri.v1 + b2 * tri.v2

    dir = hit_pos - inter.point
    dist = dir.norm()
    dir = dir / ti.max(dist, EPS)

 
    #pdf_area = triangle_pdf * (1 / area_triangle)
    # = (area_triangle / total_emissive_area) * (1 / area_triangle)
    # = 1 / total_emissive_area
    pdf_area = 1.0 / scene.total_emissive_area[None]
    pdf_solid_angle = (pdf_area * dist * dist) / ti.abs(tri.normal.dot(dir))


    mat_surface = scene.materials[inter.material_id]
    bsdf_val = bsdf_eval(mat_surface, inter.normal, -inter.ray.direction, dir)
    cos_theta = max(0.0, inter.normal.dot(dir))
    vis = visibility(scene, inter.point + EPS * inter.normal, hit_pos)

    contrib = vis * mat.emissive * bsdf_val * cos_theta / pdf_solid_angle

    return LightSample(direction=dir, contrib=contrib, pdf=pdf_solid_angle)
