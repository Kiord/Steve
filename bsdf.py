import taichi as ti
from datatypes import vec3f
from constants import BSDF_LAMBERT, BSDF_PHONG, EPS
from utils import reflect, random_direction_hemisphere
from scene import Material

# -------------------------------------------------------------------------
#  SUPPORT STRUCTS
# -------------------------------------------------------------------------

@ti.dataclass
class DirectionSample:
    direction: vec3f
    pdf: ti.f32
    bsdf: vec3f

@ti.func
def empty_direction_sample() -> DirectionSample:
    return DirectionSample(direction=vec3f(0.0), pdf=0.0, bsdf=vec3f(0.0))

# -------------------------------------------------------------------------
#  INTERFACE FUNCTIONS (to be used by path tracer)
# -------------------------------------------------------------------------

@ti.func
def bsdf_sample(material: Material, normal: vec3f, incoming: vec3f, sampler) -> DirectionSample:
    result = empty_direction_sample()
    if material.bsdf_type == BSDF_LAMBERT:
        result = sample_lambert(material, normal, sampler)
    elif material.bsdf_type == BSDF_PHONG:
        result = sample_phong(material, normal, incoming, sampler)
    return result

@ti.func
def bsdf_eval(material: Material, normal: vec3f, wi: vec3f, wo: vec3f) -> vec3f:
    result = vec3f(0.0)
    if material.bsdf_type == BSDF_LAMBERT:
        result = eval_lambert(material, normal, wi, wo)
    elif material.bsdf_type == BSDF_PHONG:
        result = eval_phong(material, normal, wi, wo)
    return result

@ti.func
def bsdf_pdf(material: Material, normal: vec3f, wi: vec3f, wo: vec3f) -> ti.f32:
    result = 0.0
    if material.bsdf_type == BSDF_LAMBERT:
        result = pdf_lambert(material, normal, wi, wo)
    elif material.bsdf_type == BSDF_PHONG:
        result = pdf_phong(material, normal, wi, wo)
    return result


# -------------------------------------------------------------------------
#  LAMBERT BSDF
# -------------------------------------------------------------------------

@ti.func
def sample_lambert(material: Material, normal: vec3f, sampler) -> DirectionSample:
    ds = empty_direction_sample()
    dir = random_direction_hemisphere(normal, 1.0, sampler)
    cos_theta = normal.dot(dir)
    if cos_theta > 0:
        pdf = cos_theta / ti.math.pi
        bsdf = material.albedo / ti.math.pi
        ds = DirectionSample(dir, pdf, bsdf)
    return ds

@ti.func
def eval_lambert(material: Material, normal: vec3f, wi: vec3f, wo: vec3f) -> vec3f:
    cos_theta = normal.dot(wo)
    return ti.select(cos_theta > 0, material.albedo / ti.math.pi, vec3f(0.0))

@ti.func
def pdf_lambert(material: Material, normal: vec3f, wi: vec3f, wo: vec3f) -> ti.f32:
    cos_theta = normal.dot(wo)
    return ti.select(cos_theta > 0, cos_theta / ti.math.pi, 0.0)

# -------------------------------------------------------------------------
#  PHONG BSDF
# -------------------------------------------------------------------------

@ti.func
def sample_phong(material: Material, normal: vec3f, incoming: vec3f, sampler) -> DirectionSample:
    ds = empty_direction_sample()
    reflect_dir = reflect(incoming, normal)
    dir = random_direction_hemisphere(reflect_dir, material.shininess, sampler)
    cos_theta = normal.dot(dir)
    cos_alpha = reflect_dir.dot(dir)

    if cos_theta > 0 and cos_alpha > 0:
        pdf = ((material.shininess + 1.0) * ti.pow(cos_alpha, material.shininess)) / (2.0 * ti.math.pi)
        bsdf = material.albedo * ((material.shininess + 2.0) / (2.0 * ti.math.pi)) * ti.pow(cos_alpha, material.shininess)
        ds = DirectionSample(dir, pdf, bsdf)
    return ds

@ti.func
def eval_phong(material: Material, normal: vec3f, wi: vec3f, wo: vec3f) -> vec3f:
    result = vec3f(0.0)
    r = reflect(wi, normal)
    cos_alpha = r.dot(wo)
    cos_theta = normal.dot(wo)
    if cos_alpha > 0 and cos_theta > 0:
        result = material.albedo * ((material.shininess + 2.0) / (2.0 * ti.math.pi)) * ti.pow(cos_alpha, material.shininess)
    return result

@ti.func
def pdf_phong(material: Material, normal: vec3f, wi: vec3f, wo: vec3f) -> ti.f32:
    result = 0.0
    r = reflect(wi, normal)
    cos_alpha = r.dot(wo)
    if cos_alpha > 0:
        result = ((material.shininess + 1.0) * ti.pow(cos_alpha, material.shininess)) / (2.0 * ti.math.pi)
    return result
