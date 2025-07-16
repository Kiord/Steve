import taichi as ti
from datatypes import vec3f
from constants import BSDF_LAMBERT, BSDF_PHONG, EPS, BSDF_GGX
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

@ti.func
def fresnel_schlick(cos_theta, F0: vec3f) -> vec3f:
    return F0 + (1.0 - F0) * ti.pow(1.0 - cos_theta, 5.0)

@ti.func
def D_GGX(N: vec3f, H: vec3f, alpha: ti.f32) -> ti.f32:
    NoH = ti.max(N.dot(H), 1e-4)
    a2 = alpha * alpha
    s = a2 - 1.0
    NoH2 = NoH * NoH
    denom = ti.max(NoH2 * s + 1.0, EPS)
    denom2 = denom * denom
    D = a2 / (ti.math.pi * denom2 + EPS)
    return D

@ti.func
def G1_GGX(N: vec3f, V: vec3f, alpha: ti.f32) -> ti.f32:
    NoV = ti.max(N.dot(V), EPS)
    tan_theta = ti.sqrt(1.0 - NoV * NoV) / NoV
    a = alpha * tan_theta
    return 2.0 / (1.0 + ti.sqrt(1.0 + a * a))

@ti.func
def G_Smith_GGX(N: vec3f, V: vec3f, L: vec3f, alpha: ti.f32) -> ti.f32:
    return G1_GGX(N, V, alpha) * G1_GGX(N, L, alpha)

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
    elif material.bsdf_type == BSDF_GGX:
        result = sample_ggx(material, normal, incoming, sampler)
    return result

@ti.func
def bsdf_eval(material: Material, normal: vec3f, wi: vec3f, wo: vec3f) -> vec3f:
    result = vec3f(0.0)
    if material.bsdf_type == BSDF_LAMBERT:
        result = eval_lambert(material, normal, wi, wo)
    elif material.bsdf_type == BSDF_PHONG:
        result = eval_phong(material, normal, wi, wo)
    elif material.bsdf_type == BSDF_GGX:
        result = eval_ggx(material, normal, wi, wo)
    return result

@ti.func
def bsdf_pdf(material: Material, normal: vec3f, wi: vec3f, wo: vec3f) -> ti.f32:
    result = 0.0
    if material.bsdf_type == BSDF_LAMBERT:
        result = pdf_lambert(material, normal, wi, wo)
    elif material.bsdf_type == BSDF_PHONG:
        result = pdf_phong(material, normal, wi, wo)
    elif material.bsdf_type == BSDF_GGX:
        result = pdf_ggx(material, normal, wi, wo)
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
    r = reflect(incoming, normal)
    wo = random_direction_hemisphere(r, material.shininess, sampler)
    cos_theta = normal.dot(wo)
    cos_alpha = r.dot(wo)

    if cos_theta > 0 and cos_alpha > 0:
        pdf = ((material.shininess + 1.0) * ti.pow(cos_alpha, material.shininess)) / (2.0 * ti.math.pi)
        bsdf = material.albedo * ((material.shininess + 2.0) / (2.0 * ti.math.pi)) * ti.pow(cos_alpha, material.shininess)
        ds = DirectionSample(wo, pdf, bsdf)
    return ds

@ti.func
def eval_phong(material: Material, normal: vec3f, wi: vec3f, wo: vec3f) -> vec3f:
    result = vec3f(0.0)
    r = reflect(-wi, normal)
    cos_alpha = r.dot(wo)
    cos_theta = normal.dot(wo)
    if cos_alpha > 0 and cos_theta > 0:
        result = material.albedo * ((material.shininess + 2.0) / (2.0 * ti.math.pi)) * ti.pow(cos_alpha, material.shininess)
    return result

@ti.func
def pdf_phong(material: Material, normal: vec3f, wi: vec3f, wo: vec3f) -> ti.f32:
    result = 0.0
    r = reflect(-wi, normal)
    cos_alpha = r.dot(wo)
    if cos_alpha > 0:
        result = ((material.shininess + 1.0) * ti.pow(cos_alpha, material.shininess)) / (2.0 * ti.math.pi)
    return result

# -------------------------------------------------------------------------
#  GGX BSDF
# -------------------------------------------------------------------------


@ti.func
def sample_ggx(material: Material, normal: vec3f, incoming: vec3f, sampler) -> DirectionSample:
    ds = empty_direction_sample()
    alpha = material.roughness * material.roughness

    V = -incoming.normalized()

    # Construct ONB around V
    w = V
    a = vec3f(0.0, 1.0, 0.0) if ti.abs(w.x) > 0.9 else vec3f(1.0, 0.0, 0.0)
    v = w.cross(a)
    if v.norm() > EPS:
        v = v.normalized()
        u = v.cross(w)

        u1, u2 = sampler.next2()
        phi = 2.0 * ti.math.pi * u1
        cos_theta = ti.sqrt((1.0 - u2) / (1.0 + (alpha * alpha - 1.0) * u2))
        sin_theta = ti.sqrt(1.0 - cos_theta * cos_theta)

        H_local = vec3f(
            sin_theta * ti.cos(phi),
            sin_theta * ti.sin(phi),
            cos_theta
        )

        H = (u * H_local.x + v * H_local.y + w * H_local.z)
        if H.norm() > EPS:
            H = H.normalized()

            # Ensure H is in same hemisphere as normal
            H = ti.select(normal.dot(H) < 0.0, -H, H)

            L = reflect(-V, H).normalized()

            NoV = ti.max(normal.dot(V), EPS)
            NoL = ti.max(normal.dot(L), EPS)
            VoH = ti.max(V.dot(H), EPS)
            NoH = ti.max(normal.dot(H), EPS)

            if NoL > 0.0:
                D = D_GGX(normal, H, alpha)
                G = G_Smith_GGX(normal, V, L, alpha)
                F = fresnel_schlick(VoH, material.albedo)

                bsdf = D * G * F / (4.0 * NoV * NoL + EPS)
                pdf = D * NoH / (4.0 * VoH + EPS)

                ds = DirectionSample(direction=L, pdf=pdf, bsdf=bsdf)

    return ds

@ti.func
def eval_ggx(material: Material, normal: vec3f, wi: vec3f, wo: vec3f) -> vec3f:
    alpha = material.roughness * material.roughness
    H = (wi + wo).normalized()

    NoV = ti.max(normal.dot(wi), EPS)
    NoL = ti.max(normal.dot(wo), EPS)
    VoH = ti.max(wi.dot(H), EPS)
    NoH = ti.max(normal.dot(H), EPS)

    result = vec3f(0.0)

    if NoV > 0 and NoL > 0:
        D = D_GGX(normal, H, alpha)
        G = G_Smith_GGX(normal, wi, wo, alpha)
        F = fresnel_schlick(VoH, material.albedo)
        result = D * G * F / (4.0 * NoV * NoL + EPS)

    return result


@ti.func
def pdf_ggx(material: Material, normal: vec3f, wi: vec3f, wo: vec3f) -> ti.f32:
    alpha = material.roughness * material.roughness
    H = (wi + wo).normalized()

    VoH = ti.max(wi.dot(H), EPS)
    NoH = ti.max(normal.dot(H), EPS)
    D = D_GGX(normal, H, alpha)

    # Only compute PDF if directions are valid
    NoV = normal.dot(wi)
    NoL = normal.dot(wo)
    is_valid = ti.cast(NoV > 0 and NoL > 0, ti.f32)

    return is_valid * D * NoH / (4.0 * VoH + EPS)
