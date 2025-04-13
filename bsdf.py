import taichi as ti
from path_tracing import RandomSampler
import math
from datatypes import vec3f

@ti.func
def reflect(v: vec3f, n: vec3f) -> vec3f:
    return v - 2.0 * v.dot(n) * n

@ti.func
def rotate(axis: vec3f, angle: ti.f32, v: vec3f) -> vec3f:
    c = ti.cos(angle)
    s = ti.sin(angle)
    return v * c + axis.cross(v) * s + axis * (axis.dot(v)) * (1.0 - c)

@ti.dataclass
class DirectionSample:
    direction: vec3f # type: ignore
    pdf: ti.f32  # type: ignore
    bsdf: vec3f # type: ignore

@ti.func
def randomDirectionHemisphere(main_dir: vec3f, n: ti.f32, sampler: RandomSampler) -> vec3f:
    r1 = sampler.next()
    r2 = sampler.next()
    phi = 2.0 * math.pi * r1
    cos_theta = r2 ** (1.0 / (n + 1.0))
    sin_theta = ti.sqrt(1.0 - cos_theta * cos_theta)

    local = ti.Vector([
        ti.cos(phi) * sin_theta,
        ti.sin(phi) * sin_theta,
        cos_theta
    ])

    # Create basis
    if ti.abs(main_dir.z) > 0.99999:
        return ti.Vector([main_dir.z, 0.0, 0.0]) * local
    else:
        up = ti.Vector([0.0, 0.0, 1.0])
        axis = main_dir.cross(up).normalized()
        angle = ti.acos(main_dir.dot(up))
        return rotate(axis, angle, local)

@ti.func
def sample_BSDF(inter, incoming_dir, sampler: RandomSampler) -> DirectionSample:
    n = inter.material.shininess
    is_lambert = n == 0.0
    r = ti.select(is_lambert, inter.normal, reflect(-incoming_dir, inter.normal))

    sampled = DirectionSample(
        direction=ti.Vector([0.0, 0.0, 0.0]),
        pdf=0.0,
        bsdf=ti.Vector([0.0, 0.0, 0.0])
    )

    sampled.direction = randomDirectionHemisphere(r, max(n, 1.0), sampler)

    cosr = r.dot(sampled.direction)
    correct_r = 1.0 if cosr > 0.0 else -1.0
    cosr *= correct_r
    sampled.direction *= correct_r

    coswo = inter.normal.dot(sampled.direction)
    ok = coswo > 0.0
    cosr_pow_n = ti.pow(abs(cosr), n)

    w = (n + 1.0 + ti.cast(is_lambert, ti.f32)) / (2.0 * math.pi)

    sampled.pdf = ti.select(ok, w * (coswo if is_lambert else cosr_pow_n), 0.0)
    sampled.bsdf = ti.select(ok, inter.material.albedo * w * cosr_pow_n, ti.Vector([0.0, 0.0, 0.0]))

    return sampled

@ti.func
def BSDF(inter, wi, wo):
    n = inter.material.shininess
    is_lambert = n == 0.0
    coswi = inter.normal.dot(wi)
    r = ti.select(is_lambert, inter.normal, reflect(wo, inter.normal))
    cosr = r.dot(wi)
    coswo = ti.clamp(inter.normal.dot(wo), -1.0, 1.0)
    ok = (coswi > 0.0) and (cosr > 0.0) and (coswo > 0.0)
    factor = (n + 1.0 + ti.cast(is_lambert, ti.f32)) / (2.0 * math.pi)
    return ti.select(ok, inter.material.albedo * factor * cosr ** n, ti.Vector([0.0, 0.0, 0.0]))

