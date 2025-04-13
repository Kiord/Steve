import taichi as ti
from datatypes import vec3f
from scene import Material
import math

@ti.dataclass
class RandomSampler:
    i: ti.i32#type: ignore
    j: ti.i32#type: ignore
    f: ti.i32#type: ignore  
    counter: ti.i32#type: ignore

    @ti.func
    def next(self) -> ti.f32:#type: ignore
        seed = self.i * ti.u32(73856093) ^ self.j * ti.u32(19349663) ^ self.f * ti.u32(83492791) ^ self.counter * ti.u32(2654435761)
        self.counter += 1
        return (ti.sin(seed * 0.0001) * 43758.5453) % 1.0
    
    @ti.func
    def next2(self) -> ti.f32:#type: ignore
        return self.next(), self.next()
    
    @ti.func
    def next3(self) -> ti.f32:#type: ignore
        return self.next(), self.next(), self.next()


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
    
    result = ti.Vector([main_dir.z, 0.0, 0.0]) * local

    if ti.abs(main_dir.z) <= 0.99999:
        up = ti.Vector([0.0, 0.0, 1.0])
        axis = main_dir.cross(up).normalized()
        angle = ti.acos(main_dir.dot(up))
        result = rotate(axis, angle, local)
    return result

@ti.func
def sample_BSDF(normal: vec3f, material:Material, incoming_dir: vec3f, sampler: RandomSampler) -> DirectionSample:
    n = material.shininess
    is_lambert = n == 0.0
    r = ti.select(is_lambert, normal, reflect(-incoming_dir, normal))

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

    coswo = normal.dot(sampled.direction)
    ok = coswo > 0.0
    cosr_pow_n = ti.pow(abs(cosr), n)

    w = (n + 1.0 + ti.cast(is_lambert, ti.f32)) / (2.0 * math.pi)

    sampled.pdf = ti.select(ok, w * (coswo if is_lambert else cosr_pow_n), 0.0)
    sampled.bsdf = ti.select(ok, material.diffuse * w * cosr_pow_n, ti.Vector([0.0, 0.0, 0.0]))

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


@ti.func
def normalize(v):
    return v / v.norm()

@ti.func
def random_unit_vector(sampler:RandomSampler):
    u1, u2 = sampler.next2()
    theta = 2 * math.pi * u1
    z = u2 * 2 - 1
    r = (1 - z * z).sqrt()
    return ti.Vector([r * ti.cos(theta), r * ti.sin(theta), z])

@ti.dataclass
class Contribution:
    value: vec3f # type:ignore
    pdf: ti.f32 # type:ignore

@ti.dataclass
class LightSample:
    direction: vec3f
    contrib: vec3f
    pdf: ti.f32
