import taichi as ti
from ray import hit_scene, Ray
import math
from datatypes import vec3f
from scene import Material
from camera import make_ray

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

# @ti.func
# def BSDF(inter, wi, wo):
#     n = inter.material.shininess
#     is_lambert = n == 0.0
#     coswi = inter.normal.dot(wi)
#     r = ti.select(is_lambert, inter.normal, reflect(wo, inter.normal))
#     cosr = r.dot(wi)
#     coswo = ti.clamp(inter.normal.dot(wo), -1.0, 1.0)
#     ok = (coswi > 0.0) and (cosr > 0.0) and (coswo > 0.0)
#     factor = (n + 1.0 + ti.cast(is_lambert, ti.f32)) / (2.0 * math.pi)
#     return ti.select(ok, inter.material.albedo * factor * cosr ** n, ti.Vector([0.0, 0.0, 0.0]))



class RenderBuffers:
    def __init__(self, width:int, height:int):
        self.color = ti.Vector.field(3, ti.f32, shape=(width, height))
        self.albedo = ti.Vector.field(3, ti.f32, shape=(width, height))
        self.normal = ti.Vector.field(3, ti.f32, shape=(width, height))
        self.denoised = ti.Vector.field(3, ti.f32, shape=(width, height))

@ti.func
def normalize(v):
    return v / v.norm()

@ti.func
def random_unit_vector():
    theta = 2 * math.pi * ti.random()
    z = ti.random() * 2 - 1
    r = (1 - z * z).sqrt()
    return ti.Vector([r * ti.cos(theta), r * ti.sin(theta), z])

@ti.func
def hash_int(x: ti.i32) -> ti.u32: # type:ignore
    x = ((x >> 16) ^ x) * 0x45d9f3b
    x = ((x >> 16) ^ x) * 0x45d9f3b
    x = (x >> 16) ^ x
    return x & ti.u32(0xFFFFFFFF)

@ti.func
def spatial_random(i: ti.i32, j: ti.i32, b: ti.i32, s: ti.i32) -> ti.f32:  # type:ignore
    seed = i * ti.u32(73856093) ^ j * ti.u32(19349663) ^ b * ti.u32(83492791) ^ s * ti.u32(2654435761)
    return (hash_int(seed) & ti.u32(0xFFFFFF)) / 16777216.0



@ti.func
def path_trace(scene: ti.template(), ray : Ray, i: ti.i32, j: ti.i32, max_depth: int, sampler: RandomSampler, buffers: ti.template()):  # type: ignore
    throughput = ti.Vector([1.0, 1.0, 1.0])
    result = ti.Vector([0.0, 0.0, 0.0])
    background = ti.Vector([0.75, 0.75, 0.75])
    aux_albedo = background
    aux_normal = ti.Vector([0.0, 0.0, 0.0])

    for bounce in range(max_depth):
        inter = hit_scene(ray, scene, 0.001, 1e5)
        if not inter.hit:
            result += throughput * background
            break

        mat = inter.material

        # Direct light sampling
        light_dir = (scene.light[None].position - inter.point).normalized()
        light_dist = (scene.light[None].position - inter.point).norm()
        inter_light = hit_scene(Ray(inter.point + inter.normal * 1e-3, light_dir), scene, 0.001, light_dist - 1e-2)
        if not inter_light.hit:
            lambert = max(0.0, inter.normal.dot(light_dir))
            light_contrib = (mat.diffuse * lambert + mat.specular) * scene.light[None].color / (light_dist**2)
            result += throughput * light_contrib

        # Sample BSDF
        sample = sample_BSDF(inter.normal, mat, ray.direction, sampler)

        # Russian roulette (optional for >2 bounces)
        if bounce > 2:
            p = max(throughput.x, throughput.y, throughput.z)
            if sampler.next() > p:
                break
            throughput /= p

        if sample.pdf > 0:
            throughput *= sample.bsdf / sample.pdf
        else:
            break

        ray.origin = inter.point + inter.normal * 1e-4
        ray.direction = sample.direction

        if bounce == 0:
            aux_albedo = mat.diffuse
            aux_normal = inter.normal

    buffers.color[i, j] += result
    buffers.albedo[i, j] += aux_albedo
    buffers.normal[i, j] += aux_normal

# @ti.func
# def path_trace(scene: ti.template(), ray_o, ray_d, i: ti.i32, j: ti.i32, max_depth: int, sampler: RandomSampler, buffers: ti.template()):  # type: ignore
#     throughput = ti.Vector([1.0, 1.0, 1.0])
#     result = ti.Vector([0.0, 0.0, 0.0])
#     background = ti.Vector([0.75, 0.75, 0.75])
#     aux_albedo = background
#     aux_normal = ti.Vector([0.0, 0.0, 0.0])

#     for bounce in range(max_depth):
#         hit, rec = hit_scene(scene, ray_o, ray_d, 0.001, 1e5)
#         if not hit:
#             result += throughput * background
#             break

#         mat = scene.materials[rec.material_id]

#         # Direct light sampling
#         light_dir = (scene.light[None].position - rec.point).normalized()
#         light_dist = (scene.light[None].position - rec.point).norm()
#         blocked, _ = hit_scene(scene, rec.point + rec.normal * 1e-3, light_dir, 0.001, light_dist - 1e-2)
#         if not blocked:
#             lambert = max(0.0, rec.normal.dot(light_dir))
#             light_contrib = (mat.diffuse * lambert + mat.specular) * scene.light[None].color / (light_dist**2)
#             result += throughput * light_contrib

#         # Sample BSDF
#         sample = sample_BSDF(rec.normal, mat, ray_d, sampler)

#         # Russian roulette (optional for >2 bounces)
#         if bounce > 2:
#             p = max(throughput.x, throughput.y, throughput.z)
#             if sampler.next() > p:
#                 break
#             throughput /= p

#         if sample.pdf > 0:
#             throughput *= sample.bsdf / sample.pdf
#         else:
#             break

#         ray_o = rec.point + rec.normal * 1e-4
#         ray_d = sample.direction

#         if bounce == 0:
#             aux_albedo = mat.diffuse
#             aux_normal = rec.normal

#     buffers.color[i, j] += result
#     buffers.albedo[i, j] += aux_albedo
#     buffers.normal[i, j] += aux_normal

@ti.kernel
def render(scene: ti.template(), camera: ti.template(), spp: ti.i32, max_depth: ti.i32, buffers: ti.template(), width: ti.i32, height: ti.i32, frame_idx: ti.i32):# type:ignore
    for i, j in buffers.color:
        buffers.color[i, j] = ti.Vector([0.0, 0.0, 0.0])
        buffers.albedo[i, j] = ti.Vector([0.0, 0.0, 0.0])
        buffers.normal[i, j] = ti.Vector([0.0, 0.0, 0.0])
        sampler = RandomSampler(i, j, 0, 0)
        for s in range(spp):

            u = (i + sampler.next()) / width
            v = (j + sampler.next()) / height

            ray = make_ray(camera[None], u, v)
            path_trace(scene, ray, i, j, max_depth, sampler, buffers)

        buffers.color[i, j] /= spp
        buffers.albedo[i, j] /= spp
        buffers.normal[i, j] /= spp

