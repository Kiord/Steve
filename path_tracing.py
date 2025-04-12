import taichi as ti
from constants import MAX_DEPTH
from ray import hit_scene
import math

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
def path_trace(scene: ti.template(), ray_o, ray_d, i: ti.i32, j: ti.i32, s: ti.i32, buffers: ti.template()): # type:ignore
    throughput = ti.Vector([1.0, 1.0, 1.0])
    result = ti.Vector([0.0, 0.0, 0.0])
    aux_albedo = ti.Vector([0.0, 0.0, 0.0])
    aux_normal = ti.Vector([0.0, 0.0, 0.0])
    background = ti.Vector([0.75, 0.75, 0.75])

    for bounce in range(MAX_DEPTH):
        hit, rec = hit_scene(scene, ray_o, ray_d, 0.001, 1e5)
        if not hit:
            result += throughput * background
            aux_albedo = background
            break

        mat = scene.materials[rec.material_id]
        light_dir = (scene.light[None].position - rec.point).normalized()
        light_dist = (scene.light[None].position - rec.point).norm()

        # shadow ray
        blocked, _ = hit_scene(scene, rec.point + rec.normal * 1e-3, light_dir, 0.001, light_dist - 1e-2)
        if not blocked:
            lambert = max(0.0, rec.normal.dot(light_dir))
            light_contrib = (mat.diffuse * lambert + mat.specular) * scene.light[None].color / (light_dist**2)
            result += throughput * light_contrib

        # sample direction
        onb_u = rec.normal.cross(ti.Vector([0.0, 1.0, 0.0])).normalized()
        if onb_u.norm() < 1e-3:
            onb_u = rec.normal.cross(ti.Vector([1.0, 0.0, 0.0])).normalized()
        onb_v = rec.normal.cross(onb_u)

        r1 = 2 * math.pi * spatial_random(i, j, bounce, s)
        r2 = spatial_random(i, j, bounce, s)
        r2s = ti.sqrt(r2)

        local_dir = ti.Vector([
            ti.cos(r1) * r2s,
            ti.sqrt(1 - r2),
            ti.sin(r1) * r2s
        ])
        new_dir = (onb_u * local_dir.x + rec.normal * local_dir.y + onb_v * local_dir.z).normalized()

        ray_o = rec.point + rec.normal * 1e-4
        ray_d = new_dir
        throughput *= mat.diffuse

        if bounce == 0:
            aux_albedo = mat.diffuse
            aux_normal = rec.normal

        if bounce > 2:
            p = max(throughput.x, throughput.y, throughput.z)
            if spatial_random(i, j, bounce, s) > p:
                break
            throughput /= p

    buffers.color[i, j] += result
    buffers.albedo[i, j] += aux_albedo
    buffers.normal[i, j] += aux_normal

@ti.kernel
def render(scene: ti.template(), camera: ti.template(), spp: ti.i32, buffers: ti.template(), width: ti.i32, height: ti.i32): # type: ignore
    for i, j in buffers.color:
        buffers.color[i, j] = ti.Vector([0.0, 0.0, 0.0])
        buffers.albedo[i, j] = ti.Vector([0.0, 0.0, 0.0])
        buffers.normal[i, j] = ti.Vector([0.0, 0.0, 0.0])

        for s in range(spp):
            u = (i + spatial_random(i, j, 0, s)) / width
            v = (j + spatial_random(i, j, 1, s)) / height
            ray_o = camera[None].origin
            ray_d = normalize(
                camera[None].lower_left_corner + u * camera[None].horizontal + v * camera[None].vertical
                - camera[None].origin
            )
            path_trace(scene, ray_o, ray_d, i, j, s, buffers)

        # Final averaging
        buffers.color[i, j] /= spp
        buffers.albedo[i, j] /= spp
        buffers.normal[i, j] /= spp

