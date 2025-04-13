import taichi as ti
from ray import hit_scene, Ray, Intersection, visibility
import math
from datatypes import vec3f
from scene import Material, environment_color
from camera import make_ray
from utils import (Contribution, RandomSampler, sample_BSDF, sample_sphere_solid_angle, BSDF, 
                   sample_sphere_uniform)
from constants import EPS, MAX_DIST

class RenderBuffers:
    def __init__(self, width:int, height:int):
        self.color = ti.Vector.field(3, ti.f32, shape=(width, height))
        self.albedo = ti.Vector.field(3, ti.f32, shape=(width, height))
        self.normal = ti.Vector.field(3, ti.f32, shape=(width, height))
        self.denoised = ti.Vector.field(3, ti.f32, shape=(width, height))
        self.accum_color = ti.Vector.field(3, ti.f32, shape=(width, height))


@ti.func
def sample_bsdf_contrib(scene:ti.template(), inter : Intersection, sampler: RandomSampler) -> (vec3f, ti.f32):  # type: ignore
    contrib = vec3f(0.0, 0.0, 0.0)
    pdf = 0.0

    sample = sample_BSDF(inter, inter.ray.direction, sampler)
    if sample.pdf > 0:
        ray = Ray(inter.point + inter.normal * 1e-3, sample.direction)
        inter = hit_scene(scene, ray.origin, ray.direction, 0.001, 1e5)

        if inter.hit == 1 and inter.material.emissive.norm() > 0:
            cos_theta = inter.normal.dot(sample.direction)
            if cos_theta > 0.0:
                contrib = sample.bsdf * inter.material.emissive * cos_theta / sample.pdf
                pdf = sample.pdf

    return contrib, pdf


@ti.func
def sample_light_contrib(scene:ti.template(), inter : Intersection, sampler: RandomSampler) -> Contribution:  # type: ignore
    
    u = sampler.next()
    light_sphere_id = min(int(u * scene.num_light_spheres[None]), scene.num_light_spheres[None]-1)
    light_sphere = scene.spheres[scene.light_spheres_id[light_sphere_id]]
    light_color = scene.materials[light_sphere.material_id].emissive;
    
    #sls = sample_sphere_solid_angle(inter.point, light_sphere, sampler)
    sls = sample_sphere_uniform(inter.point, light_sphere, sampler)
    #sls.point += EPS * sls.normal
    
    pdf = sls.pdf / scene.num_light_spheres[None]

    point_to_sample = sls.point - inter.point
    dist = point_to_sample.norm()
    point_to_sample /= dist 
    bsdf = BSDF(inter, -inter.ray.direction, point_to_sample)

    cosi = abs(inter.normal.dot(point_to_sample))
    prod = bsdf * cosi

    value = vec3f(0.0)

    if prod.norm() > 0.0:
        ray_light = Ray(inter.point, point_to_sample)
        inter_light = hit_scene(ray_light, scene, EPS, dist-EPS)

        v =  float(inter_light.hit == 0 or abs(inter_light.t - dist) < EPS or inter_light.t < EPS)
        
        #v = visibility(ray_light, scene, sls.point)

        value = v * light_color * prod

    return Contribution(value, pdf)

@ti.func
def path_trace(scene: ti.template(), ray: Ray, i: ti.i32, j: ti.i32, max_depth: int, sampler: RandomSampler, buffers: ti.template()):  # type: ignore
    throughput = ti.Vector([1.0, 1.0, 1.0])
    pdf_total = 1.0
    result = ti.Vector([0.0, 0.0, 0.0])
    aux_albedo = ti.Vector([0.0, 0.0, 0.0])
    aux_normal = ti.Vector([0.0, 0.0, 0.0])

    for bounce in range(max_depth):
        inter = hit_scene(ray, scene, EPS, MAX_DIST)

        if not inter.hit:
            env_color = environment_color(ray.direction, scene)
            result += (throughput / pdf_total) * env_color
            if bounce == 0:
                aux_albedo = env_color
            break

        mat = inter.material

        light_contrib_color = vec3f(0,0,0)
        use_emissive = mat.emissive.norm() > 0 and bounce == 0
        if use_emissive:
            light_contrib_color = mat.emissive
        else:
            if mat.emissive.norm() == 0:
                light_contrib = sample_light_contrib(scene, inter, sampler)
                light_contrib_color = light_contrib.value / light_contrib.pdf
        
        result += (throughput / pdf_total) * light_contrib_color



        # # Russian roulette after a few bounces
        # if bounce > 2:
        #     p = max(throughput.x, throughput.y, throughput.z)
        #     if sampler.next() > p:
        #         break
        #     throughput /= p
        #     pdf_total *= p

        ds = sample_BSDF(inter.normal, mat, ray.direction, sampler)

        cos_theta = max(0.0, inter.normal.dot(ds.direction))
        throughput *= ds.bsdf * cos_theta
        pdf_total *= ds.pdf

        # Update ray
        ray.origin = inter.point + ds.direction * 1e-4
        ray.direction = ds.direction

        if bounce == 0:
            aux_albedo = mat.diffuse
            aux_normal = inter.normal

    buffers.color[i, j] += result
    buffers.albedo[i, j] += aux_albedo
    buffers.normal[i, j] += aux_normal



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

