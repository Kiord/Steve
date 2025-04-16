import taichi as ti
from ray import intersect_scene, Ray, Intersection, visibility
import math
from datatypes import vec3f
from scene import Material, environment_color
from camera import make_ray
from utils import (Contribution, RandomSampler, sample_BSDF, sample_sphere_solid_angle, BSDF, 
                   sample_sphere_uniform, sample_sphere_hemisphere_cosine, sample_sphere_hemisphere_uniform,
                    )
from constants import EPS, MAX_DIST

class RenderBuffers:
    def __init__(self, width:int, height:int):
        self.color = ti.Vector.field(3, ti.f32, shape=(width, height))
        self.albedo = ti.Vector.field(3, ti.f32, shape=(width, height))
        self.normal = ti.Vector.field(3, ti.f32, shape=(width, height))
        self.denoised = ti.Vector.field(3, ti.f32, shape=(width, height))
        self.accum_color = ti.Vector.field(3, ti.f32, shape=(width, height))
        self.final_buffer = ti.Vector.field(3, ti.f32, shape=(width, height))


# @ti.func
# def sample_bsdf_contrib(scene:ti.template(), inter : Intersection, sampler: RandomSampler) -> (vec3f, ti.f32):  # type: ignore
#     contrib = vec3f(0.0, 0.0, 0.0)
#     pdf = 0.0

#     sample = sample_BSDF(inter, inter.ray.direction, sampler)
#     if sample.pdf > 0:
#         ray = Ray(inter.point + inter.normal * 1e-3, sample.direction)
#         inter = intersect_scene(ray, scene, ray.origin, ray.direction, 0.001, 1e5)

#         if inter.hit == 1 and inter.material.emissive.norm() > 0:
#             cos_theta = inter.normal.dot(sample.direction)
#             if cos_theta > 0.0:
#                 contrib = sample.bsdf * inter.material.emissive * cos_theta / sample.pdf
#                 pdf = sample.pdf

#     return contrib, pdf


@ti.func
def sample_light_contrib(scene:ti.template(), inter : Intersection, sampler: RandomSampler) -> Contribution:  # type: ignore
    
    u = sampler.next()
    light_sphere_id = min(int(u * scene.num_light_spheres[None]), scene.num_light_spheres[None]-1)
    sphere_id = scene.light_spheres_id[light_sphere_id]
    light_sphere = scene.spheres[sphere_id]
    mat = scene.materials[light_sphere.material_id]
    light_color = mat.emissive;
    
    sls = sample_sphere_solid_angle(inter.point, light_sphere, sampler)
    #sls = sample_sphere_uniform(inter.point, light_sphere, sampler)
    #sls = sample_sphere_hemisphere_uniform(inter.point, light_sphere, sampler)
    #sls = sample_sphere_hemisphere_cosine(inter.point, light_sphere, sampler)
    
    pdf = sls.pdf / scene.num_light_spheres[None]

    point_to_sample = sls.point - inter.point
    dist = point_to_sample.norm()
    point_to_sample /= dist 
    bsdf = BSDF(mat, inter.normal, -inter.ray.direction, point_to_sample)

    cosi = abs(inter.normal.dot(point_to_sample))
    prod = bsdf * cosi

    value = vec3f(0.0)

    if prod.norm() > 0.0:
        ray_light = Ray(inter.point, point_to_sample)
        inter_light = intersect_scene(ray_light, scene, EPS, dist-EPS)
        v = 1
        if inter_light.hit:# and (inter_light.sphere_id != sphere_id):
            v = 0
        # if sls.normal.dot(point_to_sample) > 0:
        #     v = 0 # ...

        #print(inter.point, sls.point, sls.point-inter.point)
        #v = visibility(scene, inter.point, sls.point)
        #v =  float(inter_light.hit == 0 or abs(inter_light.t - dist) < EPS or inter_light.t < EPS)
        
        #v = visibility(ray_light, scene, sls.point)

        value = v * light_color  * prod

    return Contribution(value, pdf)

@ti.func
def path_trace(scene: ti.template(), ray: Ray, i: ti.i32, j: ti.i32, max_depth: int, sampler: RandomSampler):  # type: ignore
    throughput = ti.Vector([1.0, 1.0, 1.0])
    pdf_total = 1.0
    result = ti.Vector([0.0, 0.0, 0.0])
    aux_albedo = ti.Vector([0.0, 0.0, 0.0])
    aux_normal = ti.Vector([0.0, 0.0, 0.0])

    for bounce in range(max_depth):
        inter = intersect_scene(ray, scene, 0, MAX_DIST)

        if not inter.hit:
            env_color = environment_color(ray.direction, scene)
            result += (throughput / pdf_total) * env_color
            if bounce == 0:
                aux_albedo = env_color
            break

        mat = scene.materials[inter.material_id]

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
        if bounce > 2:
            p = max(throughput.x, throughput.y, throughput.z)
            if sampler.next() > p:
                break
        #     throughput /= p
        #     pdf_total *= p

        ds = sample_BSDF(inter.normal, mat, ray.direction, sampler)
        if ds.pdf < EPS:
            #result = ti.Vector([1000.,0,0])
            break

        cos_theta = max(0.0, inter.normal.dot(ds.direction))
        throughput *= ds.bsdf * cos_theta
        pdf_total *= ds.pdf

        # Update ray
        ray.origin = inter.point + inter.normal * EPS
        ray.direction = ds.direction

        if bounce == 0:
            aux_albedo = mat.albedo
            aux_normal = inter.normal
            # result = 0.5 * ds.direction + 0.5
            # break
    return result, aux_albedo, aux_normal




@ti.kernel
def render(scene: ti.template(), camera: ti.template(), spp: ti.i32, max_depth: ti.i32, buffers: ti.template(), width: ti.i32, height: ti.i32, frame_id: ti.i32):# type:ignore
    for i, j in buffers.color:
        buffers.color[i, j] = ti.Vector([0.0, 0.0, 0.0])
        buffers.albedo[i, j] = ti.Vector([0.0, 0.0, 0.0])
        buffers.normal[i, j] = ti.Vector([0.0, 0.0, 0.0])
        if frame_id == 0:
            buffers.accum_color[i, j] = ti.Vector([0.0, 0.0, 0.0])
        sampler = RandomSampler(i, j, frame_id, 0)
        for s in range(spp):

            u = (i + sampler.next()) / width
            v = (j + sampler.next()) / height

            ray = make_ray(camera[None], u, v)
            color, aux_albedo, aux_normal = path_trace(scene, ray, i, j, max_depth, sampler)
            buffers.color[i, j] += color
            buffers.albedo[i, j] += aux_albedo
            buffers.normal[i, j] += aux_normal
        buffers.color[i, j] /= spp
        buffers.albedo[i, j] /= spp
        buffers.normal[i, j] /= spp
        t1 = 1.0/ (frame_id+1)
        t2 = 1 - t1
        buffers.accum_color[i, j] = buffers.accum_color[i, j] * t2 + buffers.color[i, j] * t1
        buffers.final_buffer[i, j] =  buffers.accum_color[i, j]
