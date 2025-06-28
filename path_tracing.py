import taichi as ti
from ray import intersect_scene, Ray, Intersection
from datatypes import vec3f
from scene import  environment_color
from camera import make_ray
from utils import (Contribution, RandomSampler,  pdf_solid_angle_sphere, LightSample)
from constants import *
from bsdf import bsdf_sample, bsdf_pdf
from emitters import emitter_sample


class RenderBuffers:
    def __init__(self, width:int, height:int):
        self.color = ti.Vector.field(3, ti.f32, shape=(width, height))
        self.albedo = ti.Vector.field(3, ti.f32, shape=(width, height))
        self.normal = ti.Vector.field(3, ti.f32, shape=(width, height))
        self.bvh_depth = ti.field(ti.f32, shape=(width, height))
        self.box_test_count = ti.field(ti.f32, shape=(width, height))
        self.depth = ti.field(ti.f32, shape=(width, height))
        self.denoised = ti.Vector.field(3, ti.f32, shape=(width, height))
        self.accum_color = ti.Vector.field(3, ti.f32, shape=(width, height))
        self.final_buffer = ti.Vector.field(3, ti.f32, shape=(width, height))


@ti.func
def sample_bsdf_contrib(scene:ti.template(), inter : Intersection, sampler: RandomSampler) -> Contribution:  # type: ignore
    contrib= Contribution(vec3f(0.0, 0.0, 0.0), 0.0)
    mat = scene.materials[inter.material_id]
    #ds = sample_BSDF(inter.normal, mat, inter.ray.direction, sampler)
    ds = bsdf_sample(mat, inter.normal, inter.ray.direction, sampler)
    pdf_light = 0.0
    if ds.pdf > EPS:
        ray = Ray(inter.point + inter.normal * EPS, ds.direction)
        inter_check = intersect_scene(ray, scene, EPS, MAX_DIST)
        mat_check = scene.materials[inter_check.material_id]
        if inter_check.hit == 1 and mat_check.emissive.norm() > 0:
            
            light_sphere = scene.spheres[inter_check.sphere_id]

            pdf_light = pdf_solid_angle_sphere(light_sphere, inter.point) / scene.num_light_spheres[None]

            cos_theta = inter.normal.dot(ds.direction)
            if cos_theta > 0.0:
                contrib.value = ds.bsdf * mat_check.emissive * cos_theta / ds.pdf
                contrib.pdf = ds.pdf

    return contrib, pdf_light

@ti.func
def sample_light_contrib(scene: ti.template(), inter: Intersection, sampler: RandomSampler):
    total_lights = scene.num_light_spheres[None] + scene.num_light_triangles[None]
    ls = LightSample(direction=vec3f(0.0), contrib=vec3f(0.0), pdf=0.0)

    if total_lights > 0:
        u = sampler.next()
        light_index = int(u * total_lights)

        emitter_type = EMITTER_SPHERE
        if light_index >= scene.num_light_spheres[None]:
            emitter_type = EMITTER_TRIANGLE

        ls = emitter_sample(emitter_type, scene, inter, sampler)

    # Compute PDF_surface (for BSDF MIS)
    wi = -inter.ray.direction
    wo = ls.direction
    mat = scene.materials[inter.material_id]
    pdf_surface = bsdf_pdf(mat, inter.normal, wi, wo)

    return Contribution(ls.contrib, ls.pdf), pdf_surface


@ti.func
def balance_heuristic(pdf_a: ti.f32, pdf_b: ti.f32) -> ti.f32:# type: ignore
    denom = pdf_a + pdf_b
    return min(1,max(0,pdf_a / ti.max(denom, EPS)))

@ti.func
def sample_mis_contrib(scene: ti.template(), inter: Intersection, sampler: RandomSampler) -> vec3f:  # type: ignore
    light_contrib, pdf_surface = sample_light_contrib(scene, inter, sampler)
    bsdf_contrib, pdf_light = sample_bsdf_contrib(scene, inter, sampler)
    # Compute weights
    w_light = balance_heuristic(light_contrib.pdf, pdf_surface)
    w_bsdf  = balance_heuristic(bsdf_contrib.pdf, pdf_light)
    # w_light = balance_heuristic(light_contrib.pdf, bsdf_contrib.pdf)
    # w_bsdf  = balance_heuristic(bsdf_contrib.pdf, light_contrib.pdf)

    return  w_light * light_contrib.value + w_bsdf * bsdf_contrib.value
    #return vec3f(light_contrib.pdf, bsdf_contrib.pdf, 0)

@ti.func
def path_trace(scene: ti.template(), ray: Ray, max_depth: int, sampler: RandomSampler):  # type: ignore
    prod_color = vec3f(1.0)
    prod_ratio = vec3f(1.0)
    prod_pdf = 1.0
    result = ti.Vector([0.0, 0.0, 0.0])
    aux_albedo = ti.Vector([0.0, 0.0, 0.0])
    aux_normal = ti.Vector([0.0, 0.0, 0.0])
    aux_depth = MAX_DIST
    aux_bvh_depth = 0
    aux_box_test_count = 0

    for bounce in range(max_depth):
        inter = intersect_scene(ray, scene, 0, MAX_DIST)

        if bounce == 0:
            aux_box_test_count = inter.box_test_count

        if not inter.hit:
            env_color = environment_color(ray.direction, scene)
            result += prod_ratio * env_color
            if bounce == 0:
                aux_albedo = env_color
            break

        mat = scene.materials[inter.material_id]

        contrib = vec3f(0,0,0)
        use_emissive = mat.emissive.norm() > 0 and bounce == 0
        if use_emissive:
            contrib = mat.emissive
        else:
            if mat.emissive.norm() == 0:
                #light_sampling_contrib = sample_light_contrib(scene, inter, sampler)
                #contrib = light_sampling_contrib.value / light_sampling_contrib.pdf
                contrib = sample_mis_contrib(scene, inter, sampler)
        
        result += prod_ratio * contrib

        #result += prod_ratio * mat.emissive


        # Russian roulette after a few bounces
        # if bounce > 2:
        #     p = max(throughput.x, throughput.y, throughput.z, EPS)
        #     if sampler.next() > p:
        #         break
        #     throughput /= p
        #     pdf_total *= p

        #ds = sample_BSDF(inter.normal, mat, ray.direction, sampler)
        ds = bsdf_sample(mat, inter.normal, ray.direction, sampler)

        if ds.pdf < EPS:
            #result = ti.Vector([1000.,0,0])
            break

        cos_theta = max(0.0, inter.normal.dot(ds.direction))
        prod_color *= ds.bsdf * cos_theta
        prod_pdf *= ds.pdf
        prod_ratio = prod_color / ti.max(prod_pdf, EPS)

        # Update ray
        ray.origin = inter.point + inter.normal_geom * EPS
        ray.direction = ds.direction

        if bounce == 0:
            aux_albedo = mat.albedo
            aux_normal = inter.normal
            aux_depth = inter.t
            aux_bvh_depth = inter.bvh_depth

    return result, aux_albedo, aux_normal, aux_depth, aux_bvh_depth, aux_box_test_count




@ti.kernel
def render(scene: ti.template(), camera: ti.template(), spp: ti.i32, max_depth: ti.i32, buffers: ti.template(), width: ti.i32, height: ti.i32, frame_id: ti.i32):# type:ignore
    for i, j in buffers.color:
        buffers.color[i, j] = ti.Vector([0.0, 0.0, 0.0])
        buffers.albedo[i, j] = ti.Vector([0.0, 0.0, 0.0])
        buffers.normal[i, j] = ti.Vector([0.0, 0.0, 0.0])
        buffers.depth[i, j] = 0.0
        buffers.bvh_depth[i, j] = 0.0
        buffers.box_test_count[i, j] = 0.0
        if frame_id == 0:
            buffers.accum_color[i, j] = ti.Vector([0.0, 0.0, 0.0])
        sampler = RandomSampler(i, j, frame_id, 0)
        for s in range(spp):

            u = (i + sampler.next()) / width
            v = (j + sampler.next()) / height

            ray = make_ray(camera[None], u, v)
            color, aux_albedo, aux_normal, aux_depth, aux_bvh_depth, aux_box_test_count = path_trace(scene, ray, max_depth, sampler)
            buffers.color[i, j] += color
            buffers.albedo[i, j] += aux_albedo
            buffers.normal[i, j] += aux_normal
            buffers.depth[i, j] += aux_depth
            buffers.bvh_depth[i, j] += aux_bvh_depth
            buffers.box_test_count[i, j] += aux_box_test_count
        buffers.color[i, j] /= spp
        buffers.albedo[i, j] /= spp
        buffers.normal[i, j] /= spp
        buffers.depth[i, j] /= spp
        buffers.bvh_depth[i, j] /= spp
        buffers.box_test_count[i, j] /= spp
        t1 = 1.0/ (frame_id+1)
        t2 = 1 - t1
        buffers.accum_color[i, j] = buffers.accum_color[i, j] * t2 + buffers.color[i, j] * t1
        buffers.final_buffer[i, j] =  buffers.accum_color[i, j]
