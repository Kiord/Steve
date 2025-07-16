import taichi as ti
from ray import intersect_scene, Ray, Intersection
from datatypes import vec3f
from scene import  environment_color
from camera import make_ray
from utils import (Contribution, RandomSampler)
from constants import *
from bsdf import bsdf_sample, bsdf_pdf
from emitters import emitter_sample, pdf_solid_angle


class RenderBuffers:
    def __init__(self, width:int, height:int):
        self.color = ti.Vector.field(3, ti.f32, shape=(width, height))
        self.accum_color = ti.Vector.field(3, ti.f32, shape=(width, height))

        self.direct_light = ti.Vector.field(3, ti.f32, shape=(width, height))
        self.direct_bsdf = ti.Vector.field(3, ti.f32, shape=(width, height))
        self.mis_weights = ti.Vector.field(3, ti.f32, shape=(width, height))
        self.accum_direct_light = ti.Vector.field(3, ti.f32, shape=(width, height))
        self.accum_direct_bsdf = ti.Vector.field(3, ti.f32, shape=(width, height))
        self.accum_mis_weights = ti.Vector.field(3, ti.f32, shape=(width, height))

        self.albedo = ti.Vector.field(3, ti.f32, shape=(width, height))
        self.normal = ti.Vector.field(3, ti.f32, shape=(width, height))
        self.bvh_depth = ti.field(ti.f32, shape=(width, height))
        self.box_test_count = ti.field(ti.f32, shape=(width, height))
        self.depth = ti.field(ti.f32, shape=(width, height))
        
        self.denoised = ti.Vector.field(3, ti.f32, shape=(width, height))
    
        self.final_buffer = ti.Vector.field(3, ti.f32, shape=(width, height))


@ti.func
def sample_direct_bsdf_contrib_mis(scene:ti.template(), inter : Intersection, sampler: RandomSampler) -> Contribution:  # type: ignore
    contrib= Contribution(vec3f(0.0, 0.0, 0.0), 0.0)
    mat = scene.materials[inter.material_id]
    ds = bsdf_sample(mat, inter.normal, inter.ray.direction, sampler)
    cos_theta = inter.normal.dot(ds.direction)
    pdf_light = 0.0
    if cos_theta > 0.0 and ds.pdf > EPS:

        ray = Ray(inter.point + inter.normal * EPS, ds.direction)
        inter_check = intersect_scene(ray, scene, EPS, MAX_DIST)
        mat_check = scene.materials[inter_check.material_id]
        if inter_check.hit == 1 and mat_check.emissive.norm() > 0:
            
            pdf_light = pdf_solid_angle(scene, inter_check)

            contrib.value = ds.bsdf * mat_check.emissive * cos_theta / ds.pdf
            contrib.pdf = ds.pdf

    return contrib, pdf_light

@ti.func
def sample_direct_light_contrib_mis(scene: ti.template(), inter: Intersection, sampler: RandomSampler):
    ls = emitter_sample(scene, inter, sampler)

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
def sample_direct_mis_contrib(scene: ti.template(), inter: Intersection, sampler: RandomSampler) -> vec3f:  # type: ignore
    light_contrib, pdf_surface = sample_direct_light_contrib_mis(scene, inter, sampler)
    bsdf_contrib, pdf_light = sample_direct_bsdf_contrib_mis(scene, inter, sampler)
    # Compute weights
    w_light = balance_heuristic(light_contrib.pdf, pdf_surface)
    w_bsdf  = balance_heuristic(bsdf_contrib.pdf, pdf_light)
    
    return w_light, light_contrib.value, w_bsdf, bsdf_contrib.value
    #return w_light * vec3f(1,0,0) + w_bsdf * vec3f(0,1,0)
    #return vec3f(light_contrib.pdf, bsdf_contrib.pdf, 0)

@ti.func
def path_trace(scene: ti.template(), ray: Ray, max_depth: int, sampler: RandomSampler):  # type: ignore
    prod_color = vec3f(1.0)
    prod_ratio = vec3f(1.0)
    prod_pdf = 1.0
    result = ti.Vector([0.0, 0.0, 0.0])
    aux_mis_weights = ti.Vector([0.0, 0.0, 0.0])
    aux_direct_light = ti.Vector([0.0, 0.0, 0.0])
    aux_direct_bsdf = ti.Vector([0.0, 0.0, 0.0])
    result = ti.Vector([0.0, 0.0, 0.0])
    aux_albedo = ti.Vector([0.0, 0.0, 0.0])
    aux_normal = ti.Vector([0.0, 0.0, 0.0])
    aux_depth = MAX_DIST
    aux_bvh_depth = 0
    aux_box_test_count = 0

    for bounce in range(max_depth):
        inter = intersect_scene(ray, scene, 0, MAX_DIST)
        mat = scene.materials[inter.material_id]

        if bounce == 0:
            aux_albedo = mat.albedo
            aux_normal = inter.normal
            aux_depth = inter.t
            aux_bvh_depth = inter.bvh_depth
        
        if bounce == 0:
            aux_box_test_count = inter.box_test_count

        if not inter.hit:
            env_color = environment_color(ray.direction, scene)
            result += prod_ratio * env_color
            if bounce == 0:
                aux_albedo = env_color
            break
        

        contrib = vec3f(0,0,0)
        use_emissive = mat.emissive.norm() > 0 and bounce == 0
        if use_emissive:
            contrib = mat.emissive
        elif mat.emissive.norm() == 0:
            w_light, light_contrib, w_bsdf, bsdf_contrib = sample_direct_mis_contrib(scene, inter, sampler)
            contrib = w_light * light_contrib + w_bsdf * bsdf_contrib
            aux_mis_weights[0] += w_light 
            aux_mis_weights[1] += w_bsdf
            aux_direct_light += light_contrib * prod_ratio
            aux_direct_bsdf += bsdf_contrib * prod_ratio
        
        result += prod_ratio * contrib

        # Russian roulette after a few bounces
        # if bounce > 2:
        #     p = max(throughput.x, throughput.y, throughput.z, EPS)
        #     if sampler.next() > p:
        #         break
        #     throughput /= p
        #     pdf_total *= p

        ds = bsdf_sample(mat, inter.normal, ray.direction, sampler)

        if ds.pdf < EPS:
            break

        cos_theta = max(0.0, inter.normal.dot(ds.direction))
        prod_color *= ds.bsdf * cos_theta
        prod_pdf *= ds.pdf
        prod_ratio = prod_color / ti.max(prod_pdf, EPS)

        # Update ray
        ray.origin = inter.point + inter.normal_geom * EPS
        ray.direction = ds.direction

    return result, aux_mis_weights, aux_direct_light, aux_direct_bsdf, aux_albedo, aux_normal, aux_depth, aux_bvh_depth, aux_box_test_count

@ti.kernel
def reset_accum_buffers(buffers: ti.template()): # type:ignore
    for i, j in buffers.accum_color:
        buffers.accum_color[i, j] = ti.Vector([0.0, 0.0, 0.0])
        buffers.accum_mis_weights[i, j] = ti.Vector([0.0, 0.0, 0.0])
        buffers.accum_direct_light[i, j] = ti.Vector([0.0, 0.0, 0.0])
        buffers.accum_direct_bsdf[i, j] = ti.Vector([0.0, 0.0, 0.0])


@ti.kernel
def render(scene: ti.template(), camera: ti.template(), spp: ti.i32, max_depth: ti.i32, buffers: ti.template(), width: ti.i32, height: ti.i32, frame_id: ti.i32):# type:ignore
    for i, j in buffers.color:
        buffers.color[i, j] = ti.Vector([0.0, 0.0, 0.0])
        buffers.mis_weights[i, j] = ti.Vector([0.0, 0.0, 0.0])
        buffers.direct_light[i, j] = ti.Vector([0.0, 0.0, 0.0])
        buffers.direct_bsdf[i, j] = ti.Vector([0.0, 0.0, 0.0])
        buffers.albedo[i, j] = ti.Vector([0.0, 0.0, 0.0])
        buffers.normal[i, j] = ti.Vector([0.0, 0.0, 0.0])
        buffers.depth[i, j] = 0.0
        buffers.bvh_depth[i, j] = 0.0
        buffers.box_test_count[i, j] = 0.0
        sampler = RandomSampler(i, j, frame_id, 0)
        for s in range(spp):

            u = (i + sampler.next()) / width
            v = (j + sampler.next()) / height
            ray = make_ray(camera[None], u, v)
            
            color, aux_mis_weights, aux_direct_light, aux_direct_bsdf, aux_albedo, aux_normal, aux_depth, aux_bvh_depth, aux_box_test_count = path_trace(scene, ray, max_depth, sampler)
            
            buffers.color[i, j] += color
            buffers.mis_weights[i, j] += aux_mis_weights
            buffers.direct_light[i, j] += aux_direct_light
            buffers.direct_bsdf[i, j] += aux_direct_bsdf
            buffers.albedo[i, j] += aux_albedo
            buffers.normal[i, j] += aux_normal
            buffers.depth[i, j] += aux_depth
            buffers.bvh_depth[i, j] += aux_bvh_depth
            buffers.box_test_count[i, j] += aux_box_test_count

        buffers.color[i, j] /= spp
        buffers.mis_weights[i, j] /= spp 
        buffers.direct_light[i, j] /= spp 
        buffers.direct_bsdf[i, j] /= spp 
        buffers.albedo[i, j] /= spp
        buffers.normal[i, j] /= spp
        buffers.depth[i, j] /= spp
        buffers.bvh_depth[i, j] /= spp
        buffers.box_test_count[i, j] /= spp
        t1 = 1.0/ (frame_id+1)
        t2 = 1 - t1
        buffers.accum_color[i, j] = buffers.accum_color[i, j] * t2 + buffers.color[i, j] * t1
        buffers.accum_mis_weights[i, j] = buffers.accum_mis_weights[i, j] * t2 + buffers.mis_weights[i, j] * t1
        buffers.accum_direct_light[i, j] = buffers.accum_direct_light[i, j] * t2 + buffers.direct_light[i, j] * t1
        buffers.accum_direct_bsdf[i, j] = buffers.accum_direct_bsdf[i, j] * t2 + buffers.direct_bsdf[i, j] * t1
        #buffers.final_buffer[i, j] =  buffers.accum_color[i, j]
