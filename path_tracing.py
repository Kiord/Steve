import taichi as ti
from ray import intersect_scene, Ray, Intersection, visibility
import math
from datatypes import vec3f
from scene import Material, environment_color
from camera import make_ray
from utils import (Contribution, RandomSampler, sample_BSDF, sample_sphere_solid_angle, BSDF, PDF, PDF_solid_angle_sphere,
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


@ti.func
def sample_bsdf_contrib(scene:ti.template(), inter : Intersection, sampler: RandomSampler) -> Contribution:  # type: ignore
    contrib= Contribution(vec3f(0.0, 0.0, 0.0), 0.0)
    mat = scene.materials[inter.material_id]
    ds = sample_BSDF(inter.normal, mat, inter.ray.direction, sampler)
    pdf_light = 0
    if ds.pdf > EPS:
        ray = Ray(inter.point + inter.normal * EPS, ds.direction)
        inter_check = intersect_scene(ray, scene, EPS, MAX_DIST)
        mat_check = scene.materials[inter_check.material_id]
        if inter_check.hit == 1 and mat_check.emissive.norm() > 0:
            
            light_sphere = scene.spheres[inter_check.sphere_id]
            N = ti.cast(scene.num_light_spheres[None], ti.f32)
            pdf_light = PDF_solid_angle_sphere(light_sphere, inter.point) / N

            cos_theta = inter.normal.dot(ds.direction)
            if cos_theta > 0.0:
                contrib.value = ds.bsdf * mat_check.emissive * cos_theta / ds.pdf
                contrib.pdf = ds.pdf

    return contrib, pdf_light


@ti.func
def sample_light_contrib(scene:ti.template(), inter : Intersection, sampler: RandomSampler) -> Contribution:  # type: ignore
    
    u = sampler.next()
    light_sphere_id = min(int(u * scene.num_light_spheres[None]), scene.num_light_spheres[None]-1)
    sphere_id = scene.light_spheres_id[light_sphere_id]
    light_sphere = scene.spheres[sphere_id]
    emissive_mat = scene.materials[light_sphere.material_id]
    light_color = emissive_mat.emissive
    inter_mat = scene.materials[inter.material_id]
    
    sls = sample_sphere_solid_angle(inter.point, light_sphere, sampler)
    #sls = sample_sphere_uniform(inter.point, light_sphere, sampler)
    #sls = sample_sphere_hemisphere_uniform(inter.point, light_sphere, sampler)
    #sls = sample_sphere_hemisphere_cosine(inter.point, light_sphere, sampler)
    
    pdf = sls.pdf / scene.num_light_spheres[None]

    point_to_sample = sls.point - inter.point
    dist = point_to_sample.norm()
    point_to_sample /= dist 
    bsdf = BSDF(inter_mat, inter.normal, -inter.ray.direction, point_to_sample)

    pdf_surface = PDF(inter_mat, inter.normal, -inter.ray.direction, point_to_sample)

    cosi = abs(inter.normal.dot(point_to_sample))
    prod = bsdf * cosi

    value = vec3f(0.0)

    if prod.norm() > 0.0:
        ray_light = Ray(inter.point, point_to_sample)
        inter_light = intersect_scene(ray_light, scene, EPS, dist-EPS)
        v = float(inter_light.hit == 0) 
        value = v * light_color  * prod / pdf

    return Contribution(value, pdf), pdf_surface

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

    for bounce in range(max_depth):
        inter = intersect_scene(ray, scene, 0, MAX_DIST)

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

        ds = sample_BSDF(inter.normal, mat, ray.direction, sampler)
        if ds.pdf < EPS:
            #result = ti.Vector([1000.,0,0])
            break

        cos_theta = max(0.0, inter.normal.dot(ds.direction))
        prod_color *= ds.bsdf * cos_theta
        prod_pdf *= ds.pdf
        prod_ratio = prod_color / ti.max(prod_pdf, EPS)

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
            color, aux_albedo, aux_normal = path_trace(scene, ray, max_depth, sampler)
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
