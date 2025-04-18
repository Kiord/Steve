import taichi as ti
from scene import Scene, Sphere, Plane, Triangle
from datatypes import vec3f
from constants import EPS, MAX_DIST

@ti.dataclass
class Ray:
    origin: vec3f # type: ignore
    direction: vec3f # type: ignore

@ti.dataclass
class Intersection:
    hit: ti.i32     # type:ignore              
    point: vec3f # type:ignore
    normal: vec3f # type:ignore
    t: ti.f32 # type:ignore
    front_face: ti.i32 # type:ignore
    material_id: ti.i32 # type:ignore
    sphere_id:ti.i32 # type:ignore
    plane_id : ti.i32 # type:ignore
    triangle_id : ti.i32 # type:ignore
    shape_id : ti.i32 # type:ignore  0 sphere, 1 plane, 2 tri 
    ray: Ray

@ti.func
def empty_intersection(ray:Ray) -> Intersection:
    return Intersection(
        hit=0,
        point=vec3f(0.0, 0.0, 0.0),
        normal=vec3f(0.0, 0.0, 0.0),
        t=MAX_DIST,
        front_face=1,
        material_id=-1,
        sphere_id=-1,
        plane_id=-1,
        triangle_id=-1,
        shape_id=-1,
        ray=ray
    )               

@ti.func
def ray_sphere_intersection(ray: Ray, sphere: Sphere, inter: ti.template(), sphere_id: ti.i32, t_min:ti.f32, t_max:ti.f32):  # type: ignore
    oc = ray.origin - sphere.center
    a = ray.direction.dot(ray.direction)
    b = oc.dot(ray.direction)
    c = oc.dot(oc) - sphere.radius * sphere.radius
    discriminant = b * b - a * c

    t_max_ = min(inter.t, t_max)

    if discriminant > 1e-6:
        sqrt_d = ti.sqrt(discriminant)
        root = (-b - sqrt_d) / a
        if root < t_max_ and root > t_min:
            point = ray.origin + root * ray.direction
            outward = (point - sphere.center).normalized()
            front_face = 1 if ray.direction.dot(outward) < 0 else 0
            normal = outward if front_face else -outward

            inter.t = root
            inter.point = point
            inter.normal = normal
            inter.front_face = front_face
            inter.material_id = sphere.material_id
            inter.hit = 1
            inter.sphere_id = sphere_id  # optional for light ID or reuse
            inter.shape_id = 0

@ti.func
def ray_plane_intersection(ray: Ray, plane: Plane, inter: ti.template(), plane_id: ti.i32, t_min: ti.f32, t_max: ti.f32):  # type: ignore
    denom = ray.direction.dot(plane.normal)
    if ti.abs(denom) > 1e-6:  # not parallel
        t = (plane.point - ray.origin).dot(plane.normal) / denom
        t_max_ = ti.min(inter.t, t_max)

        if t > t_min and t < t_max_:
            hit_point = ray.origin + ray.direction * t
            front_face = 1 if denom < 0 else 0
            normal = plane.normal if front_face else -plane.normal

            inter.t = t
            inter.point = hit_point
            inter.normal = normal
            inter.front_face = front_face
            inter.material_id = plane.material_id
            inter.hit = 1
            inter.plane_id = plane_id  # optional
            inter.shape_id = 1

@ti.func
def ray_triangle_intersection(ray: Ray, tri: Triangle, inter: ti.template(), tri_id: ti.i32, t_min: ti.f32, t_max: ti.f32):  # type: ignore
    EPS = 1e-6
    edge1 = tri.v1 - tri.v0
    edge2 = tri.v2 - tri.v0
    h = ray.direction.cross(edge2)
    a = edge1.dot(h)

    if ti.abs(a) > EPS:
        f = 1.0 / a
        s = ray.origin - tri.v0
        u = f * s.dot(h)

        if 0.0 <= u <= 1.0:
            q = s.cross(edge1)
            v = f * ray.direction.dot(q)

            if 0.0 <= v <= 1.0 and u + v <= 1.0:
                t = f * edge2.dot(q)
                t_max_ = ti.min(inter.t, t_max)

                if t > t_min and t < t_max_:
                    hit_point = ray.origin + t * ray.direction
                    normal = edge1.cross(edge2).normalized()
                    front_face = 1 if ray.direction.dot(normal) < 0 else 0
                    final_normal = normal if front_face else -normal

                    inter.t = t
                    inter.point = hit_point
                    inter.normal = final_normal
                    inter.front_face = front_face
                    inter.material_id = tri.material_id
                    inter.hit = 1
                    inter.triangle_id = tri_id
                    inter.shape_id = 2

@ti.func
def ray_scene_intersection(ray: Ray, scene: ti.template(), inter:ti.template(), t_min: ti.f32, t_max: ti.f32):  # type: ignore
    for i in range(scene.num_spheres[None]):
        ray_sphere_intersection(ray, scene.spheres[i], inter, i, t_min, t_max)

    for i in range(scene.num_planes[None]):
        ray_plane_intersection(ray, scene.planes[i], inter, i, t_min, t_max)

    for i in range(scene.num_triangles[None]):
        ray_triangle_intersection(ray, scene.triangles[i], inter, i, t_min, t_max)

@ti.func
def intersect_scene(ray: Ray, scene: ti.template(), t_min: ti.f32, t_max: ti.f32):  # type: ignore
    inter = empty_intersection(ray)
    ray_scene_intersection(ray, scene, inter, t_min, t_max)
    return inter

@ti.func
def visibility(scene:ti.template(), a:vec3f, b:vec3f)->ti.f32: #type: ignore
    ab = b - a
    dist = ab.norm()
    ab /= dist
    ray = Ray(a, ab)
    inter = intersect_scene(ray, scene, EPS, dist-EPS)
    return float(inter.hit == 0 or (inter.point-b).norm() < EPS * 10)


# @ti.func
# def hit_sphere(ray: Ray, sphere: Sphere, t_min, t_max):
#     oc = ray.origin - sphere.center
#     a = ray.direction.dot(ray.direction)
#     b = 2.0 * oc.dot(ray.direction)
#     c = oc.dot(oc) - sphere.radius * sphere.radius
#     discriminant = b * b - 4 * a * c

#     inter = empty_intersection()

#     if discriminant > 0:
#         sqrt_d = ti.sqrt(discriminant)
#         root = (-b - sqrt_d) / (2.0 * a)
#         if t_min < root < t_max:
#             inter.t = root
#             inter.point = ray.origin + root * ray.direction
#             outward = (inter.point - sphere.center).normalized()
#             inter.normal = outward
#             inter.front_face = 1 if ray.direction.dot(outward) < 0 else 0
#             inter.material_id = sphere.material_id
#             inter.ray = ray
#             if inter.front_face == 0:
#                 inter.normal = -inter.normal
#             inter.hit = 1

#     return inter


# @ti.func
# def hit_plane(ray : Ray, plane: Plane, t_min, t_max):
#     inter = empty_intersection()

#     denom = ray.direction.dot(plane.normal)
#     if abs(denom) > 1e-6:
#         t = (plane.point - ray.origin).dot(plane.normal) / denom
#         if t_min < t < t_max:
#             inter.t = t
#             inter.point = ray.origin + t * ray.direction
#             inter.front_face = 1 if denom < 0 else 0
#             inter.normal = plane.normal if inter.front_face else -plane.normal
#             inter.hit=1
#             inter.ray=ray
#             inter.material_id = plane.material_id

#     return inter


# @ti.func
# def hit_scene(ray:Ray, scene: ti.template(), t_min, t_max): # type: ignore
#     closest = t_max

#     final_inter = empty_intersection() 

#     for i in range(scene.num_spheres[None]):
#         inter = hit_sphere(ray, scene.spheres[i], t_min, closest)
#         if inter.hit:
#             closest = inter.t
#             final_inter = inter

#     for i in range(scene.num_planes[None]):
#         inter = hit_plane(ray, scene.planes[i], t_min, closest)
#         if inter.hit:
#             closest = inter.t
#             final_inter = inter
    
#     # if final_inter.hit:
#     #     final_inter.material = scene.materials[final_inter.material_id]

#     return final_inter

# @ti.func
# def visibility(ray:Ray, scene: ti.template(), point:vec3f): # type: ignore
#     inter = hit_scene(ray, scene, EPS, MAX_DIST)
#     #delta_start = (inter.point - ray.origin).norm() 
#     delta_finish = (inter.point - point).norm() 
#     return float(inter.hit == 0 or delta_finish < EPS)


