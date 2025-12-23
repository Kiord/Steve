import taichi as ti
from scene import Scene, Sphere, Plane, Triangle
from datatypes import vec3f, mat4f
from constants import *

@ti.dataclass
class Ray:
    origin: vec3f # type: ignore
    direction: vec3f # type: ignore

@ti.dataclass
class Intersection:
    hit: ti.i32     # type:ignore              
    point: vec3f # type:ignore
    normal: vec3f # type:ignore
    normal_geom: vec3f # type:ignore
    t: ti.f32 # type:ignore
    front_face: ti.i32 # type:ignore
    material_id: ti.i32 # type:ignore
    primitive_id:ti.i32 # type:ignore
    primitive_type : ti.i32 # type:ignore
    ray: Ray
    bvh_depth : ti.int32 # type:ignore
    box_test_count: ti.int32 # type:ignore

@ti.func
def transform_point(p_local, transform: mat4f ): # type: ignore
    p_h = ti.Vector([p_local[0], p_local[1], p_local[2], 1.0])
    return (transform @ p_h).xyz

@ti.func
def transform_direction(d_local, transform: mat4f ): # type: ignore
    d_h = ti.Vector([d_local[0], d_local[1], d_local[2], 0.0])
    return (transform @ d_h).xyz.normalized()

@ti.func
def transform_ray(ray: Ray, transform: mat4f ): # type: ignore
    new_origin = transform_point(ray.origin, transform)
    new_direction = transform_direction(ray.direction, transform)
    return Ray(new_origin, new_direction)

@ti.func
def empty_intersection(ray:Ray) -> Intersection:
    return Intersection(
        hit=0,
        point=vec3f(0.0, 0.0, 0.0),
        normal=vec3f(0.0, 0.0, 0.0),
        normal_geom=vec3f(0.0, 0.0, 0.0),
        t=MAX_DIST,
        front_face=1,
        material_id=-1,
        primitive_id=-1,
        primitive_type=PRIMITIVE_NONE,
        ray=ray,
        bvh_depth=0,
        box_test_count=0,
    )               

@ti.func
def ray_sphere_intersection(ray: Ray, sphere: Sphere, inter: ti.template(), primitive_id: ti.i32, t_min:ti.f32, t_max:ti.f32):  # type: ignore
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
            inter.normal_geom = normal
            inter.front_face = front_face
            inter.material_id = sphere.material_id
            inter.hit = 1
            inter.primitive_id = primitive_id  # optional for light ID or reuse
            inter.primitive_type = PRIMITIVE_SPHERE

@ti.func
def ray_plane_intersection(ray: Ray, plane: Plane, inter: ti.template(), primitive_id: ti.i32, t_min: ti.f32, t_max: ti.f32):  # type: ignore
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
            inter.normal_geom = normal
            inter.front_face = front_face
            inter.material_id = plane.material_id
            inter.hit = 1
            inter.primitive_id = primitive_id  # optional
            inter.primitive_type = PRIMITIVE_PLANE

@ti.func
def ray_triangle_intersection(ray: Ray, tri: Triangle, inter: ti.template(), tri_id: ti.i32, t_min: ti.f32, t_max: ti.f32, bvh_depth:ti.i32=0):  # type: ignore
    edge1 = tri.v1 - tri.v0
    edge2 = tri.v2 - tri.v0
    h = ray.direction.cross(edge2)
    a = edge1.dot(h)

    edge1_norm = edge1.norm()
    h_norm = h.norm()
    relative_eps  = EPS * edge1_norm * h_norm + EPS  # small absolute epsilon to cover near-zero

    if ti.abs(a) > relative_eps :
        f = 1.0 / a
        s = ray.origin - tri.v0
        u = f * s.dot(h)

        if 0.0 <= u <= 1.0:
            q = s.cross(edge1)
            v = f * ray.direction.dot(q)
            
            if 0.0 <= v <= 1.0 and u + v <= 1.0 + relative_eps :
                t = f * edge2.dot(q)
                t_max_ = ti.min(inter.t, t_max)

                if t > t_min and t < t_max_:
                    hit_point = ray.origin + t * ray.direction
                    w = 1 - u - v

                    normal = (w * tri.n0 + u * tri.n1 + v * tri.n2).normalized()
                    front_face = 1 if ray.direction.dot(tri.normal) < 0 else 0
                    final_normal = normal if front_face else -normal
                    inter.t = t
                    inter.point = hit_point
                    inter.normal = final_normal
                    inter.normal_geom = tri.normal
                    inter.front_face = front_face
                    inter.material_id = tri.material_id
                    inter.hit = 1
                    inter.primitive_id = tri_id
                    inter.primitive_type = PRIMITIVE_TRIANGLE
                    inter.bvh_depth = bvh_depth

@ti.func
def inflate_aabb(aabb_min, aabb_max, value):
    aabb_centroid = 0.5 * aabb_min + 0.5 * aabb_max
    direction = (aabb_max - aabb_centroid).normalized()
    aabb_max = aabb_max + value * direction
    aabb_min = aabb_min - value * direction
    return aabb_min, aabb_max

@ti.func
def ray_aabb_intersection_test(ray, aabb_min, aabb_max, t_min, t_max):
    
    t0 = t_min
    t1 = t_max
    hit = True

    for i in ti.static(range(3)):
        inv_d = 1.0 / ray.direction[i]
        t_near = (aabb_min[i] - ray.origin[i]) * inv_d
        t_far = (aabb_max[i] - ray.origin[i]) * inv_d

        if inv_d < 0.0:
            t_near, t_far = t_far, t_near

        t0 = ti.max(t0, t_near)
        t1 = ti.min(t1, t_far)

        if t1 < t0:
            hit = False

    if not hit:
        t0 = MAX_DIST
    return hit, t0



@ti.func
def ray_bvhs_intersection(ray: Ray, bvhs:ti.template(), bvh_infos:ti.template(), num_bvhs:ti.int32, triangles:ti.template(), inter:ti.template(), t_min: ti.f32, t_max: ti.f32): # type: ignore
    
    stack = ti.Vector([0 for _ in range(64)])  # fixed-size stack
    for bvh_id in range(num_bvhs):
        stack_ptr = 0
        stack[0] = 0  # start from BVH root

        aabb_min = bvhs.aabb_min[bvh_id, 0]
        aabb_max = bvhs.aabb_max[bvh_id, 0]

        bvh_transform = bvh_infos.transform[bvh_id]
        bvh_inv_transform = bvh_infos.inv_transform[bvh_id]

        ray_local = transform_ray(ray, bvh_inv_transform)

        hit_aabb, t_aabb = ray_aabb_intersection_test(ray_local, aabb_min, aabb_max, t_min, inter.t)

        if not hit_aabb or t_aabb > inter.t:
            continue
        
        inter_local = empty_intersection(ray_local)

        while stack_ptr >= 0:
            node_id = stack[stack_ptr]
            stack_ptr -= 1

            if bvhs.is_leaf[bvh_id, node_id]:
                tri_start = bvhs.left_or_start[bvh_id, node_id]
                tri_count = bvhs.right_or_count[bvh_id, node_id]
                bvh_depth = bvhs.depth[bvh_id, node_id]

                for i in range(tri_count):
                    tri_id = bvh_infos[bvh_id].triangle_offset + tri_start + i
                    tri = triangles[tri_id]
                    ray_triangle_intersection(ray_local, tri, inter_local, tri_id, t_min, inter_local.t, bvh_depth)

            else:
             
                # Carefully push children
                right_id = bvhs.right_or_count[bvh_id, node_id]
                left_id = bvhs.left_or_start[bvh_id, node_id]

                aabb_min_right = bvhs.aabb_min[bvh_id, right_id]
                aabb_max_right = bvhs.aabb_max[bvh_id, right_id]
                hit_aabb_right, t_aabb_right = ray_aabb_intersection_test(ray_local, aabb_min_right, aabb_max_right, t_min, inter_local.t)

                aabb_min_left = bvhs.aabb_min[bvh_id, left_id]
                aabb_max_left = bvhs.aabb_max[bvh_id, left_id]
                hit_aabb_left, t_aabb_left = ray_aabb_intersection_test(ray_local, aabb_min_left, aabb_max_left, t_min, inter_local.t)

                inter.box_test_count += 2

                should_push_left = hit_aabb_left and inter_local.t > t_aabb_left
                should_push_right = hit_aabb_right and inter_local.t > t_aabb_right
                if should_push_left and should_push_right:
                    right_closer = t_aabb_right < t_aabb_left
                    first = ti.select(right_closer, right_id, left_id)
                    second = ti.select(right_closer, left_id, right_id)
                    stack_ptr += 1
                    stack[stack_ptr] = second
                    stack_ptr += 1
                    stack[stack_ptr] = first
                elif should_push_right:
                    stack_ptr += 1
                    stack[stack_ptr] = right_id
                elif should_push_left:
                    stack_ptr += 1
                    stack[stack_ptr] = left_id
        
        if inter_local.hit: # If an intersection has occured inside this bvh
            point_world = transform_point(inter_local.point, bvh_transform)
            t_world = (point_world - ray.origin).norm()
            if inter.t < t_world:
                continue

            bvh_inv_transpose_transform = bvh_inv_transform.transpose()
            inter.point = point_world
            inter.normal = transform_direction(inter_local.normal, bvh_inv_transpose_transform)
            inter.normal_geom = transform_direction(inter_local.normal_geom, bvh_inv_transpose_transform)
            inter.t = t_world

            inter.front_face = 1 if ray.direction.dot(inter.normal_geom) < 0 else 0
            inter.material_id = inter_local.material_id
            inter.hit = 1
            inter.primitive_id = inter_local.primitive_id
            inter.primitive_type = PRIMITIVE_TRIANGLE
            inter.bvh_depth = inter_local.bvh_depth

@ti.func
def ray_scene_intersection(ray: Ray, scene: ti.template(), inter:ti.template(), t_min: ti.f32, t_max: ti.f32):  # type: ignore
    for i in range(scene.num_spheres[None]):
        ray_sphere_intersection(ray, scene.spheres[i], inter, i, t_min, t_max)

    for i in range(scene.num_planes[None]):
        ray_plane_intersection(ray, scene.planes[i], inter, i, t_min, t_max)

    for i in range(scene.num_free_triangles[None]):
        primitive_id = scene.free_triangles[i]
        ray_triangle_intersection(ray, scene.triangles[primitive_id], inter, primitive_id, t_min, t_max)

    ray_bvhs_intersection(ray, scene.bvhs, scene.bvh_infos, scene.num_bvhs[None], scene.triangles, inter, t_min, t_max)

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


