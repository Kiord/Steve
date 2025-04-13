import taichi as ti
from scene import Scene, Sphere, Plane, Material
from datatypes import vec3f

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
    material: Material 
    ray: Ray         

@ti.func
def empty_intersection() -> Intersection:
    return Intersection(
        hit=0,
        point=vec3f(0.0, 0.0, 0.0),
        normal=vec3f(0.0, 0.0, 0.0),
        t=1e6,
        front_face=1,
        material_id=-1,
        material=Material(  # use defaults from your Material dataclass
            diffuse=vec3f(0.0, 0.0, 0.0),
            specular=vec3f(0.0, 0.0, 0.0),
            shininess=0.0,
            emissive=vec3f(0.0, 0.0, 0.0)
        ),
        ray=Ray(origin=vec3f(0.0, 0.0, 0.0),
                direction=vec3f(0.0, 0.0, 0.0))
    )               


@ti.func
def hit_sphere(ray: Ray, sphere: Sphere, t_min, t_max):
    oc = ray.origin - sphere.center
    a = ray.direction.dot(ray.direction)
    b = 2.0 * oc.dot(ray.direction)
    c = oc.dot(oc) - sphere.radius * sphere.radius
    discriminant = b * b - 4 * a * c

    inter = empty_intersection()

    if discriminant > 0:
        sqrt_d = ti.sqrt(discriminant)
        root = (-b - sqrt_d) / (2.0 * a)
        if t_min < root < t_max:
            inter.t = root
            inter.point = ray.origin + root * ray.direction
            outward = (inter.point - sphere.center).normalized()
            inter.normal = outward
            inter.front_face = 1 if ray.direction.dot(outward) < 0 else 0
            inter.material_id = sphere.material_id
            inter.ray = ray
            if inter.front_face == 0:
                inter.normal = -inter.normal
            inter.hit = 1

    return inter

@ti.func
def hit_plane(ray : Ray, plane: Plane, t_min, t_max):
    inter = empty_intersection()

    denom = ray.direction.dot(plane.normal)
    if abs(denom) > 1e-6:
        t = (plane.point - ray.origin).dot(plane.normal) / denom
        if t_min < t < t_max:
            inter.t = t
            inter.point = ray.origin + t * ray.direction
            inter.front_face = 1 if denom < 0 else 0
            inter.normal = plane.normal if inter.front_face else -plane.normal
            inter.hit=1
            inter.ray=ray
            inter.material_id = plane.material_id

    return inter


@ti.func
def hit_scene(ray:Ray, scene: ti.template(), t_min, t_max): # type: ignore
    closest = t_max

    final_inter = empty_intersection() 

    for i in range(scene.num_spheres[None]):
        inter = hit_sphere(ray, scene.spheres[i], t_min, closest)
        if inter.hit:
            closest = inter.t
            final_inter = inter

    for i in range(scene.num_planes[None]):
        inter = hit_plane(ray, scene.planes[i], t_min, closest)
        if inter.hit:
            closest = inter.t
            final_inter = inter
    
    if final_inter.hit:
        final_inter.material = scene.materials[final_inter.material_id]

    return final_inter