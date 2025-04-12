import taichi as ti
from scene import Scene, Sphere, Plane
from datatypes import vec3f

@ti.dataclass
class Ray:
    origin: vec3f # type: ignore
    direction: vec3f # type: ignore

@ti.dataclass
class HitRecord:
    point: vec3f  # type: ignore
    normal: vec3f  # type: ignore
    t: ti.f32 # type: ignore
    material_id: ti.i32 # type: ignore
    front_face: ti.i32 # type: ignore


@ti.func
def hit_sphere(ray_o, ray_d, sphere: Sphere, t_min, t_max):
    oc = ray_o - sphere.center
    a = ray_d.dot(ray_d)
    b = 2.0 * oc.dot(ray_d)
    c = oc.dot(oc) - sphere.radius * sphere.radius
    discriminant = b * b - 4 * a * c

    hit = False
    rec = HitRecord(
        point=ti.Vector([0.0, 0.0, 0.0]),
        normal=ti.Vector([0.0, 0.0, 0.0]),
        t=t_max,
        material_id=sphere.material_id,
        front_face=1
    )

    if discriminant > 0:
        sqrt_d = ti.sqrt(discriminant)
        root = (-b - sqrt_d) / (2.0 * a)
        if t_min < root < t_max:
            rec.t = root
            rec.point = ray_o + root * ray_d
            outward = (rec.point - sphere.center).normalized()
            rec.normal = outward
            rec.front_face = 1 if ray_d.dot(outward) < 0 else 0
            if rec.front_face == 0:
                rec.normal = -rec.normal
            hit = True

    return hit, rec

@ti.func
def hit_plane(ray_o, ray_d, plane: Plane, t_min, t_max):
    hit = False
    rec = HitRecord(
        point=ti.Vector([0.0, 0.0, 0.0]),
        normal=ti.Vector([0.0, 0.0, 0.0]),
        t=t_max,
        front_face=1,
        material_id=-1
    )

    denom = ray_d.dot(plane.normal)
    if abs(denom) > 1e-6:
        t = (plane.point - ray_o).dot(plane.normal) / denom
        if t_min < t < t_max:
            p = ray_o + t * ray_d
            front_face = 1 if denom < 0 else 0
            n = plane.normal if front_face else -plane.normal
            rec = HitRecord(
                point=p,
                normal=n,
                t=t,
                front_face=front_face,
                material_id=plane.material_id
            )
            hit = True

    return hit, rec


@ti.func
def hit_scene(scene: ti.template(), ray_o, ray_d, t_min, t_max): # type: ignore
    closest = t_max
    hit_anything = False
    final_rec = HitRecord(
        point=ti.Vector([0.0, 0.0, 0.0]),
        normal=ti.Vector([0.0, 0.0, 0.0]),
        t=closest,
        material_id=-1,
        front_face=1
    )

    for i in range(scene.num_spheres[None]):
        hit, temp_rec = hit_sphere(ray_o, ray_d, scene.spheres[i], t_min, closest)
        if hit:
            closest = temp_rec.t
            final_rec = temp_rec
            hit_anything = True

    for i in range(scene.num_planes[None]):
        hit, temp_rec = hit_plane(ray_o, ray_d, scene.planes[i], t_min, closest)
        if hit:
            closest = temp_rec.t
            final_rec = temp_rec
            hit_anything = True

    return hit_anything, final_rec