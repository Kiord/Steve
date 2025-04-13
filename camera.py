import taichi as ti
import math
import numpy as np
from ray import Ray
from datatypes import vec3f

@ti.dataclass
class Camera:
    origin: vec3f # type: ignore
    lower_left_corner: vec3f # type: ignore
    horizontal: vec3f # type: ignore
    vertical: vec3f # type: ignore


@ti.func
def make_ray(cam: Camera, u: ti.f32, v: ti.f32) -> Ray: #type:ignore
    ray_o = cam.origin
    ray_d = (cam.lower_left_corner + u * cam.horizontal + v * cam.vertical - cam.origin).normalized()
    return Ray(ray_o, ray_d)

def compute_camera_vectors(
        origin : np.ndarray,
        lookat : np.ndarray,
        vup : np.ndarray,
        vfov : float,
        aspect_ratio : float):

    theta = math.radians(vfov)
    h = math.tan(theta / 2)
    viewport_height = 2.0 * h
    viewport_width = aspect_ratio * viewport_height

    w = origin - lookat
    w /= np.linalg.norm(w)
    u = np.cross(vup, w)
    u /= np.linalg.norm(u)
    v = np.cross(w, u)

    horizontal = viewport_width * u
    vertical = viewport_height * v
    lower_left_corner = origin - horizontal * 0.5 - vertical * 0.5 - w

    return lower_left_corner, horizontal, vertical


def create_camera(origin : np.ndarray,
                  lookat : np.ndarray,
                  vup : np.ndarray,
                  vfov : float,
                  aspect_ratio : float):
    lower_left_corner, horizontal, vertical = compute_camera_vectors(
        origin, lookat, vup, vfov, aspect_ratio)


    cam = Camera.field(shape=())
    cam[None] = Camera(origin=ti.Vector(list(origin)),
                       lower_left_corner=ti.Vector(list(lower_left_corner)),
                       horizontal=ti.Vector(list(horizontal)),
                       vertical=ti.Vector(list(vertical)))
    return cam