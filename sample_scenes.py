import taichi as ti
from scene import Scene, Material
from camera import Camera
from control import FreeFlyCameraController as FFCC
import numpy as np
import math


def setup_veach_scene(scene:Scene, ffcc:FFCC):
    # Camera
    scene_scale = 1

    ffcc.pos = np.array([0.0,-10.0, -40.0]) * scene_scale
    ffcc.yaw = 90
    ffcc.pitch = -math.degrees(0.1)
    ffcc.move_speed = 15 * scene_scale
    ffcc.fov = 40
    
    
    diffuse_white = scene.add_material([1,1,1],[0,0,0], 0)
    # x -> z
    # y -> x
    # z -> y
    scene.add_quad(scene_scale * 50.0 * np.array([0,-0.5,0]),  [scene_scale * 50.0] * 2, [1,0,0], 0, diffuse_white)
    scene.add_quad(scene_scale * 50.0 * np.array([0, 0, 0.5]), [scene_scale * 50.0] * 2, [1,0,0], 0.5 * math.pi, diffuse_white)

    # // Spheres
    sphere_colors = [(1.0,0.1,0.1),(0.2,1.0,0.4),(1,0.15,1.0),(0.1,0.3, 1.0)]

    color_multipliers =  (10.0, 4.0, 3.0, 2.8)
    sphere_sizes = (0.1, 0.4, 1.0, 2.0)
    shift = np.array([-7.0, 0, 0])
    base = np.array([10.0 , -5.0, 18.0])
    for i in range(4):
        material_id = scene.add_material([0,0,0], np.array(sphere_colors[i]) * color_multipliers[i] / sphere_sizes[i], 0)
        scene.add_sphere(scene_scale * (base + shift * i), scene_scale * sphere_sizes[i], material_id)

    # Panels
    shininesses = (1.0, 10.0, 100.0, 1000.0)
    multiplier = 250.0

    angles = (math.radians(13.0), math.radians(22.0), math.radians(32.0), math.radians(53.0))
    shifts = np.array([[0,0,0], [0, 3.0,8.0], [0, 6.0, 14.0], [0, 11.0, 20.0]])
    offset = np.array([0,-22.0,0]) * scene_scale
    scale = np.array([5.0, 25.0]) * scene_scale
    for i in range(4):
        material_id = scene.add_material([1,1,1], [0,0,0], multiplier * shininesses[i])
        scene.add_quad(offset + shifts[i], scale, [1,0,0], -angles[i], material_id)

    # scene.ground_color[None] =  ti.Vector([0,0,0])
    # scene.horizon_color[None] = ti.Vector([0,0,0])
    # scene.sky_color[None] = ti.Vector([0,0,0])
    # scene.sun_color[None] = ti.Vector([0,0,0])

    # mat2 = scene.add_material([0.0,0.0,.0], [20.0,0,0], 0) 

    # scene.add_sphere([-11.,  -7.,  18.], 2.0, mat2)


    #scene.add_plane([0,0,0], [0,1,0], mat1)