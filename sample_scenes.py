import taichi as ti
from scene import Scene, Material
from camera import Camera
from control import FreeFlyCameraController as FFCC
import numpy as np
import math
import trimesh as tm
import pyvista as pv
from utils import load_mesh
from bvh import print_bvh_summary

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

def setup_suzanne_scene(scene:Scene, ffcc:FFCC):
    # Camera

    # ffcc.pos = np.array([0.0,-10.0, -40.0]) * scene_scale
    # ffcc.yaw = 90
    # ffcc.pitch = -math.degrees(0.1)
    
    # ffcc.fov = 40
    mesh, bvh_dict = load_mesh('suzanne.stl')
    print_bvh_summary(bvh_dict)


    ffcc.move_speed = mesh.scale
    # print(mesh)
    # plotter = pv.Plotter()
    # plotter.add_mesh(mesh)
    # plotter.add_arrows(mesh.vertices, mesh.vertex_normals, mag=0.1)
    # plotter.show()
    diffuse_white = scene.add_material([1,1,1],[0,0,0], 0)

    #scene.add_mesh(mesh, diffuse_white)
    scene.add_mesh_bvh(mesh, bvh_dict, diffuse_white)
    t = np.array([
        [1, 0, 0, 2],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ], dtype=np.float32)
    scene.add_mesh_bvh(mesh, bvh_dict, diffuse_white, t)
    t = np.array([
        [0, 1, 0, 1],
        [1, 0, 0, 0],
        [0, 0, 1, 2],
        [0, 0, 0, 1],
    ], dtype=np.float32)
    scene.add_mesh_bvh(mesh, bvh_dict, diffuse_white, t)

    t = np.array([
        [2, 0, 0, 0],
        [0, 0.5, 0, 0],
        [0, 0, 1, 4],
        [0, 0, 0, 1],
    ], dtype=np.float32)
    scene.add_mesh_bvh(mesh, bvh_dict, diffuse_white, t)

def setup_dragon_scene(scene:Scene, ffcc:FFCC):
    # Camera

    # ffcc.pos = np.array([0.0,-10.0, -40.0]) * scene_scale
    # ffcc.yaw = 90
    # ffcc.pitch = -math.degrees(0.1)
    
    # ffcc.fov = 40
    mesh, bvh_dict = load_mesh('bunny.stl', bvh_type='sweep', recompute_bvh=True)
    print_bvh_summary(bvh_dict)
    s = 1.0 / mesh.scale

    centroid = mesh.centroid * s
    ffcc.set_look_at(centroid + [s, s, s], centroid)
    #ffcc.move_speed = mesh.scale
    # print(mesh)
    # plotter = pv.Plotter()
    # plotter.add_mesh(mesh)
    # plotter.add_arrows(mesh.vertices, mesh.vertex_normals, mag=0.1)
    # plotter.show()
    diffuse_white = scene.add_material([1,1,1],[0,0,0], 0)

    
    t = np.array([
        [0, s, 0, 0],
        [0, 0, s, 0],
        [s, 0, 0, 0],
        [0, 0, 0, 1],
    ], dtype=np.float32)
    scene.add_mesh_bvh(mesh, bvh_dict, diffuse_white, t)