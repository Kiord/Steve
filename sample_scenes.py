import taichi as ti
from scene import Scene, create_lambert, create_ggx, create_phong
from camera import Camera
from control import FreeFlyCameraController as FFCC
import numpy as np
import math
import trimesh as tm
import pyvista as pv
from utils import load_mesh
from bvh import print_bvh_summary

def setup_spheres_scene(scene:Scene, ffcc:FFCC):
    
    lambert_white = scene.add_material(create_lambert([1,1,1]))
    emissive_white = scene.add_material(create_lambert([0,0,0], [2,2,2]))

    shininesses = [1, 10, 50, 150, 500, 2000]
    roughnesses = [0.01, 0.05, 0.15, 0.3, 0.5, 0.8]

    scene.add_sphere((0,0.5,0), 0.5, lambert_white)

    for i in range(6):
        scene.add_sphere((1,0.5,i-6//2), 0.5, scene.add_material(create_phong([1,1,1], shininesses[i])))
        scene.add_sphere((-1,0.5,i-6//2), 0.5, scene.add_material(create_ggx([1,1,1], roughnesses[i])))

    ffcc.set_look_at((3,3,3), (0,0.5,0))

    #scene.add_quad((0,5,0), (3,3,3), (1,0,0), 0, emissive_white)
    scene.add_sphere((0,5,0),1,emissive_white)

    scene.add_plane([0,0,0], [0,1,0], lambert_white)

    scene.ground_color[None] =  ti.Vector([0,0,0])
    scene.horizon_color[None] = ti.Vector([0,0,0])
    scene.sky_color[None] = ti.Vector([0,0,0])
    scene.sun_color[None] = ti.Vector([0,0,0])

def setup_veach_scene(scene:Scene, ffcc:FFCC):
    # Camera
    scene_scale = 1

    ffcc.pos = np.array([0.0,-10.0, -40.0]) * scene_scale
    ffcc.yaw = 90
    ffcc.pitch = -math.degrees(0.1)
    ffcc.move_speed = 15 * scene_scale
    ffcc.fov = 40
    
    
    
    diffuse_white = scene.add_material(create_lambert([1,1,1],[0,0,0]))
    emissive_white = scene.add_material(create_lambert([0,0,0],[1,1,1]))
    # x -> z
    # y -> x
    # z -> y
    scene.add_quad(scene_scale * 50.0 * np.array([0,-0.5,0]),  [scene_scale * 50.0] * 2, [1,0,0], 0, diffuse_white)
    scene.add_quad(scene_scale * 50.0 * np.array([0, 0, 0.5]), [scene_scale * 50.0] * 2, [1,0,0], 0.5 * math.pi, emissive_white)

    # // Spheres
    sphere_colors = [(1.0,0.1,0.1),(0.2,1.0,0.4),(1,0.15,1.0),(0.1,0.3, 1.0)]

    color_multipliers =  (10.0, 4.0, 3.0, 2.8)
    sphere_sizes = (0.1, 0.4, 1.0, 2.0)
    shift = np.array([-7.0, 0, 0])
    base = np.array([10.0 , -5.0, 18.0])
    for i in range(4):
        material = create_lambert([0,0,0], np.array(sphere_colors[i]) * color_multipliers[i] / sphere_sizes[i])
        material_id = scene.add_material(material)
        scene.add_sphere(scene_scale * (base + shift * i), scene_scale * sphere_sizes[i], material_id)

    # Panels
    shininesses = (1.0, 10.0, 100.0, 1000.0)
    multiplier = 250.0

    angles = (math.radians(13.0), math.radians(22.0), math.radians(32.0), math.radians(53.0))
    shifts = np.array([[0,0,0], [0, 3.0,8.0], [0, 6.0, 14.0], [0, 11.0, 20.0]])
    offset = np.array([0,-22.0,0]) * scene_scale
    scale = np.array([5.0, 25.0]) * scene_scale
    for i in range(4):
        material = create_phong([1,1,1], multiplier * shininesses[i], [0,0,0])
        material_id = scene.add_material(material)
        scene.add_quad(offset + shifts[i], scale, [1,0,0], -angles[i], material_id)

    scene.ground_color[None] =  ti.Vector([0,0,0])
    scene.horizon_color[None] = ti.Vector([0,0,0])
    scene.sky_color[None] = ti.Vector([0,0,0])
    scene.sun_color[None] = ti.Vector([0,0,0])

    # mat2 = scene.add_material([0.0,0.0,.0], [20.0,0,0], 0) 

    # scene.add_sphere([-11.,  -7.,  18.], 2.0, mat2)


    #scene.add_plane([0,0,0], [0,1,0], mat1)

def setup_cornell_scene(scene: Scene, ffcc: FFCC):
    room_size = 5.0
    half = room_size * 0.5
    z_center = 5.0

    # CAMERA — pull back further and aim into box
    ffcc.pos = np.array([0.0, half, z_center + 7.0])
    ffcc.yaw = -90.0  # looking into -Z
    ffcc.pitch = 0.0
    ffcc.fov = 40.0
    ffcc.move_speed = 2.0

    # MATERIALS
    white = scene.add_material(create_lambert([0.75, 0.75, 0.75]))
    white_ggx = scene.add_material(create_ggx([0.75, 0.75, 0.75], 0.5))
    red   = scene.add_material(create_lambert([0.75, 0.1, 0.1]))
    green = scene.add_material(create_lambert([0.1, 0.75, 0.1]))
    light = scene.add_material(create_lambert([0, 0, 0], [15, 15, 15]))

    def Z(p): return [p[0], p[1], p[2] + z_center]

    # ROOM: floor, ceiling, walls, back
    scene.add_quad(Z([0, 0, 0]), [room_size, room_size], [0, 0, 1], 0, white)                         # Floor
    scene.add_quad(Z([0, room_size, 0]), [room_size, room_size], [0, 0, 1], math.pi, white)           # Ceiling (flipped)
    scene.add_quad(Z([0, half, -half]), [room_size, room_size], [1, 0, 0], -0.5 * math.pi, white)     # Back wall
    scene.add_quad(Z([-half, half, 0]), [room_size, room_size], [0, 0, 1], 0.5 * math.pi, green)      # Left wall
    scene.add_quad(Z([half, half, 0]), [room_size, room_size], [0, 0, 1], -0.5 * math.pi, red)        # Right wall

    # LIGHT — properly centered on ceiling
    light_size = 1.0
    scene.add_sphere(Z([0, room_size - 0.1, 0]), 1.0, light)
    #scene.add_quad(Z([0, room_size - 0.1, 0]), [1.0, 1.0], [0, -1, 0], 0, light)

    # BLOCKS — full 5 sides, fixed face quads
    def add_block_triangles(center, size, material_id, rotation_degrees=0.0):
        cx, cy, cz = Z(center)
        sx, sy, sz = size
        hx, hy, hz = sx / 2, sy / 2, sz / 2

        # Rotation around Y-axis
        theta = math.radians(rotation_degrees)
        cos_theta = math.cos(theta)
        sin_theta = math.sin(theta)

        def rotate_y(p):
            dx = p[0] - cx
            dz = p[2] - cz
            x = cos_theta * dx - sin_theta * dz + cx
            z = sin_theta * dx + cos_theta * dz + cz
            return [x, p[1], z]

        # 8 corners
        corners = {
            'blf': [cx - hx, cy - hy, cz - hz],
            'brf': [cx + hx, cy - hy, cz - hz],
            'tlf': [cx - hx, cy + hy, cz - hz],
            'trf': [cx + hx, cy + hy, cz - hz],
            'blb': [cx - hx, cy - hy, cz + hz],
            'brb': [cx + hx, cy - hy, cz + hz],
            'tlb': [cx - hx, cy + hy, cz + hz],
            'trb': [cx + hx, cy + hy, cz + hz],
        }

        p = {k: rotate_y(v) for k, v in corners.items()}

        # Build faces
        scene.add_triangle(p['tlf'], p['trf'], p['brf'], material_id)
        scene.add_triangle(p['brf'], p['blf'], p['tlf'], material_id)

        scene.add_triangle(p['trb'], p['tlb'], p['blb'], material_id)
        scene.add_triangle(p['blb'], p['brb'], p['trb'], material_id)

        scene.add_triangle(p['tlb'], p['tlf'], p['blf'], material_id)
        scene.add_triangle(p['blf'], p['blb'], p['tlb'], material_id)

        scene.add_triangle(p['trf'], p['trb'], p['brb'], material_id)
        scene.add_triangle(p['brb'], p['brf'], p['trf'], material_id)

        scene.add_triangle(p['tlb'], p['trb'], p['trf'], material_id)
        scene.add_triangle(p['trf'], p['tlf'], p['tlb'], material_id)

    add_block_triangles(center=[-1.0, 1.0, -0.9], size=[1.0, 2.0, 1.0], material_id=white, rotation_degrees=15)

    # Short block: rotate ~-10° CW
    add_block_triangles(center=[1.0, 0.6, 1.0], size=[1.2, 1.2, 1.2], material_id=white_ggx, rotation_degrees=-10)

    # DISABLE ENVIRONMENT LIGHT
    scene.ground_color[None] = ti.Vector([0, 0, 0])
    scene.horizon_color[None] = ti.Vector([0, 0, 0])
    scene.sky_color[None] = ti.Vector([0, 0, 0])
    scene.sun_color[None] = ti.Vector([0, 0, 0])



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
    mesh, bvh_dict = load_mesh('sponza.stl', bvh_type='sweep')
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