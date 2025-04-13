import click
import taichi as ti
import threading
import dearpygui.dearpygui as dpg

from scene import Scene, PointLight, Sphere, Material, Plane
from camera import Camera
from path_tracing import render
from denoising import bilateral_filter
from tone_mapping import tone_map
from control import FreeFlyCameraController
from timer import FrameTimer
from app_state import AppState
from ui import build_ui

@click.command()
@click.option('--profiling', '-p', is_flag=True, default=False, help='Enable Taichi kernel profiling')
@click.option('--denoising', '-d', is_flag=True, default=False, help='Enable denoising')
@click.option('--no-tone-mapping', '-ntm', is_flag=True, default=False, help='Disable tone mapping')
def cli(profiling, denoising, no_tone_mapping):
    ti.init(arch=ti.gpu, default_fp=ti.f32, kernel_profiler=profiling)

    def setup_scene(scene: Scene):
        scene.num_spheres[None] = 16
        for i in range(4):
            for j in range(4):
                scene.spheres[i * 4 + j] = Sphere(center=ti.Vector([i, 0.7, j]), radius=0.7, material_id=0)

        scene.num_planes[None] = 1
        scene.planes[0] = Plane(point=ti.Vector([0, 0, 0]), normal=ti.Vector([0, 1, 0]), material_id=0)

        scene.materials[0] = Material(diffuse=ti.Vector([0.7, 0.7, 0.7]),
                                      specular=ti.Vector([0.0, 0.0, 0.0]),
                                      shininess=0.0)

        scene.light[None] = PointLight(position=ti.Vector([2, 2, 2]),
                                       color=ti.Vector([3.0, 3.0, 5.0]))

    # Set up application state
    state = AppState()
    state.profiling = profiling
    state.denoising = denoising
    state.tone_mapping = not no_tone_mapping

    scene = Scene()
    setup_scene(scene)

    camera = Camera.field(shape=())
    camera_controller = FreeFlyCameraController()
    camera_controller.update_camera_field(camera, state.aspect_ratio)

    build_ui(state)
    threading.Thread(target=dpg.start_dearpygui, daemon=True).start()

    window = ti.ui.Window("Steve", res=(state.width, state.height))
    canvas = window.get_canvas()
    timer = FrameTimer()

    while window.running:
        
        dt = timer.get_dt()

        camera_controller.update_from_input(window, dt)
        camera_controller.update_camera_field(camera, state.aspect_ratio)

        if state.profiling:
            ti.profiler.clear_kernel_profiler_info()

        render(scene, camera, spp=state.spp,
               max_depth=state.max_depth,
               buffers=state.buffers,
               width=state.width, height=state.height,
               frame_idx=state.frame_count)

        final_buffer = state.buffers.color

        if state.denoising:
            bilateral_filter(state.buffers,
                             sigma_color=state.sigma_color,
                             sigma_normal=state.sigma_normal,
                             sigma_spatial=state.sigma_spatial,
                             radius=state.radius)
            final_buffer = state.buffers.denoised
    
        if state.tone_mapping:
            tone_map(final_buffer)
       
        canvas.set_image(final_buffer)

        window.show()

        if state.profiling:
            ti.sync()
            ti.profiler.print_kernel_profiler_info()

        state.frame_count += 1


if __name__ == "__main__":
    cli()
