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
@click.option('--profiling', '-p', type=click.BOOL, default=False, help='Ënable profiling')
@click.option('--denoising', '-d', type=click.BOOL, default=False, help='Ënable denoising')
@click.option('--tone-mapping', '-tm', type=click.BOOL, default=True, help='Enable tone mapping')
@click.option('--size', '-s', type=(int, int), default=(800, 600), help='Viewport size as WIDTH HEIGHT')
@click.option('--spp', type=int, default=1, help='Samples per pixel')
@click.option('--max_depth', '-md', type=int, default=5, help='Max path depth')
@click.option('--device', type=click.Choice(['cpu', 'gpu'], case_sensitive=False), default='cpu', help='Device to run on')
def cli(profiling, denoising, tone_mapping, size, spp, max_depth, device):
    
    arch = {'cpu':ti.cpu, 'gpu':ti.gpu}[device.lower()]

    ti.init(arch=arch, default_fp=ti.f32, kernel_profiler=profiling)
    import time
    time.sleep(0.5)

    state = AppState(width=size[0], height=size[1])
    state.profiling = profiling
    state.denoising = denoising
    state.tone_mapping = tone_mapping
    state.spp = spp
    state.max_depth = max_depth
    

    def setup_scene(scene: Scene):
        scene.materials[0] = Material(albedo=ti.Vector([0.7, 0.7, 0.7]),
                                      shininess=0.0,
                                      emissive=ti.Vector([0.0, 0.0, 0.0]))
        scene.materials[1] = Material(albedo=ti.Vector([1.0, 1.0, 1.0]),
                                      shininess=0.0,
                                      emissive=ti.Vector([20.0, 0.0, 0.0]))
        scene.materials[2] = Material(albedo=ti.Vector([1.0, 1.0, 1.0]),
                                shininess=50.0,
                                emissive=ti.Vector([0.0, 0.0, 0.0]))


        for i in range(4):
            for j in range(4):
                scene.add_sphere([i, 0.5 + 0.0*float(i!=1 and j!=1), j], 0.5, int(i==1 and j==1))

        scene.spheres[1].center -= ti.Vector([10,0,0])

        scene.add_sphere([-2,1,-2], 1, 2)

        scene.add_plane([0,0,0], [0,1,0], 0)

        scene.add_triangle([0,0,0], [5,5,5], [0,0,5], 2)

    scene = Scene()
    setup_scene(scene)

    camera = Camera.field(shape=())
    camera_controller = FreeFlyCameraController(state)
    camera_controller.update_camera_field(camera)

    #build_ui(state)
    #threading.Thread(target=dpg.start_dearpygui, daemon=True).start()

    window = ti.ui.Window("Steve", res=(state.width, state.height))
    canvas = window.get_canvas()
    timer = FrameTimer()

    while window.running:
        dt = timer.get_dt()

        camera_controller.update_from_input(window, dt)
        camera_controller.update_camera_field(camera)

        if state.profiling:
            ti.profiler.clear_kernel_profiler_info()

        render(scene, camera, spp=state.spp,
               max_depth=state.max_depth,
               buffers=state.buffers,
               width=state.width, height=state.height,
               frame_id=state.frame_id)

        final_buffer = state.buffers.final_buffer

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

        state.frame_id += 1
        state.scene_changed=False

if __name__ == "__main__":
    cli()
