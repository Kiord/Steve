import click
import taichi as ti
import dearpygui.dearpygui as dpg
import numpy as np

from scene import Scene
from camera import Camera
from path_tracing import render
from denoising import bilateral_filter
from tone_mapping import tone_map
from control import FreeFlyCameraController
from timer import FrameTimer
from app_state import AppState
from ui import build_ui
from sample_scenes import setup_veach_scene, setup_suzanne_scene, setup_dragon_scene

@click.command()
@click.option('--profiling', '-p', type=click.BOOL, default=False, help='Ënable profiling')
@click.option('--denoising', '-d', type=click.BOOL, default=False, help='Ënable denoising')
@click.option('--tone-mapping', '-tm', type=click.BOOL, default=True, help='Enable tone mapping')
@click.option('--size', '-s', type=(int, int), default=(800, 600), help='Viewport size as WIDTH HEIGHT')
@click.option('--spp', type=int, default=1, help='Samples per pixel')
@click.option('--max_depth', '-md', type=int, default=5, help='Max path depth')
@click.option('--arch', '-a', type=str, default='cpu', help='Device to run on')
def cli(profiling, denoising, tone_mapping, size, spp, max_depth, arch):
    arch = arch.lower()
    if hasattr(ti, arch):
        arch = getattr(ti, arch)
    else:
        print(f'[Error] unavailable backend "{arch}"')

    ti.init(arch=arch, default_fp=ti.f32, kernel_profiler=profiling)
    #import time
    #time.sleep(0.5)

    state = AppState(width=size[0], height=size[1])
    state.profiling = profiling
    state.denoising = denoising
    state.tone_mapping = tone_mapping
    state.spp = spp
    state.max_depth = max_depth
   

    scene = Scene()
    #setup_scene(scene)

    

    camera = Camera.field(shape=())
    camera_controller = FreeFlyCameraController(state)
    camera_controller.update_camera_field(camera)


    #setup_veach_scene(scene, camera_controller)
    #setup_suzanne_scene(scene, camera_controller)
    setup_dragon_scene(scene, camera_controller)
    #setup_scene(scene)
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
               max_depth=1 if state.mode_id != 0 else state.max_depth,
               buffers=state.buffers,
               width=state.width, height=state.height,
               frame_id=state.frame_id)

        if state.mode_id == 0:
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
        elif state.mode_id == 1:
            final_buffer = state.buffers.albedo
        elif state.mode_id == 2:
            final_buffer = state.buffers.normal.to_numpy() * 0.5 + 0.5
        elif state.mode_id == 3:
            final_buffer = np.clip(np.log(0.2*np.clip(state.buffers.bvh_depth.to_numpy(), min=1)), min=0)
        elif state.mode_id == 4:
            final_buffer = np.clip(np.log(state.buffers.depth.to_numpy()), min=0)
        elif state.mode_id == 5:
            final_buffer = np.clip(np.log(0.05*np.clip(state.buffers.box_test_count.to_numpy(), min=1)), min=0)
      
        canvas.set_image(final_buffer)

        window.show()

        if state.profiling:
            ti.sync()
            ti.profiler.print_kernel_profiler_info()

        state.frame_id += 1
        state.scene_changed=False

if __name__ == "__main__":
    cli()
