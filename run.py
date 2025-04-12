import click

@click.command()
@click.option('--profiling', '-p', is_flag=True, default=False, help='Enable Taichi kernel profiling')
@click.option('--no-denosing', '-nd', is_flag=True, default=False, help='Disable denoising')
@click.option('--no-tone-mapping', '-ntm', is_flag=True, default=False, help='Disable tone mapping')
def cli(profiling, no_denosing, no_tone_mapping):
    import taichi as ti
    ti.init(arch=ti.gpu, default_fp=ti.f32, kernel_profiler=profiling)

    from scene import Scene, PointLight, Sphere, Material, Plane
    from camera import Camera
    from path_tracing import render, RenderBuffers
    from denoising import bilateral_filter
    from tone_mapping import tone_map
    from control import FreeFlyCameraController
    from timer import FrameTimer
    

    def setup_scene(scene: Scene):
        scene.num_spheres[None] = 1
        scene.spheres[0] = Sphere(center=ti.Vector([0, 0.5, 0]), radius=0.5, material_id=0)

        scene.num_planes[None] = 1
        scene.planes[0] = Plane(point=ti.Vector([0, 0, 0]), normal=ti.Vector([0, 1, 0]), material_id=0)

        scene.materials[0] = Material(diffuse=ti.Vector([0.7, 0.7, 0.7]), specular=ti.Vector([0.0, 0.0, 0.0]), shininess=0.0)
        scene.light[None] = PointLight(position=ti.Vector([1, 1, 0]), color=ti.Vector([1.0, 1.0, 1.0]))

    width, height = 1200, 800
    aspect_ratio = width / height

    buffers = RenderBuffers(width, height)
    scene = Scene()

    camera = Camera.field(shape=())
    camera_controller = FreeFlyCameraController()
    camera_controller.update_camera_field(camera, aspect_ratio)

    setup_scene(scene)

    window = ti.ui.Window("Steve", res=(width, height))
    canvas = window.get_canvas()
    frame_timer = FrameTimer()

    frame_count = 0
    while window.running:
        dt = frame_timer.get_dt()
        camera_controller.update_from_input(window, dt)
        camera_controller.update_camera_field(camera, aspect_ratio)

        if profiling:
            ti.profiler.clear_kernel_profiler_info()

        render(scene, camera, spp=1, buffers=buffers, width=width, height=height)

        if not no_denosing:
            bilateral_filter(buffers, sigma_color=0.2, sigma_normal=0.2, sigma_spatial=1.5, radius=1)
            if not no_tone_mapping:
                tone_map(buffers.denoised)
            canvas.set_image(buffers.denoised)
        else:
            if not no_tone_mapping:
                tone_map(buffers.color)
            canvas.set_image(buffers.color)

        window.show()

        if profiling:
            ti.sync()
            ti.profiler.print_kernel_profiler_info()

        frame_count+= 1

if __name__ == "__main__":
    cli()