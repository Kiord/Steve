from path_tracing import RenderBuffers

class AppState:
    modes = ['render', 'albedo', 'normal', 'bvh_depth', 'depth', 'box_test_count']

    def __init__(self, width=1200, height=800):
        self.width = width
        self.height = height
        self.aspect_ratio = width / height
        self.profiling = False
        self.denoising = False
        self.tone_mapping = True
        self.spp = 1
        self.max_depth = 3
        self.radius = 2
        self.sigma_color = 0.2
        self.sigma_normal = 0.2
        self.sigma_spatial = 2.0
        self.buffers = RenderBuffers(width, height)
        self.frame_id = 0
        self.mode_id = 0
