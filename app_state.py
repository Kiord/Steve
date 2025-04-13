from path_tracing import RenderBuffers

class AppState:
    def __init__(self, width=1200, height=800):
        self.width = width
        self.height = height
        self.aspect_ratio = width / height
        self.profiling = False
        self.denoising = True
        self.tone_mapping = True
        self.spp = 1
        self.max_depth = 5
        self.radius = 2
        self.sigma_color = 0.2
        self.sigma_normal = 0.2
        self.sigma_spatial = 2.0
        self.buffers = RenderBuffers(width, height)
        self.frame_count = 0