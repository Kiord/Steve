import numpy as np
MAX_SPHERES = 100
MAX_TRIANGLES = 5_000_000
MAX_BVHS = 5
MAX_BVH_NODES = 20_000_000
MAX_PLANES = 16
MAX_MATERIALS = 100
MAX_EMITTERS = 5_000
EPS = 1e-5
MAX_DIST = 1e5
UP = np.array([0,1,0], dtype=np.float32)

BSDF_LAMBERT = 0
BSDF_PHONG = 1
BSDF_GGX = 2

EMITTER_NONE = 0
EMITTER_SPHERE = 1
EMITTER_TRIANGLE = 2

PRIMITIVE_NONE = 0
PRIMITIVE_SPHERE = 1
PRIMITIVE_TRIANGLE = 2
PRIMITIVE_PLANE = 3

RENDER_COLOR = 0
RENDER_DIRECT_LIGHT = 1
RENDER_DIRECT_BSDF = 2
RENDER_MIS_WEIGHTS = 3
RENDER_ALBEDO = 4
RENDER_NORMAL = 5
RENDER_BVH_DEPTH = 6
RENDER_DEPTH = 7
RENDER_BOX_TEST_COUNT = 8
NUM_RENDER_MODES = 9

RENDER_MODE_NAMES = [
    'COLOR',
    'DIRECT_LIGHT',
    'DIRECT_BSDF',
    'MIS_WEIGHTS',
    'ALBEDO',
    'NORMAL',
    'BVH_DEPTH',
    'DEPTH',
    'BOX_TEST_COUNT']