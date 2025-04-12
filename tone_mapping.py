import taichi as ti
from datatypes import vec3f

@ti.func
def linear_to_gamma(rgb: vec3f): # type: ignore
    rgb = ti.max(rgb, vec3f([0.0, 0.0, 0.0]))
    return ti.max(1.055 * rgb ** (1.0 / 2.4) - 0.055, vec3f([0.0, 0.0, 0.0]))


@ti.func
def uncharted2_tonemap(x: vec3f): # type: ignore
    A = 0.15
    B = 0.50
    C = 0.10
    D = 0.20
    E = 0.02
    F = 0.30
    return ((x * (A * x + C * B) + D * E) / (x * (A * x + B) + D * F)) - (E / F)


@ti.func
def aces_film(x: vec3f): # type: ignore
    a = 2.51
    b = 0.03
    c = 2.43
    d = 0.59
    e = 0.14
    return ti.max(ti.min((x * (a * x + b)) / (x * (c * x + d) + e), 1.0), 0.0)


@ti.func
def exposure_correct(col: vec3f, linfac: float, logfac: float): # type: ignore
    return linfac * (1.0 - ti.exp(col * logfac))


@ti.func
def aces_filmic_tone_map(col: vec3f): # type: ignore
    W = 10.2
    ExposureBias = 2.0
    curr = uncharted2_tonemap(col * ExposureBias)
    white_scale = uncharted2_tonemap(vec3f([W, W, W]))
    curr /= white_scale
    return linear_to_gamma(curr)

@ti.kernel
def tone_map(buffer:ti.template()): # type: ignore
    for i, j in buffer:
        buffer[i, j] = aces_filmic_tone_map(buffer[i, j])