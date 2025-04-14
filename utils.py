import taichi as ti
from datatypes import vec3f
from scene import Material, Sphere
import math

@ti.dataclass
class RandomSampler:
    i: ti.i32#type: ignore
    j: ti.i32#type: ignore
    f: ti.i32#type: ignore  
    counter: ti.i32#type: ignore

    @ti.func
    def next(self) -> ti.f32:#type: ignore
        seed = self.i * ti.u32(73856093) ^ self.j * ti.u32(19349663) ^ self.f * ti.u32(83492791) ^ self.counter * ti.u32(2654435761)
        self.counter += 1
        return (ti.sin(seed * 0.0001) * 43758.5453) % 1.0
    
    @ti.func
    def next2(self) -> ti.f32:#type: ignore
        return self.next(), self.next()
    
    @ti.func
    def next3(self) -> ti.f32:#type: ignore
        return self.next(), self.next(), self.next()


@ti.func
def reflect(v: vec3f, n: vec3f) -> vec3f:
    return v - 2.0 * v.dot(n) * n

# @ti.func
# def rotate(axis: vec3f, angle: ti.f32, v: vec3f) -> vec3f:
#     c = ti.cos(angle)
#     s = ti.sin(angle)
#     return v * c + axis.cross(v) * s + axis * (axis.dot(v)) * (1.0 - c)

@ti.func
def rotate(axis: vec3f, angle: ti.f32, v: vec3f) -> vec3f:
    half_angle = 0.5 * angle
    s = ti.sin(half_angle)
    w = ti.cos(half_angle)
    u = axis * s

    return (
        2.0 * v.dot(u) * u +
        (w * w - u.dot(u)) * v +
        2.0 * w * u.cross(v)
    )

@ti.dataclass
class DirectionSample:
    direction: vec3f # type: ignore
    pdf: ti.f32  # type: ignore
    bsdf: vec3f # type: ignore

@ti.func
def randomDirectionHemisphere(main_dir: vec3f, n: ti.f32, sampler: RandomSampler) -> vec3f:
    r1 = sampler.next()
    r2 = sampler.next()
    phi = 2.0 * math.pi * r1
    cos_theta = r2 ** (1.0 / (n + 1.0))
    sin_theta = ti.sqrt(1.0 - cos_theta * cos_theta)

    local = ti.Vector([
        ti.cos(phi) * sin_theta,
        ti.sin(phi) * sin_theta,
        cos_theta
    ])
    
    result = ti.Vector([main_dir.z, 0.0, 0.0]) * local

    if ti.abs(main_dir.z) <= 0.99999:
        up = ti.Vector([0.0, 0.0, 1.0])
        axis = main_dir.cross(up).normalized()
        angle = ti.acos(main_dir.dot(up))
        result = rotate(axis, angle, local)
    return result

@ti.func
def sample_BSDF(normal: vec3f, material:Material, incoming_dir: vec3f, sampler: RandomSampler) -> DirectionSample:
    n = material.shininess
    is_lambert = n == 0.0
    r = ti.select(is_lambert, normal, reflect(-incoming_dir, normal))

    sampled = DirectionSample(
        direction=ti.Vector([0.0, 0.0, 0.0]),
        pdf=0.0,
        bsdf=ti.Vector([0.0, 0.0, 0.0])
    )

    sampled.direction = randomDirectionHemisphere(r, max(n, 1.0), sampler)

    cosr = r.dot(sampled.direction)
    correct_r = 1.0 if cosr > 0.0 else -1.0
    cosr *= correct_r
    sampled.direction *= correct_r

    coswo = normal.dot(sampled.direction)
    ok = coswo > 0.0
    cosr_pow_n = ti.pow(abs(cosr), n)

    w = (n + 1.0 + ti.cast(is_lambert, ti.f32)) / (2.0 * math.pi)

    sampled.pdf = ti.select(ok, w * (coswo if is_lambert else cosr_pow_n), 0.0)
    sampled.bsdf = ti.select(ok, material.diffuse * w * cosr_pow_n, ti.Vector([0.0, 0.0, 0.0]))

    return sampled


@ti.func
def BSDF(material, normal, wi, wo):
    n = material.shininess
    is_lambert = n == 0.0
    coswi = normal.dot(wi)
    r = ti.select(is_lambert, normal, reflect(wo, normal))
    cosr = r.dot(wi)
    coswo = max(min(normal.dot(wo), 1.0), -1)
    ok = (coswi > 0.0) and (cosr > 0.0) and (coswo > 0.0)
    factor = (n + 1.0 + ti.cast(is_lambert, ti.f32)) / (2.0 * math.pi)
    return ti.select(ok, material.diffuse * factor * cosr ** n, ti.Vector([0.0, 0.0, 0.0]))


@ti.func
def normalize(v):
    return v / v.norm()

@ti.func
def random_unit_vector(sampler:RandomSampler):
    u1, u2 = sampler.next2()
    theta = 2 * math.pi * u1
    z = u2 * 2 - 1
    r = (1 - z * z).sqrt()
    return ti.Vector([r * ti.cos(theta), r * ti.sin(theta), z])

@ti.dataclass
class Contribution:
    value: vec3f # type:ignore
    pdf: ti.f32 # type:ignore

@ti.dataclass
class LightSample:
    direction: vec3f
    contrib: vec3f
    pdf: ti.f32

@ti.dataclass
class SurfaceLightSample:
    pdf: ti.f32
    point: vec3f
    normal: vec3f

@ti.func
def copysign(x: ti.f32, sign_source: ti.f32) -> ti.f32:
    return ti.select(sign_source >= 0.0, ti.abs(x), -ti.abs(x))


@ti.func
def sample_sphere_solid_angle(viewer: vec3f, sphere: Sphere, sampler: RandomSampler) -> SurfaceLightSample:
    sls = SurfaceLightSample(pdf=0.0, point=vec3f(0.0), normal=vec3f(0.0))

    # Step 1: setup view direction and distance
    main_dir = viewer - sphere.center
    d2 = main_dir.dot(main_dir)
    d = ti.sqrt(d2)
    main_dir = main_dir / d

    r = sphere.radius
    r2 = r * r
    sintheta_max = r / d
    costheta_max = ti.sqrt(1.0 - sintheta_max * sintheta_max)

    # Step 2: sample θ and φ
    u1 = sampler.next()
    u2 = sampler.next()
    costheta = 1.0 - u1 * (1.0 - costheta_max)
    sintheta = ti.sqrt(1.0 - costheta * costheta)
    phi = 2.0 * math.pi * u2

    # Step 3: geometric correction
    sintheta2 = sintheta * sintheta
    D = 1.0 - d2 * sintheta2 / r2
    D_positive = D > 0.0

    cos_alpha = ti.select(
        D_positive,
        sintheta2 / sintheta_max + costheta * ti.sqrt(ti.abs(D)),
        sintheta_max
    )

    sin_alpha = ti.sqrt(1.0 - cos_alpha * cos_alpha)

    local_dir = vec3f(
        sin_alpha * ti.cos(phi),
        sin_alpha * ti.sin(phi),
        cos_alpha
    )

    # Step 4: rotate local_dir into world space aligned with main_dir
    if ti.abs(main_dir.z) > 0.99999:
        sls.normal = local_dir * ti.select(main_dir.z >= 0.0, 1.0, -1.0)
    else:
        axis = vec3f(0.0, 0.0, 1.0).cross(main_dir).normalized()
        angle = ti.acos(main_dir.z)
        sls.normal = rotate(axis, angle, local_dir)

    # Step 5: compute point and PDF
    sls.point = sphere.center + r * sls.normal
    solid_angle = 2.0 * math.pi * (1.0 - costheta_max)
    sls.pdf = 1.0 / solid_angle

    return sls


@ti.func
def sample_sphere_uniform(viewer: vec3f, sphere: Sphere, sampler: RandomSampler) -> SurfaceLightSample:
    sls = SurfaceLightSample(pdf=0.0, point=vec3f(0.0), normal=vec3f(0.0))

    ksi_x, ksi_y = sampler.next2()

    polar = ti.acos(1.0 - 2.0 * ksi_x)
    azimuth = 2.0 * math.pi * ksi_y

    sin_polar = ti.sin(polar)
    sls.normal = vec3f(
        sin_polar * ti.cos(azimuth),
        sin_polar * ti.sin(azimuth),
        ti.cos(polar)
    )

    sls.point = sphere.center + sphere.radius * sls.normal
    sls.pdf = 1.0 / (4.0 * math.pi * sphere.radius * sphere.radius)

    return sls


@ti.func
def sample_uniform_hemisphere(main_dir: vec3f, sampler: RandomSampler) -> vec3f:
    r1, r2 = sampler.next2()

    z = r1
    r = ti.sqrt(1.0 - z * z)
    phi = 2.0 * math.pi * r2
    x = r * ti.cos(phi)
    y = r * ti.sin(phi)

    # Build ONB
    w = main_dir.normalized()
    a = vec3f(0.0, 1.0, 0.0) if ti.abs(w.x) > 0.9 else vec3f(1.0, 0.0, 0.0)
    v = w.cross(a).normalized()
    u = v.cross(w)

    return (u * x + v * y + w * z).normalized()

@ti.func
def sample_cosine_hemisphere(main_dir: vec3f, sampler: RandomSampler) -> vec3f:
    r1, r2 = sampler.next2()

    phi = 2.0 * math.pi * r1
    r = ti.sqrt(r2)
    x = ti.cos(phi) * r
    y = ti.sin(phi) * r
    z = ti.sqrt(1.0 - r2)

    # Build ONB
    w = main_dir.normalized()
    a = vec3f(0.0, 1.0, 0.0) if ti.abs(w.x) > 0.9 else vec3f(1.0, 0.0, 0.0)
    v = w.cross(a).normalized()
    u = v.cross(w)

    return (u * x + v * y + w * z).normalized()

@ti.func
def sample_sphere_hemisphere_uniform(viewer: vec3f, sphere: Sphere, sampler: RandomSampler) -> SurfaceLightSample:
    sls = SurfaceLightSample(pdf=0.0, point=vec3f(0.0), normal=vec3f(0.0))

    main_dir = (viewer - sphere.center).normalized()
    sls.normal = sample_uniform_hemisphere(main_dir, sampler)

    # Flip normal to ensure it's in the visible hemisphere
    if sls.normal.dot(main_dir) < 0.0:
        sls.normal = -sls.normal

    sls.point = sphere.center + sphere.radius * sls.normal
    sls.pdf = 1.0 / (2.0 * math.pi * sphere.radius * sphere.radius)

    return sls

@ti.func
def sample_sphere_hemisphere_cosine(viewer: vec3f, sphere: Sphere, sampler: RandomSampler) -> SurfaceLightSample:
    sls = SurfaceLightSample(pdf=0.0, point=vec3f(0.0), normal=vec3f(0.0))

    main_dir = (viewer - sphere.center).normalized()
    sls.normal = sample_cosine_hemisphere(main_dir, sampler)

    sls.point = sphere.center + sphere.radius * sls.normal
    sls.pdf = ti.max(main_dir.dot(sls.normal), 0.0) / (math.pi * sphere.radius * sphere.radius)

    return sls
