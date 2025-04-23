import taichi as ti

@ti.kernel
def bilateral_filter(buffers: ti.template(), sigma_color: ti.f32, sigma_normal: ti.f32, sigma_spatial: ti.f32, radius: ti.i32): # type:ignore
    for i, j in buffers.color:

        center_color = buffers.accum_color[i, j]
        center_albedo = buffers.albedo[i, j]
        center_normal = buffers.normal[i, j]

        result = ti.Vector([0.0, 0.0, 0.0])
        total_weight = 0.0

        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                ni = i + dx
                nj = j + dy
                if 0 <= ni < buffers.accum_color.shape[0] and 0 <= nj < buffers.accum_color.shape[1]:
                    sample_color = buffers.accum_color[ni, nj]
                    sample_albedo = buffers.albedo[ni, nj]
                    sample_normal = buffers.normal[ni, nj]

                    # Gaussian spatial distance
                    dist2 = dx * dx + dy * dy
                    ws = ti.exp(-dist2 / (2 * sigma_spatial ** 2))

                    # Feature similarity
                    wc = ti.exp(-((sample_color - center_color).norm_sqr()) / (2 * sigma_color ** 2))
                    wa = ti.exp(-((sample_albedo - center_albedo).norm_sqr()) / (2 * sigma_color ** 2))
                    wn = ti.exp(-((sample_normal - center_normal).norm_sqr()) / (2 * sigma_normal ** 2))

                    weight = ws * wc * wa * wn
                    result += sample_color * weight
                    total_weight += weight

        buffers.denoised[i, j] = result / total_weight if total_weight > 0 else center_color

