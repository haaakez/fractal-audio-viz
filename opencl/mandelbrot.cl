// Reference source for the optional OpenCL preview backend.
// The compiled C++ backend embeds the same kernel so installed binaries do
// not depend on the current working directory.  It intentionally uses double
// precision and is limited to direct zooms below 1e6; deep MPFR/perturbation
// rendering remains on the validated native CPU path.
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel void mandelbrot_direct(
    __global float* output,
    const int width,
    const int height,
    const double center_real,
    const double center_imag,
    const double width_span,
    const double height_span,
    const int max_iter
) {
    const size_t pixel = get_global_id(0);
    const size_t count = (size_t)width * (size_t)height;
    if (pixel >= count) return;

    const int py = (int)(pixel / (size_t)width);
    const int px = (int)(pixel - (size_t)py * (size_t)width);
    const double cx = center_real
        + ((double)px - (double)(width - 1) * 0.5) * width_span / (double)width;
    const double cy = center_imag
        + ((double)(height - 1) * 0.5 - (double)py) * height_span / (double)height;

    const double q = (cx - 0.25) * (cx - 0.25) + cy * cy;
    const int in_cardioid = q * (q + cx - 0.25) <= 0.25 * cy * cy;
    const int in_bulb = (cx + 1.0) * (cx + 1.0) + cy * cy <= 0.0625;
    if (in_cardioid || in_bulb) {
        output[pixel] = (float)max_iter;
        return;
    }

    double zr = 0.0;
    double zi = 0.0;
    int iteration = 0;
    for (; iteration < max_iter; ++iteration) {
        const double next_real = zr * zr - zi * zi + cx;
        const double next_imag = 2.0 * zr * zi + cy;
        zr = next_real;
        zi = next_imag;
        const double magnitude_squared = zr * zr + zi * zi;
        if (magnitude_squared > 4.0) {
            const double magnitude = sqrt(fmax(magnitude_squared, 4.0000001));
            output[pixel] = (float)((double)(iteration + 1)
                - log(log(magnitude)) / log(2.0));
            return;
        }
    }
    output[pixel] = (float)max_iter;
}
