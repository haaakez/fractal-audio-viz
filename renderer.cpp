// Mandelbrot renderer with reusable MPFR reference orbits, scaled
// mantissa/exponent perturbation, and hierarchical BLA maps. The exported
// functions intentionally use a C ABI so Python can drive the native core
// without knowing its C++ types.

#include <algorithm>
#include <atomic>
#include <array>
#include <cerrno>
#include <cstdint>
#include <cmath>
#include <chrono>
#include <cstdlib>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "renderer.h"

#if defined(__AVX2__)
#include <immintrin.h>
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef FRACTAL_HAVE_MPFR
#include <mpfr.h>
#endif

#ifdef FRACTAL_HAVE_OPENCL
#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>
#endif

bool avx2_runtime_available() noexcept {
#if defined(__AVX2__)
#if defined(__GNUC__) || defined(__clang__)
    __builtin_cpu_init();
    return __builtin_cpu_supports("avx2") != 0;
#else
    return true;
#endif
#else
    return false;
#endif
}

namespace {

constexpr int ABI_VERSION = FRACTAL_ABI_VERSION;
constexpr int RENDER_OPTIONS_VERSION = FRACTAL_RENDER_OPTIONS_VERSION;
// The current degree-three bivariate composition is validated through 64
// iterations at the bundled boundary location.  Longer maps are still
// useful in the linear deep tier, but using them for the cubic map creates
// visible smooth-escape bands before the endpoint guard can notice.
constexpr int MAX_SAFE_BLA_LENGTH = 64;
// Long linear maps are enabled only in the ultra-deep tier below; ordinary
// e12--e40 frames stay on the independently validated 256/1024 limits.
constexpr int MAX_SAFE_LINEAR_BLA_LENGTH = 4096;
constexpr int MAX_SAFE_DEEP_LINEAR_BLA_LENGTH = 1024;
constexpr long double ESCAPE_RADIUS_SQUARED = 4.0L;
constexpr long double LOG_TWO = 0.693147180559945309417232121458176568L;
constexpr long double LOG_TEN = 2.302585092994045684017991454684364208L;
constexpr int MAX_NATIVE_ITERATIONS = 10'000'000;
constexpr int MAX_NATIVE_THREADS = 4096;
constexpr int MAX_NATIVE_PRECISION_BITS = 131'072;
constexpr int MAX_NATIVE_PIXELS = 100'000'000;
constexpr int MAX_NATIVE_POINTS = 100'000'000;
constexpr std::size_t MAX_NATIVE_TEXT_LENGTH = 50'000;
constexpr long double MIN_NATIVE_LOG10_ZOOM = -300.0L;
constexpr long double MAX_NATIVE_LOG10_ZOOM = 9800.0L;

bool valid_formula(int formula) noexcept {
    return formula >= FRACTAL_FORMULA_MANDELBROT
        && formula <= FRACTAL_FORMULA_TRICORN;
}

bool valid_pixel_dimensions(int width, int height) noexcept {
    if (width <= 0 || height <= 0) return false;
    const auto maximum = static_cast<std::uint64_t>(MAX_NATIVE_PIXELS);
    return static_cast<std::uint64_t>(width)
        <= maximum / static_cast<std::uint64_t>(height);
}

bool valid_colour_controls(
    double phase,
    double vocal,
    double instrumental,
    double pitch
) noexcept {
    return std::isfinite(phase)
        && std::isfinite(vocal)
        && std::isfinite(instrumental)
        && std::isfinite(pitch);
}

bool valid_iteration_count(int value) noexcept {
    return value > 0 && value <= MAX_NATIVE_ITERATIONS;
}

bool valid_thread_count(int value) noexcept {
    return value >= 0 && value <= MAX_NATIVE_THREADS;
}

bool valid_precision_bits(int value) noexcept {
    return value >= 128 && value <= MAX_NATIVE_PRECISION_BITS;
}

bool valid_series_parameters(int series_order, int series_block) noexcept {
    return series_order >= 1 && series_order <= 32
        && series_block >= 2 && series_block <= 4096;
}

bool valid_c_string(const char* text) noexcept {
    if (!text) return false;
    for (std::size_t length = 0; length <= MAX_NATIVE_TEXT_LENGTH; ++length) {
        if (text[length] == '\0') return true;
    }
    return false;
}

long double parse_coordinate(const char* text, const char* label) {
    if (!valid_c_string(text) || !label) {
        throw std::runtime_error("native coordinate text is too long or null");
    }
    errno = 0;
    char* end = nullptr;
    const long double value = std::strtold(text, &end);
    if (end == text || *end != '\0' || !std::isfinite(value)) {
        throw std::runtime_error(std::string("invalid ") + label + " coordinate");
    }
    // Underflow to zero is harmless for the direct fallback and the original
    // text is still passed to MPFR for deep references. Overflow, handled by
    // the non-finite check above, cannot be represented by any native path.
    return value;
}

int formula_power(int formula) noexcept {
    (void)formula;
    return 2;
}

void iterate_direct_formula(
    int formula,
    double zr,
    double zi,
    double parameter_real,
    double parameter_imag,
    double& next_real,
    double& next_imag
) noexcept {
    if (formula == FRACTAL_FORMULA_BURNING_SHIP) {
        const double absolute_real = std::abs(zr);
        const double absolute_imag = std::abs(zi);
        // Keep the piecewise alternate maps reproducible with the NumPy
        // fallback.  A fused multiply-add changes a late orbit by a few ulps
        // and can move a boundary pixel to the other side of escape.
        const volatile double real_square = absolute_real * absolute_real;
        const volatile double imag_square = absolute_imag * absolute_imag;
        const volatile double cross = 2.0 * absolute_real * absolute_imag;
        const volatile double real_difference = real_square - imag_square;
        const volatile double imaginary_product = cross + parameter_imag;
        next_real = real_difference + parameter_real;
        next_imag = imaginary_product;
    } else if (formula == FRACTAL_FORMULA_TRICORN) {
        const volatile double real_square = zr * zr;
        const volatile double imag_square = zi * zi;
        const volatile double cross = -2.0 * zr * zi;
        const volatile double real_difference = real_square - imag_square;
        const volatile double imaginary_product = cross + parameter_imag;
        next_real = real_difference + parameter_real;
        next_imag = imaginary_product;
    } else if (formula == FRACTAL_FORMULA_JULIA) {
        const volatile double real_square = zr * zr;
        const volatile double imag_square = zi * zi;
        const volatile double cross = 2.0 * zr * zi;
        const volatile double real_difference = real_square - imag_square;
        const volatile double imaginary_product = cross + parameter_imag;
        next_real = real_difference + parameter_real;
        next_imag = imaginary_product;
    } else {
        next_real = zr * zr - zi * zi + parameter_real;
        next_imag = 2.0 * zr * zi + parameter_imag;
    }
}

#ifdef FRACTAL_HAVE_OPENCL

// OpenCL reports collection sizes through the driver. Treat those values as
// untrusted metadata: a broken ICD must not turn capability probing into an
// arbitrarily large host allocation or diagnostic string.
constexpr cl_uint MAX_OPENCL_PLATFORMS = 256;
constexpr cl_uint MAX_OPENCL_DEVICES = 256;
constexpr size_t MAX_OPENCL_INFO_BYTES = 1U << 20;

// The OpenCL path is deliberately limited to the direct, shallow renderer.
// Deep MPFR reference construction and perturbation/BLA remain on the CPU
// until their scaled representation has a validated device implementation.
// This keeps backend=2 useful for previews without silently making deep output
// less correct merely because an OpenCL device is present.
constexpr const char* OPENCL_DIRECT_KERNEL = R"CLC(
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
)CLC";

struct OpenClRuntime {
    cl_context context = nullptr;
    cl_command_queue queue = nullptr;
    cl_program program = nullptr;
    cl_kernel kernel = nullptr;
    std::mutex mutex;
    std::string error;

    ~OpenClRuntime() {
        if (kernel) clReleaseKernel(kernel);
        if (program) clReleaseProgram(program);
        if (queue) clReleaseCommandQueue(queue);
        if (context) clReleaseContext(context);
    }
};

std::once_flag opencl_once;
std::unique_ptr<OpenClRuntime> opencl_runtime;

std::string opencl_error_text(cl_int status) {
    return "OpenCL error " + std::to_string(static_cast<int>(status));
}

void initialise_opencl() {
    std::call_once(opencl_once, [] {
        auto runtime = std::make_unique<OpenClRuntime>();
        cl_uint platform_count = 0;
        cl_int status = clGetPlatformIDs(0, nullptr, &platform_count);
        if (status != CL_SUCCESS || platform_count == 0) {
            runtime->error = status == CL_SUCCESS
                ? "no OpenCL platform is installed"
                : opencl_error_text(status);
            opencl_runtime = std::move(runtime);
            return;
        }
        if (platform_count > MAX_OPENCL_PLATFORMS) {
            runtime->error = "OpenCL reported too many platforms";
            opencl_runtime = std::move(runtime);
            return;
        }
        std::vector<cl_platform_id> platforms(platform_count);
        status = clGetPlatformIDs(platform_count, platforms.data(), nullptr);
        if (status != CL_SUCCESS) {
            runtime->error = opencl_error_text(status);
            opencl_runtime = std::move(runtime);
            return;
        }

        cl_device_id selected = nullptr;
        cl_platform_id selected_platform = nullptr;
        for (cl_platform_id platform : platforms) {
            cl_uint device_count = 0;
            if (clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, nullptr, &device_count)
                    == CL_SUCCESS && device_count > 0) {
                if (device_count > MAX_OPENCL_DEVICES) continue;
                std::vector<cl_device_id> devices(device_count);
                if (clGetDeviceIDs(
                        platform, CL_DEVICE_TYPE_GPU, device_count,
                        devices.data(), nullptr) == CL_SUCCESS) {
                    selected = devices.front();
                    selected_platform = platform;
                    break;
                }
            }
        }
        if (!selected) {
            for (cl_platform_id platform : platforms) {
                cl_uint device_count = 0;
                if (clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 0, nullptr, &device_count)
                        == CL_SUCCESS && device_count > 0) {
                    if (device_count > MAX_OPENCL_DEVICES) continue;
                    std::vector<cl_device_id> devices(device_count);
                    if (clGetDeviceIDs(
                            platform, CL_DEVICE_TYPE_CPU, device_count,
                            devices.data(), nullptr) == CL_SUCCESS) {
                        selected = devices.front();
                        selected_platform = platform;
                        break;
                    }
                }
            }
        }
        if (!selected || !selected_platform) {
            runtime->error = "no OpenCL GPU or CPU device is available";
            opencl_runtime = std::move(runtime);
            return;
        }

        size_t extension_size = 0;
        status = clGetDeviceInfo(
            selected, CL_DEVICE_EXTENSIONS, 0, nullptr, &extension_size);
        if (status != CL_SUCCESS || extension_size == 0
            || extension_size > MAX_OPENCL_INFO_BYTES) {
            runtime->error = status != CL_SUCCESS
                ? opencl_error_text(status)
                : "selected OpenCL device reported invalid extension metadata";
            opencl_runtime = std::move(runtime);
            return;
        }
        std::string extensions(extension_size, '\0');
        if (clGetDeviceInfo(
                selected, CL_DEVICE_EXTENSIONS, extension_size,
                extensions.data(), nullptr) != CL_SUCCESS
            || (extensions.find("cl_khr_fp64") == std::string::npos
                && extensions.find("cl_amd_fp64") == std::string::npos)) {
            runtime->error = "selected OpenCL device has no double-precision extension";
            opencl_runtime = std::move(runtime);
            return;
        }

        cl_context_properties properties[] = {
            CL_CONTEXT_PLATFORM,
            reinterpret_cast<cl_context_properties>(selected_platform),
            0,
        };
        runtime->context = clCreateContext(
            properties, 1, &selected, nullptr, nullptr, &status);
        if (status != CL_SUCCESS || !runtime->context) {
            runtime->error = opencl_error_text(status);
            opencl_runtime = std::move(runtime);
            return;
        }
        runtime->queue = clCreateCommandQueue(runtime->context, selected, 0, &status);
        if (status != CL_SUCCESS || !runtime->queue) {
            runtime->error = opencl_error_text(status);
            opencl_runtime = std::move(runtime);
            return;
        }
        const size_t source_length = std::char_traits<char>::length(OPENCL_DIRECT_KERNEL);
        const char* kernel_source = OPENCL_DIRECT_KERNEL;
        runtime->program = clCreateProgramWithSource(
            runtime->context, 1, &kernel_source, &source_length, &status);
        if (status != CL_SUCCESS || !runtime->program) {
            runtime->error = opencl_error_text(status);
            opencl_runtime = std::move(runtime);
            return;
        }
        status = clBuildProgram(runtime->program, 1, &selected, "", nullptr, nullptr);
        if (status != CL_SUCCESS) {
            size_t log_size = 0;
            clGetProgramBuildInfo(
                runtime->program, selected, CL_PROGRAM_BUILD_LOG,
                0, nullptr, &log_size);
            if (log_size > MAX_OPENCL_INFO_BYTES) {
                runtime->error = opencl_error_text(status)
                    + ": OpenCL build log exceeded the safety limit";
                opencl_runtime = std::move(runtime);
                return;
            }
            std::string build_log(log_size, '\0');
            if (log_size > 0) {
                clGetProgramBuildInfo(
                    runtime->program, selected, CL_PROGRAM_BUILD_LOG,
                    log_size, build_log.data(), nullptr);
            }
            runtime->error = opencl_error_text(status) + ": " + build_log;
            opencl_runtime = std::move(runtime);
            return;
        }
        runtime->kernel = clCreateKernel(runtime->program, "mandelbrot_direct", &status);
        if (status != CL_SUCCESS || !runtime->kernel) {
            runtime->error = opencl_error_text(status);
        }
        opencl_runtime = std::move(runtime);
    });
}

bool opencl_available() {
    initialise_opencl();
    return opencl_runtime != nullptr
        && opencl_runtime->kernel != nullptr
        && opencl_runtime->error.empty();
}

void render_direct_opencl(
    float* output,
    int width,
    int height,
    double zoom,
    double x_center,
    double y_center,
    int max_iter
) {
    initialise_opencl();
    if (!opencl_available()) {
        throw std::runtime_error(
            opencl_runtime && !opencl_runtime->error.empty()
                ? opencl_runtime->error
                : "OpenCL direct backend is unavailable");
    }
    OpenClRuntime& runtime = *opencl_runtime;
    std::lock_guard<std::mutex> lock(runtime.mutex);
    const double height_span = 2.8 / zoom;
    const double width_span = height_span * static_cast<double>(width)
        / static_cast<double>(height);
    const size_t count = static_cast<size_t>(width) * static_cast<size_t>(height);
    cl_int status = CL_SUCCESS;
    cl_mem device_output = clCreateBuffer(
        runtime.context, CL_MEM_WRITE_ONLY,
        count * sizeof(float), nullptr, &status);
    if (status != CL_SUCCESS || !device_output) {
        throw std::runtime_error(opencl_error_text(status));
    }
    auto release_output = [&] { clReleaseMemObject(device_output); };
    status = clSetKernelArg(runtime.kernel, 0, sizeof(device_output), &device_output);
    status |= clSetKernelArg(runtime.kernel, 1, sizeof(width), &width);
    status |= clSetKernelArg(runtime.kernel, 2, sizeof(height), &height);
    status |= clSetKernelArg(runtime.kernel, 3, sizeof(x_center), &x_center);
    status |= clSetKernelArg(runtime.kernel, 4, sizeof(y_center), &y_center);
    status |= clSetKernelArg(runtime.kernel, 5, sizeof(width_span), &width_span);
    status |= clSetKernelArg(runtime.kernel, 6, sizeof(height_span), &height_span);
    status |= clSetKernelArg(runtime.kernel, 7, sizeof(max_iter), &max_iter);
    if (status != CL_SUCCESS) {
        release_output();
        throw std::runtime_error(opencl_error_text(status));
    }
    const size_t global_size = ((count + 255U) / 256U) * 256U;
    status = clEnqueueNDRangeKernel(
        runtime.queue, runtime.kernel, 1, nullptr,
        &global_size, nullptr, 0, nullptr, nullptr);
    if (status == CL_SUCCESS) {
        status = clEnqueueReadBuffer(
            runtime.queue, device_output, CL_TRUE, 0,
            count * sizeof(float), output, 0, nullptr, nullptr);
    }
    if (status == CL_SUCCESS) status = clFinish(runtime.queue);
    release_output();
    if (status != CL_SUCCESS) throw std::runtime_error(opencl_error_text(status));
}

#endif

// Each C-ABI caller gets its own diagnostic buffer. Native reference tiers
// are prepared concurrently, so a process-wide error string would otherwise
// let one failed worker overwrite another worker's useful message.
thread_local std::string last_error;

struct PaletteBasis {
    int max_iter = -1;
    std::vector<float> cosine;
    std::vector<float> sine;
};

struct AuroraPalette {
    std::vector<std::array<std::uint8_t, 3>> rgb;
};

// These counters are intentionally opt-in.  A normal render pays only for
// one predictable null-pointer branch at the points where a diagnostic event
// is recorded; the per-thread arrays and aggregation are allocated only when
// the benchmark explicitly enables statistics.
struct RenderStats {
    std::uint64_t pixels = 0;
    std::uint64_t logical_iterations = 0;
    std::uint64_t bla_blocks = 0;
    std::uint64_t linear_blocks = 0;
    std::uint64_t cubic_blocks = 0;
    std::uint64_t exact_steps = 0;
    std::uint64_t replay_steps = 0;
    std::uint64_t bla_retries = 0;
    std::uint64_t cycle_inside = 0;
    std::uint64_t double_tail_pixels = 0;
    std::uint64_t bla_disabled_pixels = 0;
    std::uint64_t tail_steps = 0;
    std::uint64_t max_tail_steps = 0;
    std::uint64_t tail_rebases = 0;
    std::uint64_t tail_rebase_fallbacks = 0;
    std::uint64_t max_pixel_iterations = 0;
    std::uint64_t series_pixels = 0;
    std::uint64_t series_jumps = 0;
    std::uint64_t glitch_count = 0;
    std::uint64_t unresolved_pixels = 0;
    std::uint64_t deadline_aborts = 0;
    std::uint64_t secondary_references = 0;
    std::uint64_t render_ns = 0;
    std::array<std::uint64_t, 16> bla_length_histogram{};
};

constexpr int RENDER_STATS_FIELDS = 16;
constexpr int RENDER_STATS_EXTENDED_FIELDS = 39;

std::atomic<bool> render_stats_enabled{false};
std::mutex stats_mutex;
RenderStats last_render_stats;

[[maybe_unused]] void publish_render_stats(const RenderStats& stats) {
    std::lock_guard<std::mutex> lock(stats_mutex);
    last_render_stats = stats;
}

int copy_render_stats(std::uint64_t* values, int capacity) {
    if (!values || capacity < RENDER_STATS_FIELDS) return -1;
    std::lock_guard<std::mutex> lock(stats_mutex);
    values[0] = last_render_stats.pixels;
    values[1] = last_render_stats.logical_iterations;
    values[2] = last_render_stats.bla_blocks;
    values[3] = last_render_stats.linear_blocks;
    values[4] = last_render_stats.cubic_blocks;
    values[5] = last_render_stats.exact_steps;
    values[6] = last_render_stats.replay_steps;
    values[7] = last_render_stats.bla_retries;
    values[8] = last_render_stats.cycle_inside;
    values[9] = last_render_stats.double_tail_pixels;
    values[10] = last_render_stats.bla_disabled_pixels;
    values[11] = last_render_stats.tail_steps;
    values[12] = last_render_stats.max_tail_steps;
    values[13] = last_render_stats.tail_rebases;
    values[14] = last_render_stats.tail_rebase_fallbacks;
    values[15] = last_render_stats.max_pixel_iterations;
    return RENDER_STATS_FIELDS;
}

int copy_extended_render_stats(std::uint64_t* values, int capacity) {
    if (!values || capacity < RENDER_STATS_EXTENDED_FIELDS) return -1;
    std::lock_guard<std::mutex> lock(stats_mutex);
    values[0] = last_render_stats.pixels;
    values[1] = last_render_stats.logical_iterations;
    values[2] = last_render_stats.bla_blocks;
    values[3] = last_render_stats.linear_blocks;
    values[4] = last_render_stats.cubic_blocks;
    values[5] = last_render_stats.exact_steps;
    values[6] = last_render_stats.replay_steps;
    values[7] = last_render_stats.bla_retries;
    values[8] = last_render_stats.cycle_inside;
    values[9] = last_render_stats.double_tail_pixels;
    values[10] = last_render_stats.bla_disabled_pixels;
    values[11] = last_render_stats.tail_steps;
    values[12] = last_render_stats.max_tail_steps;
    values[13] = last_render_stats.tail_rebases;
    values[14] = last_render_stats.tail_rebase_fallbacks;
    values[15] = last_render_stats.max_pixel_iterations;
    values[16] = last_render_stats.series_pixels;
    values[17] = last_render_stats.series_jumps;
    values[18] = last_render_stats.glitch_count;
    values[19] = last_render_stats.unresolved_pixels;
    values[20] = last_render_stats.deadline_aborts;
    values[21] = last_render_stats.secondary_references;
    values[22] = last_render_stats.render_ns;
    for (size_t index = 0; index < last_render_stats.bla_length_histogram.size(); ++index) {
        values[23 + index] = last_render_stats.bla_length_histogram[index];
    }
    return RENDER_STATS_EXTENDED_FIELDS;
}

FractalRenderOptions default_render_options() {
    FractalRenderOptions options{};
    options.struct_size = sizeof(FractalRenderOptions);
    options.version = RENDER_OPTIONS_VERSION;
    options.strict = 1;
    options.allow_recovery = 0;
    options.time_budget_ms = 0;
    options.disable_bla = 0;
    options.disable_cycle = 0;
    options.strict_cycle = 0;
    options.series_min_terms = 8;
    options.series_max_terms = 32;
    options.max_bla_length = MAX_SAFE_BLA_LENGTH;
    options.max_linear_bla_length = MAX_SAFE_LINEAR_BLA_LENGTH;
    options.backend = 0; // scalar/native backend; future values are explicit.
    return options;
}

FractalRenderOptions checked_render_options(const FractalRenderOptions* supplied) {
    FractalRenderOptions options = default_render_options();
    if (!supplied) return options;
    if (supplied->struct_size < sizeof(FractalRenderOptions)
        || supplied->version != RENDER_OPTIONS_VERSION) {
        throw std::runtime_error("unsupported FractalRenderOptions version or size");
    }
    options = *supplied;
    const auto validate_flag = [](std::int32_t value, const char* label) {
        if (value != 0 && value != 1) {
            throw std::runtime_error(std::string(label) + " must be 0 or 1");
        }
    };
    validate_flag(options.strict, "strict render flag");
    validate_flag(options.allow_recovery, "recovery flag");
    validate_flag(options.disable_bla, "disable-BLA flag");
    validate_flag(options.disable_cycle, "disable-cycle flag");
    validate_flag(options.strict_cycle, "strict-cycle flag");
    options.series_min_terms = std::clamp(options.series_min_terms, 1, 32);
    options.series_max_terms = std::clamp(options.series_max_terms, options.series_min_terms, 32);
    options.max_bla_length = std::clamp(options.max_bla_length, 1, MAX_SAFE_BLA_LENGTH);
    options.max_linear_bla_length = std::clamp(
        options.max_linear_bla_length, 1, MAX_SAFE_LINEAR_BLA_LENGTH);
    if (options.time_budget_ms < 0) {
        throw std::runtime_error("render time budget cannot be negative");
    }
    return options;
}

inline void record_bla_length(RenderStats* stats, int length) {
    if (!stats || length <= 0) return;
    int bucket = 0;
    unsigned int value = static_cast<unsigned int>(length);
    while (value > 1U && bucket < 15) {
        value >>= 1U;
        ++bucket;
    }
    ++stats->bla_length_histogram[static_cast<size_t>(bucket)];
}

// The colourizer is called from Python's calling thread, while OpenMP only
// parallelises its pixel loop.  Thread-local storage therefore gives each
// caller a reusable LUT without locks or cross-renderer interference.
thread_local PaletteBasis palette_basis;
thread_local AuroraPalette aurora_palette;

void set_error(const std::string& message) {
    last_error = message;
}

const PaletteBasis& colour_basis_for(int max_iter) {
    const int palette_size = std::min(65536, std::max(4096, max_iter * 4));
    if (palette_basis.max_iter == max_iter
        && static_cast<int>(palette_basis.cosine.size()) == palette_size) {
        return palette_basis;
    }
    palette_basis.max_iter = max_iter;
    palette_basis.cosine.resize(static_cast<size_t>(palette_size));
    palette_basis.sine.resize(static_cast<size_t>(palette_size));
    const double denominator = std::max(1, palette_size - 1);
    for (int index = 0; index < palette_size; ++index) {
        const double angle = 0.15 * static_cast<double>(max_iter)
            * static_cast<double>(index) / denominator;
        palette_basis.cosine[static_cast<size_t>(index)] = static_cast<float>(std::cos(angle));
        palette_basis.sine[static_cast<size_t>(index)] = static_cast<float>(std::sin(angle));
    }
    return palette_basis;
}

inline std::uint8_t colour_byte(double value) {
    return static_cast<std::uint8_t>(std::clamp(value, 0.0, 255.0));
}

constexpr double TWO_PI = 6.283185307179586476925286766559005768;
// Keep the old blue/yellow gradient at the median pitch.  A full-range pitch
// deviation rotates it by about 58 degrees, enough to move the two anchors
// through neighbouring hues without destroying their separation.
constexpr double PITCH_HUE_SWING_TURNS = 0.16;

double pitch_hue_angle(double pitch) {
    const double signed_pitch = std::clamp(2.0 * (pitch - 0.5), -1.0, 1.0);
    return signed_pitch * PITCH_HUE_SWING_TURNS * TWO_PI;
}

std::array<double, 3> rotate_hue_rgb(
    const std::array<double, 3>& rgb,
    double cosine,
    double sine
) {
    // Rotate chroma in YIQ space. This preserves the old channel-wave
    // luminance while moving both of its characteristic hues together.
    const double y = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2];
    const double i = 0.596 * rgb[0] - 0.275 * rgb[1] - 0.321 * rgb[2];
    const double q = 0.212 * rgb[0] - 0.523 * rgb[1] + 0.311 * rgb[2];
    const double rotated_i = i * cosine - q * sine;
    const double rotated_q = i * sine + q * cosine;
    return {
        y + 0.956 * rotated_i + 0.621 * rotated_q,
        y - 0.272 * rotated_i - 0.647 * rotated_q,
        y - 1.106 * rotated_i + 1.703 * rotated_q,
    };
}

const AuroraPalette& aurora_palette_for(
    int max_iter,
    double phase,
    double vocal,
    double instrumental,
    double pitch
) {
    const PaletteBasis& basis = colour_basis_for(max_iter);
    const int palette_size = static_cast<int>(basis.cosine.size());
    aurora_palette.rgb.resize(static_cast<size_t>(palette_size));

    const double vocal_mix = std::clamp(vocal, 0.0, 1.0);
    const double instrumental_mix = std::clamp(instrumental, 0.0, 1.0);
    // Keep the original liquid-gradient equations.  Instrumental energy is
    // intentionally not a colour brightness control: it owns the zoom curve.
    // The argument remains in the ABI for compatibility and future palettes.
    (void)instrumental_mix;
    const double split = 5.0 * vocal_mix * vocal_mix;
    const double red_cos = std::cos(phase);
    const double red_sin = std::sin(phase);
    const double green_cos = std::cos(phase + split * 0.4);
    const double green_sin = std::sin(phase + split * 0.4);
    const double blue_cos = std::cos(phase + split);
    const double blue_sin = std::sin(phase + split);
    constexpr double red_gain = 140.0;
    constexpr double green_gain = 140.0;
    constexpr double blue_gain = 140.0;
    const double hue_angle = pitch_hue_angle(pitch);
    const bool rotate = std::abs(hue_angle) > 1.0e-15;
    const double hue_cos = rotate ? std::cos(hue_angle) : 1.0;
    const double hue_sin = rotate ? std::sin(hue_angle) : 0.0;

    for (int index = 0; index < palette_size; ++index) {
        const double cosine = basis.cosine[static_cast<size_t>(index)];
        const double sine = basis.sine[static_cast<size_t>(index)];
        const std::array<double, 3> original = {{
            (0.5 - 0.5 * (cosine * red_cos + sine * red_sin)) * red_gain,
            (0.5 - 0.5 * (cosine * green_cos + sine * green_sin)) * green_gain,
            (0.5 - 0.5 * (cosine * blue_cos + sine * blue_sin)) * blue_gain,
        }};
        const std::array<double, 3> colour = rotate
            ? rotate_hue_rgb(original, hue_cos, hue_sin)
            : original;
        aurora_palette.rgb[static_cast<size_t>(index)] = {{
            colour_byte(colour[0]),
            colour_byte(colour[1]),
            colour_byte(colour[2]),
        }};
    }
    return aurora_palette;
}

struct BilinearAxis {
    std::vector<int> index0;
    std::vector<int> index1;
    std::vector<float> weight;
};

void fill_bilinear_axis(
    BilinearAxis& axis,
    int source_size,
    int destination_size,
    double zoom_factor
) {
    if (source_size <= 0 || destination_size <= 0) {
        throw std::runtime_error("invalid bilinear axis dimensions");
    }
    zoom_factor = std::max(zoom_factor, 1.0);
    const double inverse_zoom = 1.0 / zoom_factor;
    const double crop_size = static_cast<double>(source_size) * inverse_zoom;
    const double left = (static_cast<double>(source_size) - crop_size) * 0.5;
    const double step = crop_size / static_cast<double>(destination_size);

    axis.index0.resize(static_cast<size_t>(destination_size));
    axis.index1.resize(static_cast<size_t>(destination_size));
    axis.weight.resize(static_cast<size_t>(destination_size));
    for (int destination = 0; destination < destination_size; ++destination) {
        double source = left
            + (static_cast<double>(destination) + 0.5) * step - 0.5;
        source = std::clamp(source, 0.0, static_cast<double>(source_size - 1));
        const int index0 = static_cast<int>(std::floor(source));
        axis.index0[static_cast<size_t>(destination)] = index0;
        axis.index1[static_cast<size_t>(destination)] =
            std::min(index0 + 1, source_size - 1);
        axis.weight[static_cast<size_t>(destination)] =
            static_cast<float>(source - static_cast<double>(index0));
    }
}

// Atlas and crop colourisation are called once per video frame.  Keep the
// coordinate maps in thread-local storage so changing the zoom reuses their
// capacity instead of allocating several vectors for every frame.  The maps
// are still fully regenerated, so this is allocation-only optimization.
struct BilinearWorkspace {
    BilinearAxis parent_x_axis;
    BilinearAxis parent_y_axis;
    BilinearAxis child_x_axis;
    BilinearAxis child_y_axis;
    std::vector<float> child_edge_x;
    std::vector<float> child_edge_y;
};

thread_local BilinearWorkspace bilinear_workspace;

inline float sample_bilinear_mapped(
    const float* source,
    int source_width,
    const BilinearAxis& x_axis,
    const BilinearAxis& y_axis,
    int x,
    int y
) {
    const int x_index = x_axis.index0[static_cast<size_t>(x)];
    const int x_next = x_axis.index1[static_cast<size_t>(x)];
    const float x_weight = x_axis.weight[static_cast<size_t>(x)];
    const int y_index = y_axis.index0[static_cast<size_t>(y)];
    const int y_next = y_axis.index1[static_cast<size_t>(y)];
    const float y_weight = y_axis.weight[static_cast<size_t>(y)];
    const float* top = source + static_cast<size_t>(y_index) * source_width;
    const float* bottom = source + static_cast<size_t>(y_next) * source_width;
    const float top_value = top[x_index] * (1.0F - x_weight)
        + top[x_next] * x_weight;
    const float bottom_value = bottom[x_index] * (1.0F - x_weight)
        + bottom[x_next] * x_weight;
    return top_value * (1.0F - y_weight) + bottom_value * y_weight;
}

// An iteration cap is an interior sentinel, not a colourable scalar.  A
// normal bilinear average can turn one interior corner and three escaping
// corners into an ordinary-looking iteration count, which leaks the tile
// boundary into the final image.  Keep a conservative coverage mask beside
// the interpolation and only average exterior samples.
inline float sample_bilinear_mapped_preserving_interior(
    const float* source,
    int source_width,
    const BilinearAxis& x_axis,
    const BilinearAxis& y_axis,
    int x,
    int y,
    int source_max_iter,
    bool& inside
) {
    const int x_index = x_axis.index0[static_cast<size_t>(x)];
    const int x_next = x_axis.index1[static_cast<size_t>(x)];
    const double x_weight = static_cast<double>(
        x_axis.weight[static_cast<size_t>(x)]);
    const int y_index = y_axis.index0[static_cast<size_t>(y)];
    const int y_next = y_axis.index1[static_cast<size_t>(y)];
    const double y_weight = static_cast<double>(
        y_axis.weight[static_cast<size_t>(y)]);
    const float* top = source + static_cast<size_t>(y_index) * source_width;
    const float* bottom = source + static_cast<size_t>(y_next) * source_width;
    const float values[4] = {
        top[x_index],
        top[x_next],
        bottom[x_index],
        bottom[x_next],
    };
    const double weights[4] = {
        (1.0 - y_weight) * (1.0 - x_weight),
        (1.0 - y_weight) * x_weight,
        y_weight * (1.0 - x_weight),
        y_weight * x_weight,
    };
    double interior_weight = 0.0;
    double exterior_weight = 0.0;
    double exterior_value = 0.0;
    for (int index = 0; index < 4; ++index) {
        const float value = values[index];
        const double weight = weights[index];
        if (!std::isfinite(value)
            || value >= static_cast<float>(source_max_iter) - 0.5F) {
            interior_weight += weight;
        } else {
            exterior_weight += weight;
            exterior_value += static_cast<double>(value) * weight;
        }
    }
    if (interior_weight >= 0.5 || exterior_weight <= 1.0e-12) {
        inside = true;
        return static_cast<float>(source_max_iter);
    }
    inside = false;
    return static_cast<float>(exterior_value / exterior_weight);
}

inline void write_colour_pixel(
    float smooth,
    int source_max_iter,
    const AuroraPalette& palette,
    float palette_index_scale,
    std::uint8_t* destination
) {
    if (!std::isfinite(smooth)
        || smooth >= static_cast<float>(source_max_iter) - 0.5F) {
        destination[0] = 0;
        destination[1] = 0;
        destination[2] = 0;
        return;
    }
    const int palette_size = static_cast<int>(palette.rgb.size());
    const double scaled_index = static_cast<double>(smooth)
        * static_cast<double>(palette_index_scale);
    const int palette_index = scaled_index >= static_cast<double>(palette_size - 1)
        ? palette_size - 1
        : !std::isfinite(scaled_index) || scaled_index <= 0.0
            ? 0
            : static_cast<int>(scaled_index);
    const auto& colour = palette.rgb[static_cast<size_t>(palette_index)];
    destination[0] = colour[0];
    destination[1] = colour[1];
    destination[2] = colour[2];
}

long double parse_zoom(const char* text) {
    if (!valid_c_string(text)) throw std::runtime_error("native zoom text is too long or null");
    errno = 0;
    char* end = nullptr;
    const long double zoom = std::strtold(text, &end);
    if (end == text || *end != '\0' || errno == ERANGE
        || !std::isfinite(zoom) || zoom <= 0.0L) {
        throw std::runtime_error("invalid Mandelbrot zoom");
    }
    const long double log10_zoom = std::log10(zoom);
    if (!std::isfinite(log10_zoom)
        || log10_zoom < MIN_NATIVE_LOG10_ZOOM
        || log10_zoom > MAX_NATIVE_LOG10_ZOOM) {
        throw std::runtime_error("Mandelbrot zoom is outside the supported range");
    }
    return zoom;
}

inline int saturating_exponent(long long value) noexcept {
    if (value > static_cast<long long>(std::numeric_limits<int>::max())) {
        return std::numeric_limits<int>::max();
    }
    if (value < static_cast<long long>(std::numeric_limits<int>::min())) {
        return std::numeric_limits<int>::min();
    }
    return static_cast<int>(value);
}

// A normalized mantissa/exponent number.  The mantissa keeps ordinary CPU
// arithmetic while the exponent keeps tiny perturbations representable far
// beyond the range of double or long double.  This is the same numerical
// split used by deep-zoom renderers: the reference orbit is compact, while
// per-pixel deltas can reach 1e-4000 without becoming zero.
struct FloatExp {
    double mantissa = 0.0;
    int exponent = 0; // value = mantissa * 2^exponent

    static FloatExp from_parts(double value, int exponent) {
        if (value == 0.0 || std::isnan(value)) return {value, 0};
        int shift = 0;
        const double normalized = std::frexp(value, &shift);
        return {
            normalized,
            saturating_exponent(
                static_cast<long long>(exponent) + static_cast<long long>(shift)),
        };
    }

    static FloatExp from_long_double(long double value) {
        if (value == 0.0L || std::isnan(value)) return {static_cast<double>(value), 0};
        int exponent = 0;
        const long double mantissa = std::frexp(value, &exponent);
        return from_parts(static_cast<double>(mantissa), exponent);
    }

#ifdef FRACTAL_HAVE_MPFR
    static FloatExp from_mpfr(const mpfr_t value) {
        signed long exponent = 0;
        const double mantissa = mpfr_get_d_2exp(&exponent, value, MPFR_RNDN);
        if (!std::isfinite(mantissa)) return {mantissa, 0};
        if (exponent > std::numeric_limits<int>::max()) {
            return {std::copysign(std::numeric_limits<double>::infinity(), mantissa), 0};
        }
        if (exponent < std::numeric_limits<int>::min()) return {0.0L, 0};
        return from_parts(mantissa, static_cast<int>(exponent));
    }
#endif

    long double as_long_double() const {
        if (mantissa == 0.0) return 0.0L;
        if (exponent > std::numeric_limits<int>::max() / 2) {
            return std::copysign(std::numeric_limits<long double>::infinity(), mantissa);
        }
        return std::ldexp(static_cast<long double>(mantissa), exponent);
    }

    bool finite() const { return std::isfinite(mantissa); }
    bool zero() const { return mantissa == 0.0; }
};

// A complex value represented with one shared binary exponent.  The two
// components of a perturbation normally have comparable magnitudes, so a
// shared exponent avoids normalizing real and imaginary parts separately in
// every complex multiply.  It remains valid far beyond double's exponent
// range while keeping the hot BLA arithmetic in ordinary doubles.
struct ScaledComplex {
    double real = 0.0;
    double imag = 0.0;
    int exponent = 0;

    void normalize() {
        const double magnitude = std::max(std::abs(real), std::abs(imag));
        if (magnitude == 0.0) {
            real = 0.0;
            imag = 0.0;
            exponent = 0;
            return;
        }
        // Products and sums of normalized values overwhelmingly land within
        // one binary shift of the target interval. Handle that common case
        // without a libm frexp/ldexp pair; only severe cancellation needs the
        // general fallback.
        if (magnitude >= 1.0) {
            real *= 0.5;
            imag *= 0.5;
            exponent = saturating_exponent(static_cast<long long>(exponent) + 1);
            return;
        }
        if (magnitude >= 0.5) return;
        if (magnitude >= 0.25) {
            real *= 2.0;
            imag *= 2.0;
            exponent = saturating_exponent(static_cast<long long>(exponent) - 1);
            return;
        }
        int shift = 0;
        (void)std::frexp(magnitude, &shift);
        real = std::ldexp(real, -shift);
        imag = std::ldexp(imag, -shift);
        exponent = saturating_exponent(
            static_cast<long long>(exponent) + static_cast<long long>(shift));
    }

    static ScaledComplex from_float_exp(const FloatExp& real_part, const FloatExp& imag_part) {
        if (real_part.zero() && imag_part.zero()) return {};
        const int common_exponent = std::max(real_part.exponent, imag_part.exponent);
        const long long real_difference = static_cast<long long>(real_part.exponent)
            - static_cast<long long>(common_exponent);
        const long long imag_difference = static_cast<long long>(imag_part.exponent)
            - static_cast<long long>(common_exponent);
        ScaledComplex result{
            real_difference < -1074
                ? 0.0 : std::ldexp(real_part.mantissa, static_cast<int>(real_difference)),
            imag_difference < -1074
                ? 0.0 : std::ldexp(imag_part.mantissa, static_cast<int>(imag_difference)),
            common_exponent,
        };
        result.normalize();
        return result;
    }
};

inline ScaledComplex sc_add(const ScaledComplex& a, const ScaledComplex& b) {
    if (a.real == 0.0 && a.imag == 0.0) return b;
    if (b.real == 0.0 && b.imag == 0.0) return a;
    const ScaledComplex* larger = &a;
    const ScaledComplex* smaller = &b;
    if (b.exponent > a.exponent) {
        larger = &b;
        smaller = &a;
    }
    const long long exponent_difference =
        static_cast<long long>(larger->exponent)
        - static_cast<long long>(smaller->exponent);
    if (exponent_difference > 60) return *larger;
    const int difference = static_cast<int>(exponent_difference);
    ScaledComplex result{
        larger->real + std::ldexp(smaller->real, -difference),
        larger->imag + std::ldexp(smaller->imag, -difference),
        larger->exponent,
    };
    result.normalize();
    return result;
}

inline ScaledComplex sc_neg(const ScaledComplex& value) {
    return {-value.real, -value.imag, value.exponent};
}

inline ScaledComplex sc_sub(const ScaledComplex& a, const ScaledComplex& b) {
    return sc_add(a, sc_neg(b));
}

inline ScaledComplex sc_mul(const ScaledComplex& a, const ScaledComplex& b) {
    if ((a.real == 0.0 && a.imag == 0.0) || (b.real == 0.0 && b.imag == 0.0)) return {};
    ScaledComplex result{
        a.real * b.real - a.imag * b.imag,
        a.real * b.imag + a.imag * b.real,
        saturating_exponent(
            static_cast<long long>(a.exponent) + static_cast<long long>(b.exponent)),
    };
    result.normalize();
    return result;
}

inline ScaledComplex sc_double(const ScaledComplex& value) {
    if (value.real == 0.0 && value.imag == 0.0) return {};
    // Multiplication by two is exact in this representation and does not
    // need another mantissa normalization.
    return {
        value.real,
        value.imag,
        saturating_exponent(static_cast<long long>(value.exponent) + 1),
    };
}

inline bool sc_finite(const ScaledComplex& value) noexcept {
    return std::isfinite(value.real) && std::isfinite(value.imag);
}

struct ScaledNorm {
    double mantissa = 0.0;
    int exponent = 0;
};

inline ScaledNorm sc_norm_squared(const ScaledComplex& value) {
    if (!sc_finite(value)) {
        return {std::numeric_limits<double>::infinity(),
                std::numeric_limits<int>::max()};
    }
    const double norm = value.real * value.real + value.imag * value.imag;
    if (norm == 0.0) return {};
    if (!std::isfinite(norm)) {
        return {std::numeric_limits<double>::infinity(),
                std::numeric_limits<int>::max()};
    }
    const long long squared_exponent = 2LL * static_cast<long long>(value.exponent);
    if (norm >= 1.0) {
        const long long exponent = squared_exponent + 1;
        if (exponent > std::numeric_limits<int>::max()) {
            return {std::numeric_limits<double>::infinity(),
                    std::numeric_limits<int>::max()};
        }
        return {norm * 0.5, static_cast<int>(exponent)};
    }
    if (norm < 0.5) {
        const long long exponent = squared_exponent - 1;
        if (exponent < std::numeric_limits<int>::min()) return {};
        return {norm * 2.0, static_cast<int>(exponent)};
    }
    if (squared_exponent > std::numeric_limits<int>::max()) {
        return {std::numeric_limits<double>::infinity(),
                std::numeric_limits<int>::max()};
    }
    if (squared_exponent < std::numeric_limits<int>::min()) return {};
    return {norm, static_cast<int>(squared_exponent)};
}

inline int sc_compare_norm(const ScaledNorm& a, const ScaledNorm& b) {
    if (!std::isfinite(a.mantissa)) return std::isfinite(b.mantissa) ? 1 : 0;
    if (!std::isfinite(b.mantissa)) return -1;
    if (a.mantissa == 0.0 && b.mantissa == 0.0) return 0;
    if (a.mantissa == 0.0) return -1;
    if (b.mantissa == 0.0) return a.mantissa > 0.0 ? 1 : -1;
    if (a.exponent != b.exponent) return a.exponent < b.exponent ? -1 : 1;
    return (a.mantissa > b.mantissa) - (a.mantissa < b.mantissa);
}

inline bool sc_outside_escape(const ScaledNorm& norm) {
    // 4 == 0.5 * 2^3 in the normalized representation.
    return sc_compare_norm(norm, ScaledNorm{0.5, 3}) > 0;
}

inline double sc_to_double(const ScaledComplex& value) {
    return std::ldexp(value.real, value.exponent);
}

inline float smooth_escape_scaled(int iteration, const ScaledNorm& norm) {
    if (norm.mantissa == 0.0) return static_cast<float>(iteration);
    const long double log_magnitude = 0.5L * (
        std::log(static_cast<long double>(norm.mantissa))
        + static_cast<long double>(norm.exponent) * LOG_TWO);
    if (!(log_magnitude > 0.0L) || !std::isfinite(log_magnitude)) {
        return static_cast<float>(iteration);
    }
    return static_cast<float>(static_cast<long double>(iteration)
        - std::log(log_magnitude) / LOG_TWO);
}

inline FloatExp fe_neg(const FloatExp& value) {
    return {-value.mantissa, value.exponent};
}

inline FloatExp fe_abs(const FloatExp& value) {
    return {std::abs(value.mantissa), value.exponent};
}

inline FloatExp fe_add(const FloatExp& a, const FloatExp& b) {
    if (a.zero()) return b;
    if (b.zero()) return a;
    const FloatExp* larger = &a;
    const FloatExp* smaller = &b;
    if (b.exponent > a.exponent) {
        larger = &b;
        smaller = &a;
    }
    const long long exponent_difference = static_cast<long long>(larger->exponent)
        - static_cast<long long>(smaller->exponent);
    if (exponent_difference > 60) return *larger;
    const int difference = static_cast<int>(exponent_difference);
    const double mantissa = larger->mantissa
        + std::ldexp(smaller->mantissa, -difference);
    if (mantissa == 0.0 || !std::isfinite(mantissa)) {
        return FloatExp::from_parts(mantissa, larger->exponent);
    }

    // Both inputs are normalized, so their aligned sum is in (-2, 2).
    // Most additions need at most one multiply by two.  Falling back to
    // frexp is only needed for cancellation, which is much less frequent
    // than the old unconditional normalization in the inner pixel loop.
    const double magnitude = std::abs(mantissa);
    if (magnitude >= 1.0) {
        return {
            mantissa * 0.5,
            saturating_exponent(static_cast<long long>(larger->exponent) + 1),
        };
    }
    if (magnitude >= 0.5) {
        return {mantissa, larger->exponent};
    }
    int shift = 0;
    return {
        std::frexp(mantissa, &shift),
        saturating_exponent(
            static_cast<long long>(larger->exponent) + static_cast<long long>(shift)),
    };
}

inline FloatExp fe_sub(const FloatExp& a, const FloatExp& b) {
    return fe_add(a, fe_neg(b));
}

inline FloatExp fe_mul(const FloatExp& a, const FloatExp& b) {
    if (a.zero() || b.zero()) return {0.0, 0};
    const double mantissa = a.mantissa * b.mantissa;
    if (!std::isfinite(mantissa)) return FloatExp::from_parts(mantissa, 0);

    // The product of two normalized mantissas lies in [0.25, 1), apart from
    // a possible rounding hit at 1.  Normalize it without calling frexp.
    const double magnitude = std::abs(mantissa);
    const long long exponent_sum = static_cast<long long>(a.exponent)
        + static_cast<long long>(b.exponent);
    if (magnitude >= 1.0) {
        return {
            mantissa * 0.5,
            saturating_exponent(exponent_sum + 1),
        };
    }
    if (magnitude < 0.5) {
        return {
            mantissa * 2.0,
            saturating_exponent(exponent_sum - 1),
        };
    }
    return {mantissa, saturating_exponent(exponent_sum)};
}

inline FloatExp fe_mul(const FloatExp& a, double b) {
    return fe_mul(a, FloatExp::from_parts(static_cast<long double>(b), 0));
}

inline FloatExp fe_div(const FloatExp& a, const FloatExp& b) {
    if (a.zero()) return {0.0, 0};
    const double mantissa = a.mantissa / b.mantissa;
    if (!std::isfinite(mantissa)) return FloatExp::from_parts(mantissa, 0);
    const double magnitude = std::abs(mantissa);
    const long long exponent_difference = static_cast<long long>(a.exponent)
        - static_cast<long long>(b.exponent);
    if (magnitude >= 1.0) {
        return {
            mantissa * 0.5,
            saturating_exponent(exponent_difference + 1),
        };
    }
    if (magnitude < 0.5) {
        return {
            mantissa * 2.0,
            saturating_exponent(exponent_difference - 1),
        };
    }
    return {mantissa, saturating_exponent(exponent_difference)};
}

inline FloatExp fe_sqr(const FloatExp& value) {
    return fe_mul(value, value);
}

inline FloatExp fe_sqrt(const FloatExp& value) {
    if (value.zero()) return value;
    long long exponent = value.exponent;
    double mantissa = value.mantissa;
    if (mantissa < 0.0 || !std::isfinite(mantissa)) {
        return FloatExp::from_parts(std::sqrt(mantissa), 0);
    }
    if (exponent % 2 != 0) {
        mantissa *= 2.0;
        --exponent;
        // sqrt(2 * normalized_mantissa) is in [1, sqrt(2)); normalize the
        // result directly rather than sending it through frexp.
        return {
            std::sqrt(mantissa) * 0.5,
            saturating_exponent(exponent / 2 + 1),
        };
    }
    return {std::sqrt(mantissa), saturating_exponent(exponent / 2)};
}

inline int fe_compare(const FloatExp& a, const FloatExp& b) {
    // Zero is represented canonically with exponent zero, but its exponent
    // is not a numeric magnitude. Handle it before comparing exponents;
    // otherwise 0 would compare larger than tiny positive values and a
    // zero-radius BLA map would be accepted as if it had radius one.
    if (!a.finite()) return b.finite() ? 1 : 0;
    if (!b.finite()) return -1;
    if (a.mantissa == 0.0) return b.mantissa == 0.0 ? 0 : -1;
    if (b.mantissa == 0.0) return a.mantissa > 0.0 ? 1 : -1;
    if (a.mantissa < 0.0 && b.mantissa >= 0.0) return -1;
    if (a.mantissa >= 0.0 && b.mantissa < 0.0) return 1;
    const bool negative = a.mantissa < 0.0;
    if (a.exponent != b.exponent) {
        const int result = a.exponent < b.exponent ? -1 : 1;
        return negative ? -result : result;
    }
    // Mantissas retain their sign.  The numeric ordering of two negative
    // values at the same exponent is therefore already the direct mantissa
    // ordering; negating it reverses -0.75 and -0.5 and corrupts every
    // radius/escape comparison involving a negative FloatExp value.
    return (a.mantissa > b.mantissa) - (a.mantissa < b.mantissa);
}

inline FloatExp fe_norm_squared(const FloatExp& real, const FloatExp& imag) {
    return fe_add(fe_sqr(real), fe_sqr(imag));
}

inline long double fe_log(const FloatExp& value) {
    if (value.mantissa == 0.0) return -std::numeric_limits<long double>::infinity();
    return std::log(std::abs(static_cast<long double>(value.mantissa)))
        + static_cast<long double>(value.exponent) * LOG_TWO;
}

struct FloatExpComplex {
    FloatExp real;
    FloatExp imag;
};

inline FloatExpComplex fec_add(const FloatExpComplex& a, const FloatExpComplex& b) {
    return {fe_add(a.real, b.real), fe_add(a.imag, b.imag)};
}

inline FloatExpComplex fec_neg(const FloatExpComplex& value) {
    return {fe_neg(value.real), fe_neg(value.imag)};
}

inline FloatExpComplex fec_sub(const FloatExpComplex& a, const FloatExpComplex& b) {
    return fec_add(a, fec_neg(b));
}

inline FloatExpComplex fec_mul(const FloatExpComplex& a, const FloatExpComplex& b) {
    return {
        fe_sub(fe_mul(a.real, b.real), fe_mul(a.imag, b.imag)),
        fe_add(fe_mul(a.real, b.imag), fe_mul(a.imag, b.real)),
    };
}

inline FloatExpComplex fec_mul(const FloatExpComplex& a, const FloatExp& b) {
    return {fe_mul(a.real, b), fe_mul(a.imag, b)};
}

inline FloatExp fec_norm_squared(const FloatExpComplex& value) {
    return fe_norm_squared(value.real, value.imag);
}

inline FloatExp fec_escape_margin_with_delta(
    const FloatExpComplex& reference,
    const FloatExpComplex& delta
) {
    const FloatExp cross = fe_mul(
        fe_add(
            fe_mul(reference.real, delta.real),
            fe_mul(reference.imag, delta.imag)),
        2.0);
    const FloatExp reference_margin = fe_sub(
        fec_norm_squared(reference),
        FloatExp::from_parts(4.0, 0));
    return fe_add(
        fe_add(reference_margin, cross),
        fec_norm_squared(delta));
}

inline FloatExp sc_component_as_float_exp(
    const ScaledComplex& value,
    bool imaginary
) {
    return FloatExp::from_parts(
        imaginary ? value.imag : value.real,
        value.exponent);
}

inline FloatExp sc_escape_margin_with_delta(
    const ScaledComplex& reference,
    const ScaledComplex& delta
) {
    // Adding a tiny delta to an O(1) reference is intentionally lossy in the
    // compact ScaledComplex representation: the fast aligned sum discards a
    // term more than 60 binary exponents below the reference. That is safe
    // for ordinary orbit values, but not for a reference sitting exactly on
    // |z|=2 (for example the Mandelbrot -2 tip), where the sign of the tiny
    // perturbation is the escape decision itself. Compute the signed margin
    // around the escape radius before adding it back to 4, so cancellation
    // at the boundary remains representable.
    const FloatExpComplex reference_parts{
        sc_component_as_float_exp(reference, false),
        sc_component_as_float_exp(reference, true),
    };
    const FloatExpComplex delta_parts{
        sc_component_as_float_exp(delta, false),
        sc_component_as_float_exp(delta, true),
    };
    return fec_escape_margin_with_delta(reference_parts, delta_parts);
}

inline ScaledNorm sc_norm_squared_with_delta(
    const ScaledComplex& reference,
    const ScaledComplex& delta
) {
    const FloatExp norm = fe_add(
        FloatExp::from_parts(4.0, 0),
        sc_escape_margin_with_delta(reference, delta));
    return {norm.mantissa, norm.exponent};
}

inline bool sc_outside_escape_with_delta(
    const ScaledComplex& reference,
    const ScaledComplex& delta
) {
    const FloatExp margin = sc_escape_margin_with_delta(reference, delta);
    return fe_compare(
        margin,
        FloatExp{0.0, 0}) > 0;
}

struct BlaStep {
    FloatExpComplex A;
    FloatExpComplex B;
    // Retaining terms through degree three makes this a local series
    // approximation rather than a purely linear BLA map.  The variables are
    // the incoming perturbation d and the fixed parameter offset c:
    // A*d + B*c + C*d² + D*d*c + E*c² + F*d³ + G*d²*c + H*d*c² + I*c³.
    FloatExpComplex C;
    FloatExpComplex D;
    FloatExpComplex E;
    FloatExpComplex F;
    FloatExpComplex G;
    FloatExpComplex H;
    FloatExpComplex I;
    FloatExp radius_squared;
    int length = 1;
};

struct FastBlaStep {
    std::array<ScaledComplex, 9> coefficients{};
    FloatExp radius_squared;
    int length = 1;
};

struct LinearBlaStep {
    ScaledComplex A;
    ScaledComplex B;
    FloatExp radius_squared;
    int length = 1;
};

struct LinearBlaBuilderStep {
    FloatExpComplex A;
    FloatExpComplex B;
    FloatExp radius_squared;
    int length = 1;
};

// Image-wide parameter series.  Unlike the local bivariate BLA map, this is
// a polynomial in dc that starts every pixel at one shared, validated
// iteration of the reference orbit.  It is the main Kalles-style shortcut:
// the expensive early orbit is evaluated once as coefficients, then each
// pixel enters the ordinary perturbation/BLA loop at series_iteration.
struct ImageSeries {
    bool enabled = false;
    int order = 0;
    int iteration = 1;
    FloatExp radius_squared{0.0, 0};
    std::vector<ScaledComplex> coefficients;
};

inline ScaledComplex evaluate_image_series(
    const ImageSeries& series,
    const ScaledComplex& dc
) {
    if (!series.enabled || series.coefficients.size() <= 1) return {};
    ScaledComplex result = series.coefficients.back();
    for (size_t index = series.coefficients.size() - 1; index > 1; --index) {
        result = sc_add(
            sc_mul(result, dc),
            series.coefficients[index - 1]);
    }
    return sc_mul(result, dc);
}

inline bool fec_finite(const FloatExpComplex& value) noexcept {
    return value.real.finite() && value.imag.finite();
}

FastBlaStep compact_bla_step(const BlaStep& step) {
    const bool finite = step.radius_squared.finite()
        && fec_finite(step.A)
        && fec_finite(step.B)
        && fec_finite(step.C)
        && fec_finite(step.D)
        && fec_finite(step.E)
        && fec_finite(step.F)
        && fec_finite(step.G)
        && fec_finite(step.H)
        && fec_finite(step.I);
    FastBlaStep result{
        {
        ScaledComplex::from_float_exp(step.A.real, step.A.imag),
        ScaledComplex::from_float_exp(step.B.real, step.B.imag),
        ScaledComplex::from_float_exp(step.C.real, step.C.imag),
        ScaledComplex::from_float_exp(step.D.real, step.D.imag),
        ScaledComplex::from_float_exp(step.E.real, step.E.imag),
        ScaledComplex::from_float_exp(step.F.real, step.F.imag),
        ScaledComplex::from_float_exp(step.G.real, step.G.imag),
        ScaledComplex::from_float_exp(step.H.real, step.H.imag),
        ScaledComplex::from_float_exp(step.I.real, step.I.imag),
        },
        finite ? step.radius_squared : FloatExp{0.0, 0},
        step.length,
    };
    return result;
}

LinearBlaStep compact_linear_bla_step(const LinearBlaBuilderStep& step) {
    const bool finite = step.radius_squared.finite()
        && fec_finite(step.A)
        && fec_finite(step.B);
    return {
        ScaledComplex::from_float_exp(step.A.real, step.A.imag),
        ScaledComplex::from_float_exp(step.B.real, step.B.imag),
        finite ? step.radius_squared : FloatExp{0.0, 0},
        step.length,
    };
}

inline ScaledComplex apply_bla_series(
    const FastBlaStep& step,
    const ScaledComplex& delta,
    const ScaledComplex& parameter,
    int order
) {
    const auto& coefficient = step.coefficients;
    if (order <= 1) {
        return sc_add(sc_mul(coefficient[0], delta), sc_mul(coefficient[1], parameter));
    }

    if (order == 2) {
        // P(d,c) = d*(A + d*C + c*D) + c*(B + c*E).
        const ScaledComplex d_inner = sc_add(
            sc_mul(coefficient[2], delta),
            sc_mul(coefficient[3], parameter));
        const ScaledComplex c_inner = sc_add(
            coefficient[1],
            sc_mul(coefficient[4], parameter));
        return sc_add(
            sc_mul(delta, sc_add(coefficient[0], d_inner)),
            sc_mul(parameter, c_inner));
    }

    // Cubic Horner form cuts the hot polynomial from sixteen complex
    // products to nine while preserving the same bivariate coefficients:
    // P(d,c) = d*(A + d*(C + F*d + G*c) + c*(D + H*c))
    //        + c*(B + c*(E + I*c)).
    const ScaledComplex d_cubic = sc_add(
        sc_add(sc_mul(coefficient[5], delta), sc_mul(coefficient[6], parameter)),
        coefficient[2]);
    const ScaledComplex c_cubic = sc_add(
        coefficient[3],
        sc_mul(coefficient[7], parameter));
    const ScaledComplex d_inner = sc_add(
        coefficient[0],
        sc_add(sc_mul(delta, d_cubic), sc_mul(parameter, c_cubic)));
    const ScaledComplex c_inner = sc_add(
        coefficient[1],
        sc_mul(parameter, sc_add(
            coefficient[4],
            sc_mul(coefficient[8], parameter))));
    return sc_add(
        sc_mul(delta, d_inner),
        sc_mul(parameter, c_inner));
}

struct BlaLevels {
    std::vector<std::vector<FastBlaStep>> levels;
    std::vector<std::vector<LinearBlaStep>> linear_levels;
    std::vector<std::vector<LinearBlaStep>> deep_linear_levels;
    FloatExp input_radius;
    FloatExp deep_input_radius;
    // First reference-orbit index that a BLA map must not cross.  Once the
    // reference itself escapes, its orbit grows super-exponentially; using a
    // long map across that point can hide an escape behind cancellation in a
    // finite-precision perturbation.  Maps stop at the reference escape and
    // the ordinary perturbation loop takes over.
    int map_end = 0;

    static int highest_level_for_length(int max_length) noexcept {
        return max_length > 0
            ? 31 - __builtin_clz(static_cast<unsigned int>(max_length))
            : 0;
    }

    const FastBlaStep* lookup(
        int start,
        const FloatExp& delta_norm_squared,
        int max_length
    ) const noexcept {
        if (start <= 0 || max_length <= 0 || !delta_norm_squared.finite()) return nullptr;
        const int base_count = levels.empty() ? 0 : static_cast<int>(levels[0].size());
        if (start > base_count) return nullptr;
        int highest_level = highest_level_for_length(max_length);
        highest_level = std::min(
            highest_level,
            static_cast<int>(levels.size()) - 1);
        // Start at the largest permitted block and descend. The previous
        // low-to-high walk selected an oversized map and the caller then
        // discarded it, needlessly falling all the way back to one exact
        // iteration. A bounded descent returns the best valid ancestor.
        const int offset = start - 1;
        for (int level = highest_level; level >= 0; --level) {
            const int span_mask = (1 << level) - 1;
            if ((offset & span_mask) != 0) continue;
            const int index = offset >> level;
            if (index >= static_cast<int>(levels[level].size())) continue;
            const FastBlaStep& candidate = levels[level][static_cast<size_t>(index)];
            if (candidate.length > 1
                && candidate.radius_squared.finite()
                && fe_compare(delta_norm_squared, candidate.radius_squared) < 0) {
                return &candidate;
            }
        }
        return nullptr;
    }

    const LinearBlaStep* lookup_linear(
        int start,
        const FloatExp& delta_norm_squared,
        int max_length,
        bool deep
    ) const noexcept {
        const auto& available_levels = deep ? deep_linear_levels : linear_levels;
        if (start <= 0 || max_length <= 0 || !delta_norm_squared.finite()) return nullptr;
        const int base_count = available_levels.empty()
            ? 0 : static_cast<int>(available_levels[0].size());
        if (start > base_count) return nullptr;
        int highest_level = highest_level_for_length(max_length);
        highest_level = std::min(
            highest_level,
            static_cast<int>(available_levels.size()) - 1);
        const int offset = start - 1;
        for (int level = highest_level; level >= 0; --level) {
            const int span_mask = (1 << level) - 1;
            if ((offset & span_mask) != 0) continue;
            const int index = offset >> level;
            if (index >= static_cast<int>(available_levels[level].size())) continue;
            const LinearBlaStep& candidate = available_levels[level][static_cast<size_t>(index)];
            if (candidate.length > 1
                && candidate.radius_squared.finite()
                && fe_compare(delta_norm_squared, candidate.radius_squared) < 0) {
                return &candidate;
            }
        }
        return nullptr;
    }

};

struct ReferenceOrbitData {
    std::vector<ScaledComplex> scaled;
    std::vector<double> real_double;
    std::vector<double> imag_double;
};

struct ReferenceContext {
    std::vector<FloatExpComplex> fast_orbit;
    std::shared_ptr<const ReferenceOrbitData> orbit;
    BlaLevels bla;
    ImageSeries image_series;
    int requested_max_iter = 0;
    int requested_series_order = 8;
    std::uint64_t reference_build_ns = 0;
    std::uint64_t series_build_ns = 0;
    std::uint64_t bla_build_ns = 0;
    long double x_center = 0.0L;
    long double y_center = 0.0L;
#ifdef FRACTAL_HAVE_MPFR
    mpfr_prec_t precision_bits = 0;
#endif
};

// Opaque C-ABI handles must not be blindly cast and dereferenced.  In
// addition to turning accidental double-destroys into a safe diagnostic, the
// registry gives render/clone calls a shared ownership hold while a concurrent
// destroy removes the public handle. The map owns every context returned to
// an external caller; callers never own a raw C++ allocation directly.
//
// Handles are monotonically increasing opaque tokens rather than the address
// of the context. Reusing a freed context address could otherwise make a
// stale handle accidentally refer to a later render's context (an ABA bug).
std::mutex reference_registry_mutex;
std::unordered_map<std::uintptr_t, std::shared_ptr<ReferenceContext>> reference_registry;
std::uintptr_t next_reference_handle = 0x1000U;

void* register_reference(std::unique_ptr<ReferenceContext> context) {
    if (!context) throw std::runtime_error("cannot register an empty reference");
    auto shared = std::shared_ptr<ReferenceContext>(std::move(context));
    std::lock_guard<std::mutex> lock(reference_registry_mutex);
    for (;;) {
        const std::uintptr_t token = next_reference_handle++;
        if (token == 0 || reference_registry.find(token) != reference_registry.end()) {
            continue;
        }
        reference_registry.emplace(token, std::move(shared));
        return reinterpret_cast<void*>(token);
    }
}

std::shared_ptr<ReferenceContext> acquire_reference(void* handle) {
    if (!handle) return {};
    const auto key = reinterpret_cast<std::uintptr_t>(handle);
    std::lock_guard<std::mutex> lock(reference_registry_mutex);
    const auto found = reference_registry.find(key);
    return found == reference_registry.end() ? nullptr : found->second;
}

std::shared_ptr<ReferenceContext> remove_reference(void* handle) {
    if (!handle) return {};
    const auto key = reinterpret_cast<std::uintptr_t>(handle);
    std::lock_guard<std::mutex> lock(reference_registry_mutex);
    const auto found = reference_registry.find(key);
    if (found == reference_registry.end()) return {};
    auto context = std::move(found->second);
    reference_registry.erase(found);
    return context;
}

#if defined(__AVX2__)
void render_direct_avx2(
    float* output,
    int width,
    int height,
    const std::vector<double>& x_coordinates,
    const std::vector<double>& y_coordinates,
    int max_iter,
    int threads
) {
#ifdef _OPENMP
    if (threads > 0) {
        omp_set_dynamic(0);
        omp_set_num_threads(threads);
    }
#pragma omp parallel for schedule(dynamic, 1)
#endif
    for (int py = 0; py < height; ++py) {
        const double cy_scalar = y_coordinates[static_cast<size_t>(py)];
        int px = 0;
        for (; px + 3 < width; px += 4) {
            double cx_values[4];
            std::copy_n(
                x_coordinates.data() + px,
                4,
                cx_values);
            int active_bits = 0;
            for (int lane = 0; lane < 4; ++lane) {
                const double cx = cx_values[lane];
                const double q = (cx - 0.25) * (cx - 0.25) + cy_scalar * cy_scalar;
                const bool in_cardioid = q * (q + cx - 0.25)
                    <= 0.25 * cy_scalar * cy_scalar;
                const bool in_bulb = (cx + 1.0) * (cx + 1.0)
                    + cy_scalar * cy_scalar <= 0.0625;
                if (!in_cardioid && !in_bulb) active_bits |= 1 << lane;
                else output[py * width + px + lane] = static_cast<float>(max_iter);
            }
            if (active_bits == 0) continue;

            __m256d zr = _mm256_setzero_pd();
            __m256d zi = _mm256_setzero_pd();
            const __m256d cx = _mm256_loadu_pd(cx_values);
            const __m256d cy = _mm256_set1_pd(cy_scalar);
            int escaped_iteration[4] = {
                max_iter + 1,
                max_iter + 1,
                max_iter + 1,
                max_iter + 1,
            };
            double escaped_norm[4] = {0.0, 0.0, 0.0, 0.0};
            for (int iteration = 0; iteration < max_iter && active_bits != 0; ++iteration) {
                const __m256d zr_squared = _mm256_mul_pd(zr, zr);
                const __m256d zi_squared = _mm256_mul_pd(zi, zi);
                const __m256d next_real = _mm256_add_pd(
                    _mm256_sub_pd(zr_squared, zi_squared),
                    cx);
                const __m256d next_imag = _mm256_add_pd(
                    _mm256_mul_pd(_mm256_add_pd(zr, zr), zi),
                    cy);
                zr = next_real;
                zi = next_imag;
                const __m256d norm = _mm256_add_pd(
                    _mm256_mul_pd(zr, zr),
                    _mm256_mul_pd(zi, zi));
                alignas(32) double norm_values[4];
                _mm256_store_pd(norm_values, norm);
                int escaped = 0;
                for (int lane = 0; lane < 4; ++lane) {
                    if ((active_bits & (1 << lane)) && norm_values[lane] > 4.0) {
                        escaped_iteration[lane] = iteration + 1;
                        escaped_norm[lane] = norm_values[lane];
                        escaped |= 1 << lane;
                    }
                }
                active_bits &= ~escaped;
            }
            for (int lane = 0; lane < 4; ++lane) {
                const int index = py * width + px + lane;
                if (escaped_iteration[lane] > max_iter) {
                    output[index] = static_cast<float>(max_iter);
                    continue;
                }
                const double magnitude = std::sqrt(std::max(escaped_norm[lane], 4.0000001));
                output[index] = static_cast<float>(
                    static_cast<double>(escaped_iteration[lane])
                    - std::log(std::log(magnitude)) / static_cast<double>(LOG_TWO));
            }
        }
        // Scalar cleanup handles a non-multiple-of-four width without a
        // masked load/store penalty in the common SIMD path.
        for (; px < width; ++px) {
            const double cx = x_coordinates[static_cast<size_t>(px)];
            const int index = py * width + px;
            double zr = 0.0;
            double zi = 0.0;
            int iteration = 0;
            for (; iteration < max_iter; ++iteration) {
                const double next_real = zr * zr - zi * zi + cx;
                const double next_imag = 2.0 * zr * zi + cy_scalar;
                zr = next_real;
                zi = next_imag;
                const double magnitude_squared = zr * zr + zi * zi;
                if (magnitude_squared > ESCAPE_RADIUS_SQUARED) {
                    const double magnitude = std::sqrt(std::max(magnitude_squared, 4.0000001));
                    output[index] = static_cast<float>(static_cast<double>(iteration + 1)
                        - std::log(std::log(magnitude)) / static_cast<double>(LOG_TWO));
                    break;
                }
            }
            if (iteration == max_iter) output[index] = static_cast<float>(max_iter);
        }
    }
}
#endif

void render_direct(
    float* output,
    int width,
    int height,
    long double zoom,
    long double x_center,
    long double y_center,
    int max_iter,
    int threads,
    int backend = 0,
    int formula = FRACTAL_FORMULA_MANDELBROT,
    double julia_real = 0.0,
    double julia_imag = 0.0
) {
    // This path is deliberately ordinary double precision.  The Python
    // layer routes only shallow views here; using long double for every
    // pixel made the inexpensive part of a zoom sequence disproportionately
    // slow on low-power CPUs.
    const double zoom_value = static_cast<double>(zoom);
    const double center_real = static_cast<double>(x_center);
    const double center_imag = static_cast<double>(y_center);
    if (!std::isfinite(zoom_value) || zoom_value <= 0.0
        || !std::isfinite(center_real) || !std::isfinite(center_imag)) {
        throw std::runtime_error("direct native coordinates or zoom exceed double range");
    }
    const double height_span = 2.8 / zoom_value;
    const double width_span = height_span * static_cast<double>(width) / static_cast<double>(height);
    if (!std::isfinite(height_span) || !std::isfinite(width_span)) {
        throw std::runtime_error("direct native viewport is outside double range");
    }
    // Pixel coordinates are shared by every row/column. Keeping the final c
    // coordinates out of the inner orbit loop removes one multiply and one
    // add per pixel from every shallow atlas tile.
    std::vector<double> x_coordinates(static_cast<size_t>(width));
    std::vector<double> y_coordinates(static_cast<size_t>(height));
    for (int px = 0; px < width; ++px) {
        const double x_offset =
            (static_cast<double>(px) - static_cast<double>(width - 1) / 2.0)
            * width_span / static_cast<double>(width);
        x_coordinates[static_cast<size_t>(px)] = center_real + x_offset;
    }
    for (int py = 0; py < height; ++py) {
        const double y_offset =
            (static_cast<double>(height - 1) / 2.0 - static_cast<double>(py))
            * height_span / static_cast<double>(height);
        y_coordinates[static_cast<size_t>(py)] = center_imag + y_offset;
    }
#if defined(__AVX2__)
    if (formula == FRACTAL_FORMULA_MANDELBROT
        && backend == 1 && avx2_runtime_available()) {
        render_direct_avx2(
            output,
            width,
            height,
            x_coordinates,
            y_coordinates,
            max_iter,
            threads);
        return;
    }
    if (backend == 1) {
        if (formula != FRACTAL_FORMULA_MANDELBROT) {
            throw std::runtime_error("AVX2 alternate-formula rendering is not implemented");
        }
        throw std::runtime_error("AVX2 backend requested but the CPU does not support AVX2");
    }
#else
    if (backend == 1) {
        throw std::runtime_error("AVX2 backend requested but this build has no AVX2 support");
    }
#endif
#ifdef FRACTAL_HAVE_OPENCL
    if (backend == 2 && formula == FRACTAL_FORMULA_MANDELBROT) {
        render_direct_opencl(
            output,
            width,
            height,
            static_cast<double>(zoom),
            static_cast<double>(x_center),
            static_cast<double>(y_center),
            max_iter);
        return;
    }
#else
    if (backend == 2) {
        if (formula != FRACTAL_FORMULA_MANDELBROT) {
            throw std::runtime_error("OpenCL alternate-formula rendering is not implemented");
        }
        throw std::runtime_error("OpenCL backend is not available in this build");
    }
#endif
#ifdef _OPENMP
    if (threads > 0) {
        // A requested team size is a performance contract for this native
        // renderer.  Some libgomp environments enable dynamic teams and can
        // silently collapse a deep render to one worker after it sees uneven
        // work; that is exactly the wrong choice for a pathological tile.
        omp_set_dynamic(0);
        omp_set_num_threads(threads);
    }
#pragma omp parallel for schedule(dynamic, 1)
#endif
    for (int py = 0; py < height; ++py) {
        const double cy = y_coordinates[static_cast<size_t>(py)];
        for (int px = 0; px < width; ++px) {
            const double cx = x_coordinates[static_cast<size_t>(px)];
            const int index = py * width + px;
            if (formula == FRACTAL_FORMULA_MANDELBROT) {
                const double q = (cx - 0.25) * (cx - 0.25) + cy * cy;
                const bool in_cardioid = q * (q + cx - 0.25) <= 0.25 * cy * cy;
                const bool in_bulb = (cx + 1.0) * (cx + 1.0) + cy * cy <= 0.0625;
                if (in_cardioid || in_bulb) {
                    output[index] = static_cast<float>(max_iter);
                    continue;
                }
            }

            double zr = formula == FRACTAL_FORMULA_JULIA ? cx : 0.0;
            double zi = formula == FRACTAL_FORMULA_JULIA ? cy : 0.0;
            const double parameter_real = formula == FRACTAL_FORMULA_JULIA
                ? julia_real : cx;
            const double parameter_imag = formula == FRACTAL_FORMULA_JULIA
                ? julia_imag : cy;
            int iteration = 0;
            for (; iteration < max_iter; ++iteration) {
                double next_real = 0.0;
                double next_imag = 0.0;
                iterate_direct_formula(
                    formula, zr, zi, parameter_real, parameter_imag,
                    next_real, next_imag);
                zr = next_real;
                zi = next_imag;
                const double magnitude_squared = zr * zr + zi * zi;
                if (magnitude_squared > ESCAPE_RADIUS_SQUARED
                    || !std::isfinite(magnitude_squared)) {
                    const double safe_squared = std::isfinite(magnitude_squared)
                        ? std::max(magnitude_squared, 4.0000001)
                        : std::numeric_limits<double>::max();
                    const double magnitude = std::sqrt(safe_squared);
                    output[index] = static_cast<float>(static_cast<double>(iteration + 1)
                        - std::log(std::log(magnitude))
                            / std::log(static_cast<double>(formula_power(formula))));
                    break;
                }
            }
            if (iteration == max_iter) output[index] = static_cast<float>(max_iter);
        }
    }
}

#ifdef FRACTAL_HAVE_MPFR

struct MpfrWorkspace {
    mpfr_t cx, cy, viewport_zoom, viewport_radius;
    mpfr_t zr, zi, next_real, next_imag, temporary;

    explicit MpfrWorkspace(mpfr_prec_t precision_bits) {
        mpfr_init2(cx, precision_bits);
        mpfr_init2(cy, precision_bits);
        mpfr_init2(viewport_zoom, precision_bits);
        mpfr_init2(viewport_radius, precision_bits);
        mpfr_init2(zr, precision_bits);
        mpfr_init2(zi, precision_bits);
        mpfr_init2(next_real, precision_bits);
        mpfr_init2(next_imag, precision_bits);
        mpfr_init2(temporary, precision_bits);
    }

    ~MpfrWorkspace() {
        mpfr_clears(
            cx, cy, viewport_zoom, viewport_radius, zr, zi,
            next_real, next_imag, temporary, nullptr);
    }

    MpfrWorkspace(const MpfrWorkspace&) = delete;
    MpfrWorkspace& operator=(const MpfrWorkspace&) = delete;
};

void make_reference_orbit(
    ReferenceContext& context,
    const char* x_text,
    const char* y_text,
    const char* viewport_zoom_text,
    int max_iter,
    int precision_bits
) {
    if (!valid_c_string(x_text) || !valid_c_string(y_text)
        || (viewport_zoom_text && !valid_c_string(viewport_zoom_text))) {
        throw std::runtime_error("native reference text is too long or null");
    }
    context.requested_max_iter = max_iter;
    precision_bits = std::max(128, precision_bits);
    MpfrWorkspace workspace(static_cast<mpfr_prec_t>(precision_bits));
    mpfr_ptr cx = workspace.cx;
    mpfr_ptr cy = workspace.cy;
    mpfr_ptr viewport_zoom = workspace.viewport_zoom;
    mpfr_ptr viewport_radius = workspace.viewport_radius;
    mpfr_ptr zr = workspace.zr;
    mpfr_ptr zi = workspace.zi;
    mpfr_ptr next_real = workspace.next_real;
    mpfr_ptr next_imag = workspace.next_imag;
    mpfr_ptr temporary = workspace.temporary;
    if (mpfr_set_str(cx, x_text, 10, MPFR_RNDN) != 0 || mpfr_set_str(cy, y_text, 10, MPFR_RNDN) != 0) {
        throw std::runtime_error("invalid MPFR Mandelbrot centre");
    }
    if (viewport_zoom_text && mpfr_set_str(viewport_zoom, viewport_zoom_text, 10, MPFR_RNDN) == 0
        && mpfr_sgn(viewport_zoom) > 0) {
        // Approximate diagonal half-span of the view.  BLA composition uses
        // this as the bound for |delta c|, so every cropped source frame is
        // covered by the same reusable approximation table.
        mpfr_set_d(viewport_radius, 2.8, MPFR_RNDN);
        mpfr_div(viewport_radius, viewport_radius, viewport_zoom, MPFR_RNDN);
        context.bla.input_radius = FloatExp::from_mpfr(viewport_radius);
    } else {
        context.bla.input_radius = FloatExp::from_parts(1.0, 0);
    }
    mpfr_set_zero(zr, 0); mpfr_set_zero(zi, 0);
    context.precision_bits = static_cast<mpfr_prec_t>(precision_bits);
    context.fast_orbit.clear();
    context.fast_orbit.reserve(static_cast<size_t>(max_iter) + 1U);
    FloatExp orbit_real_value = FloatExp::from_mpfr(zr);
    FloatExp orbit_imag_value = FloatExp::from_mpfr(zi);
    size_t finite_orbit_size = 0;
    for (int i = 0; i <= max_iter; ++i) {
        if (!orbit_real_value.finite() || !orbit_imag_value.finite()) break;
        context.fast_orbit.push_back({orbit_real_value, orbit_imag_value});
        finite_orbit_size = context.fast_orbit.size();
        mpfr_mul(next_real, zr, zr, MPFR_RNDN);
        mpfr_mul(temporary, zi, zi, MPFR_RNDN);
        mpfr_sub(next_real, next_real, temporary, MPFR_RNDN);
        mpfr_add(next_real, next_real, cx, MPFR_RNDN);
        mpfr_mul(next_imag, zr, zi, MPFR_RNDN);
        mpfr_mul_ui(next_imag, next_imag, 2, MPFR_RNDN);
        mpfr_add(next_imag, next_imag, cy, MPFR_RNDN);
        mpfr_set(zr, next_real, MPFR_RNDN); mpfr_set(zi, next_imag, MPFR_RNDN);
        orbit_real_value = FloatExp::from_mpfr(zr);
        orbit_imag_value = FloatExp::from_mpfr(zi);
    }

    // MPFR can represent the reference orbit far beyond the range of the
    // compact FloatExp exponent, but an escaping Mandelbrot orbit eventually
    // exceeds the signed-int exponent used by that hot-path representation.
    // Do not let those non-finite tail entries poison BLA coefficients.  The
    // render loop still retains every finite entry before the cutoff, which is
    // enough to detect the reference (and nearby) escape without a NaN map.
    if (finite_orbit_size == 0) {
        throw std::runtime_error("reference orbit lost finite state at iteration zero");
    }
    context.bla.map_end = static_cast<int>(finite_orbit_size) - 1;
    const FloatExp escape_radius_squared = FloatExp::from_parts(4.0, 0);
    for (size_t index = 1; index < finite_orbit_size; ++index) {
        if (fe_compare(
                fec_norm_squared(context.fast_orbit[index]),
                escape_radius_squared) > 0) {
            context.bla.map_end = static_cast<int>(index);
            break;
        }
    }
    auto render_orbit = std::make_shared<ReferenceOrbitData>();
    render_orbit->scaled.resize(context.fast_orbit.size());
    for (size_t index = 0; index < context.fast_orbit.size(); ++index) {
        render_orbit->scaled[index] = ScaledComplex::from_float_exp(
            context.fast_orbit[index].real,
            context.fast_orbit[index].imag);
    }
    render_orbit->real_double.resize(render_orbit->scaled.size());
    render_orbit->imag_double.resize(render_orbit->scaled.size());
    for (size_t index = 0; index < render_orbit->scaled.size(); ++index) {
        const ScaledComplex& value = render_orbit->scaled[index];
        render_orbit->real_double[index] = std::ldexp(value.real, value.exponent);
        render_orbit->imag_double[index] = std::ldexp(value.imag, value.exponent);
    }
    context.orbit = std::move(render_orbit);
}

#else

void make_reference_orbit(ReferenceContext&, const char*, const char*, const char*, int, int) {
    throw std::runtime_error("deep rendering requires MPFR/GMP; rebuild with make");
}

#endif

FloatExpComplex series_evaluate(
    const std::vector<FloatExpComplex>& coefficients,
    const FloatExpComplex& dc
) {
    if (coefficients.size() <= 1) return {FloatExp{0.0, 0}, FloatExp{0.0, 0}};
    FloatExpComplex result = coefficients.back();
    for (size_t index = coefficients.size() - 1; index > 1; --index) {
        result = fec_add(
            fec_mul(result, dc),
            coefficients[index - 1]);
    }
    return fec_mul(result, dc);
}

bool series_probe_is_safe(
    const FloatExpComplex& reference,
    const FloatExpComplex& exact_delta,
    const FloatExpComplex& approximate_delta,
    const FloatExp& minimum_scale_squared,
    const FloatExp& tolerance_squared
) {
    const FloatExpComplex error = fec_sub(approximate_delta, exact_delta);
    FloatExp scale = fec_norm_squared(exact_delta);
    if (fe_compare(scale, minimum_scale_squared) < 0) scale = minimum_scale_squared;
    const FloatExp allowed = fe_mul(scale, tolerance_squared);
    if (fe_compare(fec_norm_squared(error), allowed) > 0) return false;

    const bool exact_inside = fe_compare(
        fec_escape_margin_with_delta(reference, exact_delta),
        FloatExp{0.0, 0}) <= 0;
    const bool approximate_inside = fe_compare(
        fec_escape_margin_with_delta(reference, approximate_delta),
        FloatExp{0.0, 0}) <= 0;
    // The image-wide series is a jump from iteration zero to this endpoint.
    // Matching an already-escaped probe is not sufficient: an orbit can cross
    // |z|=2 early and later be represented by a numerically plausible endpoint,
    // which would turn a narrow escape band into a late, block-sized seam.
    // Only retain a series while every validation probe is still inside.
    return exact_inside && approximate_inside;
}

void build_image_series(
    ReferenceContext& context,
    const std::vector<FloatExpComplex>* builder_orbit_override = nullptr
) {
    context.image_series = ImageSeries{};
    const std::vector<FloatExpComplex>& builder_orbit = builder_orbit_override
        ? *builder_orbit_override
        : context.fast_orbit;
    if (builder_orbit.size() < 16 || context.requested_max_iter < 8) return;

    const int order = std::clamp(context.requested_series_order, 8, 32);
    const int map_end = std::clamp(
        context.bla.map_end,
        2,
        static_cast<int>(builder_orbit.size()) - 1);
    const int last_candidate = std::max(2, map_end - 1);
    const FloatExp viewport_radius = context.bla.input_radius;
    if (viewport_radius.zero() || !viewport_radius.finite()) return;

    // The reference stores the vertical half-span.  1.5x covers the corner
    // of the usual 16:9 viewport and leaves margin for odd source aspect
    // ratios.  The probes deliberately include axes, corners, and interior
    // points: checking only the four corners can miss a narrow coefficient
    // cancellation region in the middle of the image.
    const FloatExp probe_radius = fe_mul(viewport_radius, 1.5);
    const FloatExp diagonal = fe_mul(probe_radius, 0.7071067811865476);
    const FloatExp inner = fe_mul(probe_radius, 0.47);
    const std::array<FloatExpComplex, 12> probes{{
        {probe_radius, FloatExp{0.0, 0}},
        {fe_neg(probe_radius), FloatExp{0.0, 0}},
        {FloatExp{0.0, 0}, probe_radius},
        {FloatExp{0.0, 0}, fe_neg(probe_radius)},
        {diagonal, diagonal},
        {fe_neg(diagonal), diagonal},
        {diagonal, fe_neg(diagonal)},
        {fe_neg(diagonal), fe_neg(diagonal)},
        {inner, fe_mul(inner, 0.63)},
        {fe_neg(inner), fe_mul(inner, 0.63)},
        {fe_mul(inner, 0.63), inner},
        {fe_mul(inner, 0.63), fe_neg(inner)},
    }};

    std::vector<FloatExpComplex> coefficients(static_cast<size_t>(order + 1));
    std::vector<FloatExpComplex> next_coefficients(static_cast<size_t>(order + 1));
    std::array<FloatExpComplex, 12> exact{};
    std::vector<FloatExpComplex> best_coefficients;
    int best_iteration = 1;
    const FloatExp tolerance_squared = FloatExp::from_parts(
        std::ldexp(1.0, -38), 0);
    const FloatExp minimum_scale_squared = fe_sqr(probe_radius);
    for (int iteration = 0; iteration < last_candidate; ++iteration) {
        const FloatExpComplex& reference = builder_orbit[static_cast<size_t>(iteration)];
        std::fill(next_coefficients.begin(), next_coefficients.end(),
                  FloatExpComplex{FloatExp{0.0, 0}, FloatExp{0.0, 0}});
        for (int term = 1; term <= order; ++term) {
            FloatExpComplex value = fec_mul(
                fec_mul(reference, coefficients[static_cast<size_t>(term)]),
                FloatExp::from_parts(2.0, 0));
            for (int left = 1; left < term; ++left) {
                value = fec_add(
                    value,
                    fec_mul(
                        coefficients[static_cast<size_t>(left)],
                        coefficients[static_cast<size_t>(term - left)]));
            }
            if (term == 1) {
                value = fec_add(
                    value,
                    {FloatExp::from_parts(1.0, 0), FloatExp{0.0, 0}});
            }
            next_coefficients[static_cast<size_t>(term)] = value;
        }
        coefficients.swap(next_coefficients);

        bool safe = true;
        for (size_t probe = 0; probe < probes.size(); ++probe) {
            const FloatExpComplex& dc = probes[probe];
            const FloatExpComplex exact_next = fec_add(
                fec_mul(fec_mul(reference, exact[probe]), FloatExp::from_parts(2.0, 0)),
                fec_add(fec_mul(exact[probe], exact[probe]), dc));
            exact[probe] = exact_next;
            if (iteration + 1 >= order) {
                const FloatExpComplex approximate = series_evaluate(coefficients, dc);
                if (!series_probe_is_safe(
                        builder_orbit[static_cast<size_t>(iteration + 1)],
                        exact_next,
                        approximate,
                        minimum_scale_squared,
                        tolerance_squared)) {
                    safe = false;
                    break;
                }
            }
        }
        if (!safe) break;
        if (iteration + 1 >= order) {
            best_iteration = iteration + 1;
            best_coefficients = coefficients;
        }
    }

    if (best_iteration <= 1 || best_coefficients.empty()) return;
    context.image_series.enabled = true;
    context.image_series.order = order;
    context.image_series.iteration = best_iteration;
    context.image_series.radius_squared = fe_sqr(probe_radius);
    context.image_series.coefficients.reserve(best_coefficients.size());
    for (const FloatExpComplex& coefficient : best_coefficients) {
        context.image_series.coefficients.push_back(
            ScaledComplex::from_float_exp(coefficient.real, coefficient.imag));
    }
}

FloatExp fe_min(const FloatExp& a, const FloatExp& b) {
    return fe_compare(a, b) <= 0 ? a : b;
}

BlaStep merge_bla(const BlaStep& y, const BlaStep& x, const FloatExp& input_radius) {
    const FloatExp x_a = fe_sqrt(fec_norm_squared(x.A));
    const FloatExp x_b = fe_sqrt(fec_norm_squared(x.B));
    FloatExp radius = fe_sqrt(x.radius_squared);
    const FloatExp remaining = fe_sub(
        fe_sqrt(y.radius_squared), fe_mul(x_b, input_radius));
    if (fe_compare(remaining, FloatExp{0.0, 0}) > 0
        && fe_compare(x_a, FloatExp{0.0, 0}) > 0) {
        radius = fe_min(radius, fe_div(remaining, x_a));
    } else {
        radius = FloatExp{0.0, 0};
    }
    const FloatExp two = FloatExp::from_parts(2.0, 0);
    const FloatExp three = FloatExp::from_parts(3.0, 0);
    const FloatExpComplex x_a2 = fec_mul(x.A, x.A);
    const FloatExpComplex x_ab = fec_mul(x.A, x.B);
    const FloatExpComplex x_b2 = fec_mul(x.B, x.B);
    const FloatExpComplex x_a3 = fec_mul(x_a2, x.A);
    const FloatExpComplex x_a2b = fec_mul(x_a2, x.B);
    const FloatExpComplex x_ab2 = fec_mul(x.A, x_b2);
    const FloatExpComplex x_b3 = fec_mul(x_b2, x.B);
    BlaStep result{
        fec_mul(y.A, x.A),
        fec_add(fec_mul(y.A, x.B), y.B),
        fec_add(fec_mul(y.A, x.C), fec_mul(y.C, x_a2)),
        fec_add(
            fec_add(fec_mul(y.A, x.D),
                    fec_mul(y.C, fec_mul(x_ab, two))),
            fec_mul(y.D, x.A)),
        fec_add(
            fec_add(fec_mul(y.A, x.E), fec_mul(y.C, x_b2)),
            fec_add(fec_mul(y.D, x.B), y.E)),
        fec_add(
            fec_add(fec_mul(y.A, x.F), fec_mul(y.C, fec_mul(fec_mul(x.A, x.C), two))),
            fec_mul(y.F, x_a3)),
        fec_add(
            fec_add(
                fec_add(fec_mul(y.A, x.G),
                        fec_mul(y.C, fec_mul(
                            fec_add(fec_mul(x.A, x.D), fec_mul(x.B, x.C)), two))),
                fec_add(fec_mul(y.D, x.C), fec_mul(y.F, fec_mul(x_a2b, three)))),
            fec_mul(y.G, x_a2)),
        fec_add(
            fec_add(
                fec_add(fec_mul(y.A, x.H),
                        fec_mul(y.C, fec_mul(
                            fec_add(fec_mul(x.A, x.E), fec_mul(x.B, x.D)), two))),
                fec_add(fec_mul(y.D, x.D), fec_mul(y.F, fec_mul(x_ab2, three)))),
            fec_add(fec_mul(y.G, fec_mul(x_ab, two)), fec_mul(y.H, x.A))),
        fec_add(
            fec_add(
                fec_add(fec_mul(y.A, x.I), fec_mul(y.C, fec_mul(fec_mul(x.B, x.E), two))),
                fec_add(fec_mul(y.D, x.E), fec_mul(y.F, x_b3))),
            fec_add(fec_mul(y.G, x_b2), fec_add(fec_mul(y.H, x.B), y.I))),
        fe_sqr(radius),
        x.length + y.length,
    };
    return result;
}

LinearBlaBuilderStep merge_linear_bla(
    const LinearBlaBuilderStep& y,
    const LinearBlaBuilderStep& x,
    const FloatExp& input_radius
) {
    const FloatExp x_a = fe_sqrt(fec_norm_squared(x.A));
    const FloatExp x_b = fe_sqrt(fec_norm_squared(x.B));
    FloatExp radius = fe_sqrt(x.radius_squared);
    const FloatExp remaining = fe_sub(
        fe_sqrt(y.radius_squared), fe_mul(x_b, input_radius));
    if (fe_compare(remaining, FloatExp{0.0, 0}) > 0
        && fe_compare(x_a, FloatExp{0.0, 0}) > 0) {
        radius = fe_min(radius, fe_div(remaining, x_a));
    } else {
        radius = FloatExp{0.0, 0};
    }
    return {
        fec_mul(y.A, x.A),
        fec_add(fec_mul(y.A, x.B), y.B),
        fe_sqr(radius),
        x.length + y.length,
    };
}

void build_bla(ReferenceContext& context, bool retain_builder_orbit = false) {
    const int max_iter = static_cast<int>(context.fast_orbit.size()) - 1;
    const int map_end = std::clamp(context.bla.map_end, 0, max_iter);
    const int base_count = std::max(0, std::min(max_iter, map_end) - 1);
    const auto series_started = std::chrono::steady_clock::now();
    build_image_series(context);
    context.series_build_ns = static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - series_started).count());
    context.bla.levels.clear();
    context.bla.linear_levels.clear();
    if (base_count == 0) {
        if (!retain_builder_orbit) {
            context.fast_orbit.clear();
            context.fast_orbit.shrink_to_fit();
        }
        return;
    }

    std::vector<std::vector<BlaStep>> builder_levels;
    builder_levels.emplace_back(static_cast<size_t>(base_count));
    // Keep the fast map below a conservative relative error budget.  The
    // visualizer uses the iteration value for smooth colouring, so a map
    // that is merely visually plausible is not enough at a keyframe seam.
    // The endpoint guard below replays maps that approach escape, but it
    // cannot detect a map that has already crossed into the wrong basin.
    // Keep the radius near Kalles' double-precision perturbation budget
    // (about 2^-38) instead of the old 1e-8 visual-only bound. The endpoint
    // escape/replay guard below still rejects blocks that approach the
    // boundary, while the wider radius avoids a severe e40--e60 throughput
    // cliff when the perturbation enters the ordinary-double range.
    const FloatExp tolerance = FloatExp::from_long_double(
        std::ldexp(1.0L, -38));
    for (int start = 1; start <= base_count; ++start) {
        const FloatExpComplex& reference = context.fast_orbit[static_cast<size_t>(start)];
        const FloatExp reference_magnitude = fe_sqrt(fec_norm_squared(reference));
        const FloatExp scale = reference_magnitude;
        // For z' = 2*Z*z + z^2 + dc, the discarded term is z^2.  Keeping
        // |z| below epsilon*|Z| bounds it relative to the linear term;
        // sqrt(epsilon) would be much too loose and creates visible BLA
        // glitches near escape boundaries.
        const FloatExp radius = fe_mul(tolerance, scale);
        BlaStep step{
            fec_mul(reference, FloatExp::from_parts(2.0, 0)),
            {FloatExp::from_parts(1.0, 0), FloatExp{0.0, 0}},
            {FloatExp::from_parts(1.0, 0), FloatExp{0.0, 0}},
            {FloatExp{0.0, 0}, FloatExp{0.0, 0}},
            {FloatExp{0.0, 0}, FloatExp{0.0, 0}},
            {FloatExp{0.0, 0}, FloatExp{0.0, 0}},
            {FloatExp{0.0, 0}, FloatExp{0.0, 0}},
            {FloatExp{0.0, 0}, FloatExp{0.0, 0}},
            {FloatExp{0.0, 0}, FloatExp{0.0, 0}},
            fe_sqr(radius),
            1,
        };
        builder_levels[0][static_cast<size_t>(start - 1)] = step;
    }

    for (size_t level = 1; ; ++level) {
        if ((1ULL << level) > static_cast<unsigned long long>(MAX_SAFE_BLA_LENGTH)) {
            break;
        }
        const size_t previous_size = builder_levels[level - 1].size();
        const size_t current_size = previous_size / 2;
        if (current_size == 0) break;
        builder_levels.emplace_back(current_size);
        // The parameter perturbation is part of the composition domain.  A
        // BLA map is valid for d and dc, not just for a zero-parameter
        // perturbation.  Using zero here made the table look fast while
        // allowing maps whose radius was invalid for the actual viewport;
        // the endpoint replay guard then paid for that mistake one pixel at a
        // time.  Build the reusable hierarchy against the largest viewport
        // radius and let lookup accept only smaller frames.
        const FloatExp composition_input_radius = context.bla.input_radius;
        for (size_t index = 0; index < current_size; ++index) {
            builder_levels[level][index] = merge_bla(
                builder_levels[level - 1][index * 2 + 1],
                builder_levels[level - 1][index * 2],
                composition_input_radius);
        }
    }

    // Exact FloatExp coefficients are needed only while composing the
    // hierarchy. Keep a compact render-only copy so the hot lookup path does
    // not stride through hundreds of bytes of unused builder state.
    context.bla.levels.reserve(builder_levels.size());
    for (const auto& builder_level : builder_levels) {
        auto& render_level = context.bla.levels.emplace_back();
        render_level.reserve(builder_level.size());
        for (const BlaStep& step : builder_level) {
            render_level.push_back(compact_bla_step(step));
        }
    }

    // Keep separate linear hierarchies for the normal deep branch and for
    // genuinely tiny viewports.  The first is valid for the reusable
    // reference viewport; the second is composed with a bound 20 decades
    // smaller, so e40--e4000 frames can use long maps without paying for the
    // shallow frame's parameter-radius pessimism.
    auto build_linear_levels = [&](const FloatExp& composition_input_radius) {
        std::vector<std::vector<LinearBlaBuilderStep>> builder_levels;
        builder_levels.emplace_back(static_cast<size_t>(base_count));
        for (int start = 1; start <= base_count; ++start) {
            const FloatExpComplex& reference = context.fast_orbit[static_cast<size_t>(start)];
            const FloatExp radius = fe_mul(
                tolerance,
                fe_sqrt(fec_norm_squared(reference)));
            builder_levels[0][static_cast<size_t>(start - 1)] = {
                fec_mul(reference, FloatExp::from_parts(2.0, 0)),
                {FloatExp::from_parts(1.0, 0), FloatExp{0.0, 0}},
                fe_sqr(radius),
                1,
            };
        }
        for (size_t level = 1; ; ++level) {
            if ((1ULL << level) > static_cast<unsigned long long>(MAX_SAFE_LINEAR_BLA_LENGTH)) {
                break;
            }
            const size_t previous_size = builder_levels[level - 1].size();
            const size_t current_size = (previous_size + 1) / 2;
            if (current_size == 0) break;
            builder_levels.emplace_back(current_size);
            for (size_t index = 0; index < current_size; ++index) {
                const size_t first = index * 2;
                if (first + 1 < previous_size) {
                    builder_levels[level][index] = merge_linear_bla(
                        builder_levels[level - 1][first + 1],
                        builder_levels[level - 1][first],
                        composition_input_radius);
                } else {
                    builder_levels[level][index] =
                        builder_levels[level - 1][first];
                }
            }
        }
        std::vector<std::vector<LinearBlaStep>> render_levels;
        render_levels.reserve(builder_levels.size());
        for (const auto& builder_level : builder_levels) {
            auto& render_level = render_levels.emplace_back();
            render_level.reserve(builder_level.size());
            for (const LinearBlaBuilderStep& step : builder_level) {
                render_level.push_back(compact_linear_bla_step(step));
            }
        }
        return render_levels;
    };

    context.bla.deep_input_radius = fe_mul(
        context.bla.input_radius,
        1.0e-20);
    context.bla.linear_levels = build_linear_levels(context.bla.input_radius);
    context.bla.deep_linear_levels = build_linear_levels(context.bla.deep_input_radius);
    if (!retain_builder_orbit) {
        context.fast_orbit.clear();
        context.fast_orbit.shrink_to_fit();
    }
}

inline FloatExp compact_norm_squared(const ScaledComplex& value) {
    const ScaledNorm norm = sc_norm_squared(value);
    return {norm.mantissa, norm.exponent};
}

FloatExp retarget_bla_radius(
    const FloatExp& x_radius_squared,
    const FloatExp& y_radius_squared,
    const ScaledComplex& x_a_coefficient,
    const ScaledComplex& x_b_coefficient,
    const FloatExp& input_radius
) {
    const FloatExp x_a = fe_sqrt(compact_norm_squared(x_a_coefficient));
    const FloatExp x_b = fe_sqrt(compact_norm_squared(x_b_coefficient));
    FloatExp radius = fe_sqrt(x_radius_squared);
    const FloatExp remaining = fe_sub(
        fe_sqrt(y_radius_squared), fe_mul(x_b, input_radius));
    if (fe_compare(remaining, FloatExp{0.0, 0}) > 0
        && fe_compare(x_a, FloatExp{0.0, 0}) > 0) {
        radius = fe_min(radius, fe_div(remaining, x_a));
    } else {
        radius = FloatExp{0.0, 0};
    }
    // The root's compact render coefficients carry the same double
    // mantissas as the builder, but use a shared complex exponent.  A tiny
    // inward bias keeps a retargeted acceptance radius conservative under
    // the final norm conversion.
    radius = fe_mul(radius, 1.0 - std::ldexp(1.0, -40));
    return fe_sqr(radius);
}

void retarget_cubic_levels(
    std::vector<std::vector<FastBlaStep>>& levels,
    const FloatExp& input_radius
) {
    for (size_t level = 1; level < levels.size(); ++level) {
        const auto& previous = levels[level - 1];
        auto& current = levels[level];
        for (size_t index = 0; index < current.size(); ++index) {
            const FastBlaStep& x = previous[index * 2];
            const FastBlaStep& y = previous[index * 2 + 1];
            current[index].radius_squared = retarget_bla_radius(
                x.radius_squared,
                y.radius_squared,
                x.coefficients[0],
                x.coefficients[1],
                input_radius);
        }
    }
}

void retarget_linear_levels(
    std::vector<std::vector<LinearBlaStep>>& levels,
    const FloatExp& input_radius
) {
    for (size_t level = 1; level < levels.size(); ++level) {
        const auto& previous = levels[level - 1];
        auto& current = levels[level];
        for (size_t index = 0; index < current.size(); ++index) {
            const size_t first = index * 2;
            if (first + 1 >= previous.size()) {
                current[index].radius_squared = previous[first].radius_squared;
                continue;
            }
            const LinearBlaStep& x = previous[first];
            const LinearBlaStep& y = previous[first + 1];
            current[index].radius_squared = retarget_bla_radius(
                x.radius_squared,
                y.radius_squared,
                x.A,
                x.B,
                input_radius);
        }
    }
}

[[maybe_unused]] void build_retargeted_bla(
    ReferenceContext& context,
    const ReferenceContext& source
) {
    const auto series_started = std::chrono::steady_clock::now();
    build_image_series(context, &source.fast_orbit);
    context.series_build_ns = static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - series_started).count());

    context.bla.levels = source.bla.levels;
    retarget_cubic_levels(context.bla.levels, context.bla.input_radius);
    context.bla.linear_levels = source.bla.linear_levels;
    retarget_linear_levels(context.bla.linear_levels, context.bla.input_radius);
    context.bla.deep_input_radius = fe_mul(context.bla.input_radius, 1.0e-20);
    context.bla.deep_linear_levels = source.bla.linear_levels;
    retarget_linear_levels(
        context.bla.deep_linear_levels,
        context.bla.deep_input_radius);
    context.fast_orbit.clear();
    context.fast_orbit.shrink_to_fit();
}

#ifdef FRACTAL_HAVE_MPFR

struct RenderTimeBudget {
    bool enabled = false;
    std::chrono::steady_clock::time_point deadline{};
    std::atomic<bool> exceeded{false};
};

inline bool render_time_budget_expired(
    RenderTimeBudget* budget,
    std::uint32_t& budget_ticks
) {
    if (!budget || !budget->enabled) return false;
    if ((budget_ticks++ & 255U) != 0U) return false;
    if (budget->exceeded.load(std::memory_order_relaxed)) return true;
    if (std::chrono::steady_clock::now() >= budget->deadline) {
        budget->exceeded.store(true, std::memory_order_relaxed);
        return true;
    }
    return false;
}

FloatExp parse_zoom_float_exp(const char* text, mpfr_prec_t precision_bits) {
    if (!valid_c_string(text)) {
        throw std::runtime_error("native zoom text is too long or null");
    }
    mpfr_t value;
    mpfr_init2(value, precision_bits);
    const int status = mpfr_set_str(value, text, 10, MPFR_RNDN);
    if (status != 0 || mpfr_sgn(value) <= 0 || !mpfr_number_p(value)) {
        mpfr_clear(value);
        throw std::runtime_error("invalid deep Mandelbrot zoom");
    }
    const FloatExp result = FloatExp::from_mpfr(value);
    mpfr_clear(value);
    if (!result.finite() || result.zero()) {
        throw std::runtime_error("deep Mandelbrot zoom is outside the native exponent range");
    }
    const long double log10_zoom = fe_log(result) / LOG_TEN;
    if (!std::isfinite(log10_zoom)
        || log10_zoom < MIN_NATIVE_LOG10_ZOOM
        || log10_zoom > MAX_NATIVE_LOG10_ZOOM) {
        throw std::runtime_error("deep Mandelbrot zoom is outside the supported range");
    }
    return result;
}

template <bool CollectStats, bool EnableCycleDetection>
bool render_scaled_double_tail(
    float& output,
    const ScaledComplex& dc,
    const ReferenceContext& context,
    int max_iter,
    int& iteration,
    int& reference_index,
    ScaledComplex delta,
    bool disable_cycle_detection,
    bool strict_cycle_detection,
    bool strict_render,
    RenderTimeBudget* time_budget,
    std::uint32_t& budget_ticks,
    bool& deadline_abort,
    bool& unresolved_tail,
    RenderStats* stats
) {
    constexpr int MAX_TAIL_REBASES = 64;
    // A normal 960x540 probe at this location already needs about 4.6k
    // double-tail steps for its slowest pixel. Keep the emergency ceiling
    // above the supported video iteration range so ordinary tails do not
    // take the slower restart path.
    constexpr int MAX_TAIL_STEPS = 65536;
    const double dc_real = sc_to_double(dc);
    const double dc_imag = std::ldexp(dc.imag, dc.exponent);
    double delta_real = sc_to_double(delta);
    double delta_imag = std::ldexp(delta.imag, delta.exponent);
    output = static_cast<float>(max_iter);
    double tortoise_real = 0.0;
    double tortoise_imag = 0.0;
    int cycle_power = 1;
    int cycle_length = 0;
    int tail_steps = 0;
    int tail_iterations = 0;
    int tail_rebases = 0;
    bool cycle_ready = false;
    while (iteration < max_iter
        && reference_index >= 0
        && reference_index + 1 < static_cast<int>(context.orbit->real_double.size())) {
        if (render_time_budget_expired(time_budget, budget_ticks)) {
            output = std::numeric_limits<float>::quiet_NaN();
            deadline_abort = true;
            unresolved_tail = true;
            if constexpr (CollectStats) {
                ++stats->deadline_aborts;
                ++stats->unresolved_pixels;
            }
            return false;
        }
        const double reference_real =
            context.orbit->real_double[static_cast<size_t>(reference_index)];
        const double reference_imag =
            context.orbit->imag_double[static_cast<size_t>(reference_index)];
        const double linear_real = 2.0 * (reference_real * delta_real - reference_imag * delta_imag);
        const double linear_imag = 2.0 * (reference_real * delta_imag + reference_imag * delta_real);
        const double square_real = delta_real * delta_real - delta_imag * delta_imag;
        const double square_imag = 2.0 * delta_real * delta_imag;
        delta_real = linear_real + square_real + dc_real;
        delta_imag = linear_imag + square_imag + dc_imag;
        ++reference_index;
        ++iteration;
        if (++tail_iterations > MAX_TAIL_STEPS) {
            if constexpr (CollectStats) {
                ++stats->tail_rebase_fallbacks;
                ++stats->unresolved_pixels;
            }
            unresolved_tail = true;
            output = strict_render
                ? std::numeric_limits<float>::quiet_NaN()
                : static_cast<float>(iteration);
            return true;
        }
        if constexpr (CollectStats) {
            ++stats->logical_iterations;
            ++stats->exact_steps;
        }
        const double total_real =
            context.orbit->real_double[static_cast<size_t>(reference_index)] + delta_real;
        const double total_imag =
            context.orbit->imag_double[static_cast<size_t>(reference_index)] + delta_imag;
        const double magnitude_squared = total_real * total_real + total_imag * total_imag;
        if (!std::isfinite(magnitude_squared)) {
            // A non-finite tail is a numerical glitch, not proof of an
            // interior pixel.  Strict renders expose it to the caller as an
            // unresolved mask so a secondary reference can repair it rather
            // than painting a false escape band.
            unresolved_tail = true;
            if constexpr (CollectStats) {
                ++stats->glitch_count;
                ++stats->unresolved_pixels;
            }
            output = strict_render
                ? std::numeric_limits<float>::quiet_NaN()
                : static_cast<float>(iteration);
            return false;
        }
        if (magnitude_squared > 4.0) {
            const double magnitude = std::sqrt(std::max(magnitude_squared, 4.0000001));
            output = static_cast<float>(static_cast<double>(iteration)
                - std::log(std::log(magnitude)) / static_cast<double>(LOG_TWO));
            return false;
        }
        const double delta_magnitude_squared =
            delta_real * delta_real + delta_imag * delta_imag;

        ++tail_steps;
        if constexpr (EnableCycleDetection) {
            if (!disable_cycle_detection
                && tail_steps >= 64
                && (tail_steps & 31) == 0) {
            // Brent-style cycle detection is deliberately conservative: it
            // only runs after a long bounded tail and requires a near-exact
            // recurrence well inside the escape circle. Sample the tail at
            // the same cadence as the scaled path; checking every iteration
            // made this fallback disproportionately expensive without adding
            // useful precision for an audio frame.
            if (!cycle_ready) {
                tortoise_real = total_real;
                tortoise_imag = total_imag;
                cycle_power = 1;
                cycle_length = 0;
                cycle_ready = true;
            } else {
                const double cycle_delta_real = total_real - tortoise_real;
                const double cycle_delta_imag = total_imag - tortoise_imag;
                const double cycle_distance_squared =
                    cycle_delta_real * cycle_delta_real
                    + cycle_delta_imag * cycle_delta_imag;
                const int cycle_minimum_iteration = strict_cycle_detection ? 2048 : 512;
                const double cycle_tolerance = strict_cycle_detection ? 1.0e-24 : 1.0e-18;
                if (iteration >= cycle_minimum_iteration
                    && magnitude_squared < 3.0
                    && cycle_distance_squared
                        <= cycle_tolerance * std::max(1.0, magnitude_squared)) {
                    return false;
                }
                ++cycle_length;
                if (cycle_length >= cycle_power) {
                    tortoise_real = total_real;
                    tortoise_imag = total_imag;
                    cycle_power = std::min(cycle_power * 2, 1 << 20);
                    cycle_length = 0;
                }
            }
        }
        }
        if (magnitude_squared < delta_magnitude_squared) {
            delta_real = total_real;
            delta_imag = total_imag;
            reference_index = 0;
            tail_steps = 0;
            cycle_ready = false;
            if constexpr (CollectStats) ++stats->tail_rebases;
            if (++tail_rebases > MAX_TAIL_REBASES) {
                if constexpr (CollectStats) {
                    ++stats->tail_rebase_fallbacks;
                    ++stats->unresolved_pixels;
                }
                unresolved_tail = true;
                output = strict_render
                    ? std::numeric_limits<float>::quiet_NaN()
                    : static_cast<float>(iteration);
                return true;
            }
        }
    }
    if (iteration < max_iter) {
        // The compact reference tail ended before the requested iteration
        // budget.  Do not misclassify the unresolved pixel as an interior.
        unresolved_tail = true;
        if constexpr (CollectStats) ++stats->unresolved_pixels;
        output = strict_render
            ? std::numeric_limits<float>::quiet_NaN()
            : static_cast<float>(iteration);
    }
    return false;
}


template <bool CollectStats, bool EnableCycleDetection>
void render_bla_impl(
    float* __restrict output,
    int width,
    int height,
    const char* zoom_text,
    const ReferenceContext& context,
    int max_iter,
    int threads,
    int series_order,
    int series_block,
    const FractalRenderOptions& options,
    RenderStats* stats_out,
    const std::vector<ScaledComplex>* point_offsets = nullptr,
    const FloatExp* point_radius = nullptr
) {
    const auto render_started = std::chrono::steady_clock::now();
    const FloatExp zoom = parse_zoom_float_exp(zoom_text, context.precision_bits);
    const FloatExp inverse_zoom = fe_div(FloatExp::from_parts(1.0, 0), zoom);
    const FloatExp view_height = fe_mul(inverse_zoom, 2.8);
    const FloatExp view_width = fe_mul(
        view_height, static_cast<double>(width) / static_cast<double>(height));
    const FloatExp current_input_radius = point_radius != nullptr
        ? *point_radius
        : fe_mul(inverse_zoom, 2.8);
    const bool bla_radius_covers_view =
        fe_compare(current_input_radius, context.bla.input_radius) <= 0;
    const bool disable_bla = !bla_radius_covers_view || options.disable_bla != 0;
    // Cycle termination is an intentionally lossy interior shortcut.  A
    // near-parabolic orbit can look periodic for thousands of iterations and
    // still escape later, so strict/quality output must never enable it by
    // accident.  Callers that explicitly opt into the conservative variant
    // set strict_cycle; draft/non-strict callers may retain the faster
    // heuristic.
    const bool cycle_detection_enabled = EnableCycleDetection
        && options.disable_cycle == 0
        && (options.strict == 0 || options.strict_cycle != 0);
    const bool strict_cycle_detection = options.strict_cycle != 0;
    const int cycle_minimum_iteration = strict_cycle_detection ? 2048 : 512;
    const int cycle_exponent_margin = strict_cycle_detection ? 78 : 60;
    RenderTimeBudget render_budget;
    if (options.time_budget_ms > 0) {
        render_budget.enabled = true;
        render_budget.deadline = std::chrono::steady_clock::now()
            + std::chrono::milliseconds(options.time_budget_ms);
    }
    RenderTimeBudget* time_budget = render_budget.enabled ? &render_budget : nullptr;
    const bool use_deep_linear =
        fe_compare(current_input_radius, context.bla.deep_input_radius) <= 0;
    const FloatExp ultra_deep_input_radius = fe_mul(
        context.bla.input_radius,
        1.0e-40);
    const bool use_ultra_deep_linear =
        fe_compare(current_input_radius, ultra_deep_input_radius) <= 0;
    // The existing BLA hierarchy is a real polynomial approximation, not a
    // compatibility label.  Use its lower-precision tail only after the
    // perturbation has grown large enough for ordinary doubles; keeping BLA
    // active through e100 is what turns reference reuse into iteration reuse.
    // The cubic terms suppress the accumulated error that limited the old
    // quadratic map to short blocks.
    const int max_bla_length = std::clamp(
        std::min(series_block, options.max_bla_length), 2, MAX_SAFE_BLA_LENGTH);
    const int max_linear_bla_length = std::clamp(
        std::min(series_block, options.max_linear_bla_length),
        2,
        use_ultra_deep_linear
            ? MAX_SAFE_LINEAR_BLA_LENGTH
            : (use_deep_linear
                ? MAX_SAFE_DEEP_LINEAR_BLA_LENGTH
                : MAX_SAFE_BLA_LENGTH));
    const int approximation_order = std::clamp(series_order, 1, 3);
    const bool image_series_available = context.image_series.enabled
        && context.image_series.order >= options.series_min_terms
        && context.image_series.order <= options.series_max_terms
        && fe_compare(
            current_input_radius,
            fe_sqrt(context.image_series.radius_squared)) <= 0
        && context.image_series.iteration < max_iter;

    // These offsets are shared by every pixel in a row/column.  Computing
    // the FloatExp multiplication once here removes two viewport-scale
    // operations from the innermost perturbation loop without changing the
    // exact pixel-centre mapping.
    std::vector<FloatExp> x_offsets;
    std::vector<FloatExp> y_offsets;
    if (point_offsets == nullptr) {
        x_offsets.resize(static_cast<size_t>(width));
        y_offsets.resize(static_cast<size_t>(height));
        for (int py = 0; py < height; ++py) {
            const double y_fraction =
                (static_cast<double>(height - 1) / 2.0 - static_cast<double>(py))
                / static_cast<double>(height);
            y_offsets[static_cast<size_t>(py)] = fe_mul(view_height, y_fraction);
        }
        for (int px = 0; px < width; ++px) {
            const double x_fraction =
                (static_cast<double>(px) - static_cast<double>(width - 1) / 2.0)
                / static_cast<double>(width);
            x_offsets[static_cast<size_t>(px)] = fe_mul(view_width, x_fraction);
        }
    } else if (point_offsets->size() != static_cast<size_t>(width * height)) {
        throw std::runtime_error("point renderer dimensions do not match offset count");
    }

    // Set the requested team size before sizing the diagnostic slots. This
    // keeps --stats safe when a caller requests more threads than the runtime
    // default, while the normal render still pays no allocation cost.
#ifdef _OPENMP
    if (threads > 0) {
        omp_set_dynamic(0);
        omp_set_num_threads(threads);
    }
#endif
    std::vector<RenderStats> thread_stats;
    if constexpr (CollectStats) {
        int worker_count = 1;
#ifdef _OPENMP
        // A direct C caller can leave the OpenMP default under control of the
        // environment.  Bound it before allocating per-worker diagnostics;
        // otherwise OMP_NUM_THREADS could turn --stats into an allocation DoS.
        omp_set_dynamic(0);
        worker_count = std::clamp(
            std::max(1, omp_get_max_threads()),
            1,
            MAX_NATIVE_THREADS);
        omp_set_num_threads(worker_count);
#endif
        thread_stats.resize(static_cast<size_t>(worker_count));
    }
#ifdef _OPENMP
    // Deep zoom cost is highly non-uniform: a single hard filament can make
    // one row take orders of magnitude longer than its neighbours.  A row is
    // therefore too coarse a unit of work.  Use small contiguous pixel
    // chunks so the expensive pixels are redistributed among all workers.
#pragma omp parallel for schedule(dynamic, 64)
#endif
    for (int linear_pixel = 0; linear_pixel < width * height; ++linear_pixel) {
        const int py = linear_pixel / width;
        const int px = linear_pixel - py * width;
        RenderStats* stats = nullptr;
        if constexpr (CollectStats) {
#ifdef _OPENMP
            // The explicit team-size bound above should make this a no-op in
            // normal operation; keep the index defensive for unusual OpenMP
            // runtimes that report a larger team than requested.
            const int worker_index = std::clamp(
                omp_get_thread_num(),
                0,
                static_cast<int>(thread_stats.size()) - 1);
            stats = &thread_stats[static_cast<size_t>(worker_index)];
#else
            stats = &thread_stats[0];
#endif
        }
            if constexpr (CollectStats) ++stats->pixels;
            const ScaledComplex dc = point_offsets != nullptr
                ? (*point_offsets)[static_cast<size_t>(linear_pixel)]
                : ScaledComplex::from_float_exp(
                    x_offsets[static_cast<size_t>(px)],
                    y_offsets[static_cast<size_t>(py)]);
            const int index = py * width + px;
            ScaledComplex delta = dc;
            int reference_index = 1;
            int iteration = 1;
            if (image_series_available
                && sc_compare_norm(
                    sc_norm_squared(dc),
                    ScaledNorm{
                        context.image_series.radius_squared.mantissa,
                        context.image_series.radius_squared.exponent,
                    }) <= 0) {
                // Horner evaluation of the validated image-wide series skips
                // the same early reference iterations for every pixel.  The
                // ordinary perturbation/BLA path remains responsible for the
                // rest of the orbit and for all escape/rebase checks.
                delta = evaluate_image_series(context.image_series, dc);
                reference_index = context.image_series.iteration;
                iteration = context.image_series.iteration;
                if constexpr (CollectStats) {
                    ++stats->series_pixels;
                    stats->series_jumps += static_cast<std::uint64_t>(iteration);
                }
            }
            bool escaped = false;
            bool have_total = false;
            ScaledComplex total{};
            ScaledNorm total_norm{};
            ScaledComplex cycle_tortoise{};
            int cycle_power = 1;
            int cycle_length = 0;
            bool cycle_ready = false;
            bool pixel_disable_bla = false;
            int perturbation_rebases = 0;
            std::uint32_t budget_ticks = 0;
            bool deadline_abort = false;
            bool unresolved_pixel = false;

            while (iteration < max_iter) {
                if (render_time_budget_expired(time_budget, budget_ticks)) {
                    output[index] = std::numeric_limits<float>::quiet_NaN();
                    deadline_abort = true;
                    unresolved_pixel = true;
                    if constexpr (CollectStats) {
                        ++stats->deadline_aborts;
                        ++stats->unresolved_pixels;
                    }
                    break;
                }
                if (reference_index < 0
                    || reference_index >= static_cast<int>(context.orbit->scaled.size())) {
                    output[index] = static_cast<float>(iteration);
                    escaped = true;
                    break;
                }
                if (!have_total) {
                    total = sc_add(
                        context.orbit->scaled[static_cast<size_t>(reference_index)], delta);
                    total_norm = sc_norm_squared_with_delta(
                        context.orbit->scaled[static_cast<size_t>(reference_index)],
                        delta);
                }
                have_total = false;
                if (sc_outside_escape(total_norm)
                    || sc_outside_escape_with_delta(
                        context.orbit->scaled[static_cast<size_t>(reference_index)],
                        delta)) {
                    output[index] = smooth_escape_scaled(iteration, total_norm);
                    escaped = true;
                    break;
                }

                // The scaled perturbation path used to have no interior
                // termination at all: a deeply zoomed attracting pixel could
                // execute the entire iteration cap even after its orbit had
                // settled. Sample a Brent-style cycle detector every 32
                // iterations so its FloatExp arithmetic is negligible next
                // to the exact/BLA work. A match requires roughly 2^-80
                // relative state error and a comfortably bounded orbit; this
                // is intentionally stricter than a visual similarity test.
                if constexpr (EnableCycleDetection) {
                    if (cycle_detection_enabled
                        && iteration >= cycle_minimum_iteration
                        && (iteration & 31) == 0
                        && sc_compare_norm(total_norm, ScaledNorm{0.75, 2}) < 0) {
                        if (!cycle_ready) {
                            cycle_tortoise = total;
                            cycle_power = 1;
                            cycle_length = 0;
                            cycle_ready = true;
                        } else {
                            const ScaledNorm cycle_distance = sc_norm_squared(
                                sc_sub(total, cycle_tortoise));
                            const int scale_exponent = std::max(total_norm.exponent, 0);
                            if (cycle_distance.mantissa == 0.0
                                || cycle_distance.exponent <= scale_exponent - cycle_exponent_margin) {
                                if constexpr (CollectStats) ++stats->cycle_inside;
                                output[index] = static_cast<float>(max_iter);
                                escaped = true;
                                break;
                            }
                            ++cycle_length;
                            if (cycle_length >= cycle_power) {
                                cycle_tortoise = total;
                                cycle_power = std::min(cycle_power * 2, 1 << 20);
                                cycle_length = 0;
                            }
                        }
                    }
                }

                // Rebase to the beginning of the same reference orbit when
                // the perturbation is larger than the reference state.  This
                // is the cheap glitch-avoidance step used by deep zoomers.
                const ScaledNorm delta_norm = sc_norm_squared(delta);
                if (sc_compare_norm(total_norm, delta_norm) < 0) {
                    if (++perturbation_rebases > 64) {
                        // Repeated rebasing means this pixel has left the
                        // useful reference neighbourhood. Continue from the
                        // already computed total with the exact scaled
                        // recurrence instead of resetting reference_index and
                        // potentially spinning forever without advancing the
                        // logical iteration counter.
                        const ScaledComplex parameter = sc_add(
                            context.orbit->scaled[1],
                            dc);
                        while (iteration < max_iter) {
                            if (render_time_budget_expired(time_budget, budget_ticks)) {
                                output[index] = std::numeric_limits<float>::quiet_NaN();
                                deadline_abort = true;
                                unresolved_pixel = true;
                                if constexpr (CollectStats) {
                                    ++stats->deadline_aborts;
                                    ++stats->unresolved_pixels;
                                }
                                break;
                            }
                            total = sc_add(sc_mul(total, total), parameter);
                            ++iteration;
                            if constexpr (CollectStats) {
                                ++stats->logical_iterations;
                                ++stats->exact_steps;
                            }
                            total_norm = sc_norm_squared(total);
                            if (sc_outside_escape(total_norm)) {
                                output[index] = smooth_escape_scaled(iteration, total_norm);
                                escaped = true;
                                break;
                            }
                        }
                        if (deadline_abort) break;
                        if (!escaped) output[index] = static_cast<float>(max_iter);
                        escaped = true;
                        break;
                    }
                    delta = total;
                    reference_index = 0;
                    continue;
                }

                if (context.bla.map_end > 1
                    && reference_index >= context.bla.map_end) {
                    // The MPFR reference itself has escaped.  Continuing to
                    // express a nearby orbit as Z + delta after that point
                    // can force every BLA lookup into a pathological replay
                    // tail.  Switch to the mathematically identical direct
                    // recurrence from the current total, using z_1 as the
                    // exact reference parameter c_ref and adding dc once.
                    // This is still scaled arithmetic, so it remains valid
                    // when the centre is far beyond double precision.
                    const ScaledComplex parameter = sc_add(
                        context.orbit->scaled[1],
                        dc);
                    while (iteration < max_iter) {
                        if (render_time_budget_expired(time_budget, budget_ticks)) {
                            output[index] = std::numeric_limits<float>::quiet_NaN();
                            deadline_abort = true;
                            unresolved_pixel = true;
                            if constexpr (CollectStats) {
                                ++stats->deadline_aborts;
                                ++stats->unresolved_pixels;
                            }
                            break;
                        }
                        total = sc_add(
                            sc_mul(total, total),
                            parameter);
                        ++iteration;
                        if constexpr (CollectStats) {
                            ++stats->logical_iterations;
                            ++stats->exact_steps;
                        }
                        total_norm = sc_norm_squared(total);
                        if (sc_outside_escape(total_norm)) {
                            output[index] = smooth_escape_scaled(iteration, total_norm);
                            escaped = true;
                            break;
                        }
                    }
                    if (deadline_abort) break;
                    if (!escaped) {
                        output[index] = static_cast<float>(max_iter);
                        escaped = true;
                    }
                    break;
                }

                const FloatExp delta_norm_float{delta_norm.mantissa, delta_norm.exponent};
                const int remaining_iterations = max_iter - iteration;
                const int effective_order =
                    approximation_order >= 3 && delta_norm.exponent < -115
                        ? (delta_norm.exponent < -160 ? 1 : 2)
                        : approximation_order;
                const LinearBlaStep* linear_step = nullptr;
                const FastBlaStep* step = nullptr;
                if (!disable_bla && !pixel_disable_bla) {
                    if (effective_order <= 1) {
                        linear_step = context.bla.lookup_linear(
                            reference_index,
                            delta_norm_float,
                            std::min(max_linear_bla_length, remaining_iterations),
                            use_deep_linear);
                    } else {
                        step = context.bla.lookup(
                            reference_index,
                            delta_norm_float,
                            std::min(max_bla_length, remaining_iterations));
                    }
                }
                const int map_length = linear_step != nullptr
                    ? linear_step->length
                    : (step != nullptr ? step->length : 0);
                // A normal double keeps roughly 53 significant bits. Once
                // the perturbation is large enough for a double tail, prefer
                // that compact fallback only when no validated BLA block is
                // available. The old early branch skipped this lookup and
                // needlessly converted otherwise reusable late-orbit blocks
                // into thousands of scalar exact iterations.
                if (!disable_bla
                    && !pixel_disable_bla
                    && (map_length <= 0 || reference_index + map_length > max_iter)
                    && delta_norm.exponent > -90) {
                    if constexpr (CollectStats) ++stats->double_tail_pixels;
                    const int tail_start_iteration = iteration;
                    bool tail_pathological = false;
                    if constexpr (EnableCycleDetection) {
                        tail_pathological = cycle_detection_enabled
                            ? render_scaled_double_tail<CollectStats, true>(
                                output[index], dc, context, max_iter,
                                iteration, reference_index, delta,
                                !cycle_detection_enabled, strict_cycle_detection,
                                options.strict != 0,
                                time_budget, budget_ticks, deadline_abort,
                                unresolved_pixel,
                                stats)
                            : render_scaled_double_tail<CollectStats, false>(
                                output[index], dc, context, max_iter,
                                iteration, reference_index, delta,
                                !cycle_detection_enabled, strict_cycle_detection,
                                options.strict != 0,
                                time_budget, budget_ticks, deadline_abort,
                                unresolved_pixel,
                                stats);
                    } else {
                        tail_pathological = render_scaled_double_tail<CollectStats, false>(
                            output[index], dc, context, max_iter,
                            iteration, reference_index, delta,
                            true, strict_cycle_detection,
                            options.strict != 0,
                            time_budget, budget_ticks, deadline_abort,
                            unresolved_pixel,
                            stats);
                    }
                    if (deadline_abort) break;
                    if (unresolved_pixel) break;
                    if constexpr (CollectStats) {
                        const std::uint64_t tail_steps = static_cast<std::uint64_t>(
                            std::max(0, iteration - tail_start_iteration));
                        stats->tail_steps += tail_steps;
                        stats->max_tail_steps = std::max(stats->max_tail_steps, tail_steps);
                    }
                    if (tail_pathological) {
                        // A long tail that keeps rebasing is a perturbation
                        // glitch, not evidence that the pixel is interior.
                        // Restart from the original dc and use the slower but
                        // bounded scaled-exact recurrence for this pixel.
                        delta = dc;
                        reference_index = 1;
                        iteration = 1;
                        have_total = false;
                        pixel_disable_bla = true;
                        continue;
                    }
                    escaped = true;
                    break;
                }
                if (map_length > 0 && reference_index + map_length <= max_iter) {
                    if constexpr (CollectStats) {
                        ++stats->bla_blocks;
                        if (linear_step != nullptr) {
                            ++stats->linear_blocks;
                        } else {
                            ++stats->cubic_blocks;
                        }
                        ++stats->series_jumps;
                        record_bla_length(stats, map_length);
                    }
                    const ScaledComplex previous_delta = delta;
                    const int previous_reference_index = reference_index;
                    const int previous_iteration = iteration;
                    const ScaledComplex input_delta = delta;
                    delta = linear_step != nullptr
                        ? sc_add(
                            sc_mul(linear_step->A, input_delta),
                            sc_mul(linear_step->B, dc))
                        : apply_bla_series(*step, input_delta, dc, effective_order);
                    reference_index += map_length;
                    iteration += map_length;
                    const ScaledComplex endpoint = sc_add(
                        context.orbit->scaled[static_cast<size_t>(reference_index)], delta);
                    const ScaledNorm endpoint_norm = sc_norm_squared_with_delta(
                        context.orbit->scaled[static_cast<size_t>(reference_index)],
                        delta);
                    // A block that approaches the escape boundary is replayed
                    // one iteration at a time so smooth colouring does not
                    // acquire broad BLA-sized bands.
                    if (!sc_finite(delta) || !sc_finite(endpoint)) {
                        // A bad approximation must be retried from the same
                        // state with BLA disabled for this pixel.  Previously
                        // this path could write max_iter and create a large
                        // false black region.
                        delta = previous_delta;
                        reference_index = previous_reference_index;
                        iteration = previous_iteration;
                        if constexpr (CollectStats) {
                            ++stats->bla_retries;
                            if (!pixel_disable_bla) ++stats->bla_disabled_pixels;
                        }
                        pixel_disable_bla = true;
                        have_total = false;
                        continue;
                    }
                    if (sc_compare_norm(endpoint_norm, ScaledNorm{0.75, 2}) >= 0) {
                        delta = previous_delta;
                        reference_index = previous_reference_index;
                        iteration = previous_iteration;
                        ScaledComplex replay_total{};
                        ScaledNorm replay_norm{};
                        bool retry_without_bla = false;
                        for (int replay = 0; replay < map_length && iteration < max_iter; ++replay) {
                            if (render_time_budget_expired(time_budget, budget_ticks)) {
                                output[index] = std::numeric_limits<float>::quiet_NaN();
                                deadline_abort = true;
                                unresolved_pixel = true;
                                if constexpr (CollectStats) {
                                    ++stats->deadline_aborts;
                                    ++stats->unresolved_pixels;
                                }
                                break;
                            }
                            const ScaledComplex reference =
                                context.orbit->scaled[static_cast<size_t>(reference_index)];
                            delta = sc_add(
                                sc_double(sc_mul(reference, delta)),
                                sc_add(sc_mul(delta, delta), dc));
                            ++reference_index;
                            ++iteration;
                            if constexpr (CollectStats) {
                                ++stats->replay_steps;
                                ++stats->logical_iterations;
                            }
                            replay_total = sc_add(
                                context.orbit->scaled[static_cast<size_t>(reference_index)], delta);
                            replay_norm = sc_norm_squared_with_delta(
                                context.orbit->scaled[static_cast<size_t>(reference_index)],
                                delta);
                            if (!sc_finite(delta) || !sc_finite(replay_total)) {
                                retry_without_bla = true;
                                break;
                            }
                            if (sc_outside_escape(replay_norm)
                                || sc_outside_escape_with_delta(
                                    context.orbit->scaled[static_cast<size_t>(reference_index)],
                                    delta)) {
                                output[index] = smooth_escape_scaled(iteration, replay_norm);
                                escaped = true;
                                break;
                            }
                        }
                        if (deadline_abort) break;
                        if (escaped) break;
                        if (retry_without_bla) {
                            delta = previous_delta;
                            reference_index = previous_reference_index;
                            iteration = previous_iteration;
                            if constexpr (CollectStats) {
                                ++stats->bla_retries;
                                if (!pixel_disable_bla) ++stats->bla_disabled_pixels;
                            }
                            pixel_disable_bla = true;
                            have_total = false;
                            continue;
                        }
                        total = replay_total;
                        total_norm = replay_norm;
                        have_total = true;
                    } else {
                        total = endpoint;
                        total_norm = endpoint_norm;
                        have_total = true;
                        if constexpr (CollectStats) {
                            stats->logical_iterations += static_cast<std::uint64_t>(map_length);
                        }
                    }
                    continue;
                }

                // One exact perturbation step.  The expression is written as
                // 2*Z*delta + delta^2 + delta_c, but the symmetric product
                // keeps the same operation count as a complex multiply.
                const ScaledComplex reference =
                    context.orbit->scaled[static_cast<size_t>(reference_index)];
                delta = sc_add(
                    sc_double(sc_mul(reference, delta)),
                    sc_add(sc_mul(delta, delta), dc));
                ++reference_index;
                ++iteration;
                if constexpr (CollectStats) {
                    ++stats->logical_iterations;
                    ++stats->exact_steps;
                }
            }
            if (!escaped && !deadline_abort && !unresolved_pixel) {
                output[index] = static_cast<float>(max_iter);
            }
            if constexpr (CollectStats) {
                stats->max_pixel_iterations = std::max(
                    stats->max_pixel_iterations,
                    static_cast<std::uint64_t>(std::max(0, iteration)));
            }
    }

    if constexpr (CollectStats) {
        RenderStats total;
        for (const RenderStats& local : thread_stats) {
            total.pixels += local.pixels;
            total.logical_iterations += local.logical_iterations;
            total.bla_blocks += local.bla_blocks;
            total.linear_blocks += local.linear_blocks;
            total.cubic_blocks += local.cubic_blocks;
            total.exact_steps += local.exact_steps;
            total.replay_steps += local.replay_steps;
            total.bla_retries += local.bla_retries;
            total.cycle_inside += local.cycle_inside;
            total.double_tail_pixels += local.double_tail_pixels;
            total.bla_disabled_pixels += local.bla_disabled_pixels;
            total.tail_steps += local.tail_steps;
            total.max_tail_steps = std::max(total.max_tail_steps, local.max_tail_steps);
            total.tail_rebases += local.tail_rebases;
            total.tail_rebase_fallbacks += local.tail_rebase_fallbacks;
            total.max_pixel_iterations = std::max(
                total.max_pixel_iterations,
                local.max_pixel_iterations);
            total.series_pixels += local.series_pixels;
            total.series_jumps += local.series_jumps;
            total.glitch_count += local.glitch_count;
            total.unresolved_pixels += local.unresolved_pixels;
            total.deadline_aborts += local.deadline_aborts;
            total.secondary_references += local.secondary_references;
            for (size_t index = 0; index < total.bla_length_histogram.size(); ++index) {
                total.bla_length_histogram[index] += local.bla_length_histogram[index];
            }
        }
        total.render_ns = static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - render_started).count());
        *stats_out = total;
    }
}

template <bool CollectStats>
void render_bla_dispatch(
    float* output,
    int width,
    int height,
    const char* zoom_text,
    const ReferenceContext& context,
    int max_iter,
    int threads,
    int series_order,
    int series_block,
    const FractalRenderOptions& options,
    RenderStats* stats_out,
    const std::vector<ScaledComplex>* point_offsets = nullptr,
    const FloatExp* point_radius = nullptr
) {
    const bool enable_cycle_detection = options.disable_cycle == 0
        && (options.strict == 0 || options.strict_cycle != 0);
    if (enable_cycle_detection) {
        render_bla_impl<CollectStats, true>(
            output, width, height, zoom_text, context, max_iter, threads,
            series_order, series_block, options, stats_out,
            point_offsets, point_radius);
    } else {
        render_bla_impl<CollectStats, false>(
            output, width, height, zoom_text, context, max_iter, threads,
            series_order, series_block, options, stats_out,
            point_offsets, point_radius);
    }
}

#endif

std::unique_ptr<ReferenceContext> create_reference_context(
    const char* x_center,
    const char* y_center,
    const char* viewport_zoom,
    int max_iter,
    int precision_bits,
    int series_order,
    bool retain_builder_orbit
) {
    if (!valid_c_string(x_center) || !valid_c_string(y_center)
        || !valid_c_string(viewport_zoom)) {
        throw std::runtime_error("native reference text is too long or null");
    }
    // Validate the decimal zoom before MPFR builds the orbit. Without this
    // early range check, an ABI caller could supply an enormous exponent and
    // make the reference setup spend time on a value the scaled renderer
    // cannot represent anyway.
#ifdef FRACTAL_HAVE_MPFR
    (void)parse_zoom_float_exp(
        viewport_zoom, static_cast<mpfr_prec_t>(precision_bits));
#else
    (void)parse_zoom(viewport_zoom);
#endif
    auto context = std::make_unique<ReferenceContext>();
    context->x_center = parse_coordinate(x_center, "real");
    context->y_center = parse_coordinate(y_center, "imaginary");
    context->requested_series_order = std::clamp(series_order, 8, 32);
    const auto reference_started = std::chrono::steady_clock::now();
    make_reference_orbit(
        *context, x_center, y_center, viewport_zoom, max_iter, precision_bits);
    context->reference_build_ns = static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - reference_started).count());
    const auto bla_started = std::chrono::steady_clock::now();
    build_bla(*context, retain_builder_orbit);
    context->bla_build_ns = static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - bla_started).count());
    return context;
}

#ifdef FRACTAL_HAVE_MPFR
std::unique_ptr<ReferenceContext> clone_reference_context(
    const ReferenceContext& source,
    const char* viewport_zoom
) {
    if (!source.orbit || source.fast_orbit.empty()) {
        throw std::runtime_error(
            "reference was not created as a reusable tier root");
    }
    const auto reference_started = std::chrono::steady_clock::now();
    auto context = std::make_unique<ReferenceContext>();
    context->orbit = source.orbit;
    context->bla.map_end = source.bla.map_end;
    context->requested_max_iter = source.requested_max_iter;
    context->requested_series_order = source.requested_series_order;
    context->x_center = source.x_center;
    context->y_center = source.y_center;
    context->precision_bits = source.precision_bits;

    // Only the BLA/series input domain changes between depth tiers.  The
    // expensive MPFR recurrence and compact render orbit remain shared.
    const FloatExp zoom = parse_zoom_float_exp(viewport_zoom, context->precision_bits);
    context->bla.input_radius = fe_div(FloatExp::from_parts(2.8, 0), zoom);
    context->reference_build_ns = static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - reference_started).count());
    const auto bla_started = std::chrono::steady_clock::now();
    build_retargeted_bla(*context, source);
    context->bla_build_ns = static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - bla_started).count());
    return context;
}
#endif

} // namespace

extern "C" {

int fractal_abi_version() { return ABI_VERSION; }

void fractal_set_stats_enabled(int enabled) {
    render_stats_enabled.store(enabled != 0, std::memory_order_relaxed);
}

int fractal_get_last_stats(std::uint64_t* values, int capacity) {
    return copy_render_stats(values, capacity);
}

int fractal_get_last_stats_ex(std::uint64_t* values, int capacity) {
    return copy_extended_render_stats(values, capacity);
}

int fractal_render_options_version() { return RENDER_OPTIONS_VERSION; }

void fractal_render_options_default(FractalRenderOptions* options) {
    if (options) *options = default_render_options();
}

int fractal_backend_capabilities() {
    try {
        int capabilities = 1; // scalar CPU is always available.
#if defined(__AVX2__)
        if (avx2_runtime_available()) capabilities |= 2;
#endif
#ifdef FRACTAL_HAVE_OPENCL
        if (opencl_available()) capabilities |= 4;
#endif
        set_error("");
        return capabilities;
    } catch (const std::exception& error) {
        set_error(error.what());
        return 1;
    } catch (...) {
        set_error("native backend capability query failed with an unknown exception");
        return 1;
    }
}

const char* fractal_last_error() {
    return last_error.c_str();
}

int fractal_colourise(
    const float* field,
    std::uint8_t* output,
    int width,
    int height,
    int max_iter,
    double phase,
    double vocal,
    double instrumental,
    double pitch,
    int threads
) {
    try {
        if (!field || !output || !valid_pixel_dimensions(width, height)
            || !valid_iteration_count(max_iter) || !valid_thread_count(threads)
            || !valid_colour_controls(phase, vocal, instrumental, pitch)) {
            throw std::runtime_error("invalid native colour dimensions or palette");
        }
        const AuroraPalette& palette = aurora_palette_for(
            max_iter, phase, vocal, instrumental, pitch);
        const int palette_size = static_cast<int>(palette.rgb.size());
        // Match the float32 indexing used by the Python fallback.  A double
        // product can occasionally select the adjacent 65k-entry palette
        // bin, which is visible as a several-level RGB difference.
        const float index_scale = static_cast<float>(palette_size - 1)
            / static_cast<float>(max_iter);
        const int pixel_count = width * height;

#ifdef _OPENMP
        if (threads > 0) omp_set_num_threads(threads);
#pragma omp parallel for schedule(static)
#endif
        for (int pixel = 0; pixel < pixel_count; ++pixel) {
            const float smooth = field[pixel];
            std::uint8_t* rgb = output + static_cast<size_t>(pixel) * 3U;
            if (!std::isfinite(smooth) || smooth >= static_cast<float>(max_iter) - 0.5F) {
                rgb[0] = 0;
                rgb[1] = 0;
                rgb[2] = 0;
                continue;
            }
            const double scaled_index = static_cast<double>(smooth)
                * static_cast<double>(index_scale);
            const int palette_index = scaled_index >= static_cast<double>(palette_size - 1)
                ? palette_size - 1
                : !std::isfinite(scaled_index) || scaled_index <= 0.0
                    ? 0
                    : static_cast<int>(scaled_index);
            const auto& colour = palette.rgb[static_cast<size_t>(palette_index)];
            rgb[0] = colour[0];
            rgb[1] = colour[1];
            rgb[2] = colour[2];
        }
        set_error("");
        return 0;
    } catch (const std::exception& error) {
        set_error(error.what());
        return 1;
    } catch (...) {
        set_error("native colouriser failed with an unknown exception");
        return 1;
    }
}

// Raw-field reprojection is intentionally separate from colourisation.  A
// GUI or an exp-map cache can therefore seek/recolour the same iteration data
// without rendering RGB keyframes again.
int fractal_crop_field(
    const float* source,
    int source_width,
    int source_height,
    float* output,
    int output_width,
    int output_height,
    double zoom_factor,
    int threads
) {
    try {
        if (!source || !output || !valid_pixel_dimensions(source_width, source_height)
            || !valid_pixel_dimensions(output_width, output_height)
            || !valid_thread_count(threads)
            || !std::isfinite(zoom_factor) || zoom_factor <= 0.0) {
            throw std::runtime_error("invalid native raw-field crop dimensions");
        }
        zoom_factor = std::max(zoom_factor, 1.0);
        const double inverse_zoom = 1.0 / zoom_factor;
        const double crop_width = static_cast<double>(source_width) * inverse_zoom;
        const double crop_height = static_cast<double>(source_height) * inverse_zoom;
        const double left = (static_cast<double>(source_width) - crop_width) * 0.5;
        const double top = (static_cast<double>(source_height) - crop_height) * 0.5;
        BilinearWorkspace& workspace = bilinear_workspace;
        BilinearAxis& x_axis = workspace.parent_x_axis;
        BilinearAxis& y_axis = workspace.parent_y_axis;
        fill_bilinear_axis(x_axis, source_width, output_width, zoom_factor);
        fill_bilinear_axis(y_axis, source_height, output_height, zoom_factor);
        // The axis helper is centred and zoom-aware; left/top are retained in
        // the explicit formula above to document the raw-field mapping and
        // keep this function's semantics aligned with crop_colourise.
        (void)left;
        (void)top;
#ifdef _OPENMP
        if (threads > 0) omp_set_num_threads(threads);
#pragma omp parallel for schedule(static)
#endif
        for (int output_y = 0; output_y < output_height; ++output_y) {
            for (int output_x = 0; output_x < output_width; ++output_x) {
                output[static_cast<size_t>(output_y) * output_width + output_x] =
                    sample_bilinear_mapped(
                        source,
                        source_width,
                        x_axis,
                        y_axis,
                        output_x,
                        output_y);
            }
        }
        set_error("");
        return 0;
    } catch (const std::exception& error) {
        set_error(error.what());
        return 1;
    } catch (...) {
        set_error("native raw-field crop failed with an unknown exception");
        return 1;
    }
}

int fractal_crop_colourise(
    const float* source,
    int source_width,
    int source_height,
    std::uint8_t* output,
    int output_width,
    int output_height,
    double zoom_factor,
    int max_iter,
    double phase,
    double vocal,
    double instrumental,
    double pitch,
    int threads
) {
    try {
        if (!source || !output || !valid_pixel_dimensions(source_width, source_height)
            || !valid_pixel_dimensions(output_width, output_height)
            || !valid_iteration_count(max_iter) || !valid_thread_count(threads)
            || !std::isfinite(zoom_factor) || zoom_factor <= 0.0
            || !valid_colour_controls(phase, vocal, instrumental, pitch)) {
            throw std::runtime_error("invalid native crop/colour dimensions or palette");
        }
        zoom_factor = std::max(zoom_factor, 1.0);
        const AuroraPalette& palette = aurora_palette_for(
            max_iter, phase, vocal, instrumental, pitch);
        const int palette_size = static_cast<int>(palette.rgb.size());
        const float index_scale = static_cast<float>(palette_size - 1)
            / static_cast<float>(max_iter);
        BilinearWorkspace& workspace = bilinear_workspace;
        BilinearAxis& x_axis = workspace.parent_x_axis;
        BilinearAxis& y_axis = workspace.parent_y_axis;
        fill_bilinear_axis(x_axis, source_width, output_width, zoom_factor);
        fill_bilinear_axis(y_axis, source_height, output_height, zoom_factor);

#ifdef _OPENMP
        if (threads > 0) omp_set_num_threads(threads);
#pragma omp parallel for schedule(static)
#endif
        for (int output_y = 0; output_y < output_height; ++output_y) {
            for (int output_x = 0; output_x < output_width; ++output_x) {
                bool inside = false;
                const float smooth = sample_bilinear_mapped_preserving_interior(
                    source,
                    source_width,
                    x_axis,
                    y_axis,
                    output_x,
                    output_y,
                    max_iter,
                    inside);
                std::uint8_t* rgb = output + static_cast<size_t>(
                    output_y * output_width + output_x) * 3U;
                if (inside) {
                    rgb[0] = 0;
                    rgb[1] = 0;
                    rgb[2] = 0;
                    continue;
                }
                const double scaled_index = static_cast<double>(smooth)
                    * static_cast<double>(index_scale);
                const int palette_index = scaled_index >= static_cast<double>(palette_size - 1)
                    ? palette_size - 1
                    : !std::isfinite(scaled_index) || scaled_index <= 0.0
                        ? 0
                        : static_cast<int>(scaled_index);
                const auto& colour = palette.rgb[static_cast<size_t>(palette_index)];
                rgb[0] = colour[0];
                rgb[1] = colour[1];
                rgb[2] = colour[2];
            }
        }
        set_error("");
        return 0;
    } catch (const std::exception& error) {
        set_error(error.what());
        return 1;
    } catch (...) {
        set_error("native crop colouriser failed with an unknown exception");
        return 1;
    }
}

int fractal_atlas_colourise(
    const float* parent,
    int parent_width,
    int parent_height,
    int parent_max_iter,
    const float* child,
    int child_width,
    int child_height,
    int child_max_iter,
    std::uint8_t* output,
    int output_width,
    int output_height,
    double parent_zoom,
    double child_fraction,
    int palette_max_iter,
    double phase,
    double vocal,
    double instrumental,
    double pitch,
    int threads
) {
    try {
        if (!parent || !output
            || !valid_pixel_dimensions(parent_width, parent_height)
            || !valid_pixel_dimensions(output_width, output_height)
            || !valid_iteration_count(parent_max_iter)
            || !valid_thread_count(threads)
            || !std::isfinite(parent_zoom) || parent_zoom <= 0.0
            || !std::isfinite(child_fraction)
            || child_fraction < 0.0 || child_fraction > 1.0
            || palette_max_iter < 0 || palette_max_iter > MAX_NATIVE_ITERATIONS
            || !valid_colour_controls(phase, vocal, instrumental, pitch)) {
            throw std::runtime_error("invalid native atlas dimensions or controls");
        }
        const bool use_child = child != nullptr && child_fraction > 0.0;
        if (use_child && (!valid_pixel_dimensions(child_width, child_height)
                          || !valid_iteration_count(child_max_iter))) {
            throw std::runtime_error("invalid native atlas child tile");
        }
        const int effective_child_iter = use_child ? child_max_iter : 0;
        const int effective_palette_iter = std::max(
            1,
            std::max(palette_max_iter, std::max(parent_max_iter, effective_child_iter)));
        const AuroraPalette& palette = aurora_palette_for(
            effective_palette_iter, phase, vocal, instrumental, pitch);
        const int effective_palette_size = static_cast<int>(palette.rgb.size());
        const float palette_index_scale = static_cast<float>(effective_palette_size - 1)
            / static_cast<float>(effective_palette_iter);
        const int visible_child_width = use_child
            ? std::max(1, static_cast<int>(std::lround(
                static_cast<double>(output_width) * child_fraction)))
            : 0;
        const int visible_child_height = use_child
            ? std::max(1, static_cast<int>(std::lround(
                static_cast<double>(output_height) * child_fraction)))
            : 0;
        const int child_left = (output_width - visible_child_width) / 2;
        const int child_top = (output_height - visible_child_height) / 2;
        const int feather = use_child
            ? std::min(16, std::min(visible_child_width / 8, visible_child_height / 8))
            : 0;
        const bool full_child = use_child && child_fraction >= 0.999999;

        // Coordinate generation is intentionally outside the pixel loop.
        // The old implementation recomputed two divisions, clamps, floors
        // and four source indices for every pixel. At 4K that dominated the
        // otherwise cheap colour pass. These compact maps are per-frame and
        // are reused for every row below.
        BilinearWorkspace& workspace = bilinear_workspace;
        BilinearAxis& parent_x_axis = workspace.parent_x_axis;
        BilinearAxis& parent_y_axis = workspace.parent_y_axis;
        if (!full_child) {
            fill_bilinear_axis(parent_x_axis, parent_width, output_width, parent_zoom);
            fill_bilinear_axis(parent_y_axis, parent_height, output_height, parent_zoom);
        }
        BilinearAxis& child_x_axis = workspace.child_x_axis;
        BilinearAxis& child_y_axis = workspace.child_y_axis;
        if (use_child) {
            const int child_destination_width = full_child ? output_width : visible_child_width;
            const int child_destination_height = full_child ? output_height : visible_child_height;
            fill_bilinear_axis(child_x_axis, child_width, child_destination_width, 1.0);
            fill_bilinear_axis(child_y_axis, child_height, child_destination_height, 1.0);
        }
        std::vector<float>& child_edge_x = workspace.child_edge_x;
        std::vector<float>& child_edge_y = workspace.child_edge_y;
        if (use_child && !full_child && feather >= 2) {
            child_edge_x.resize(static_cast<size_t>(visible_child_width));
            child_edge_y.resize(static_cast<size_t>(visible_child_height));
            for (int x = 0; x < visible_child_width; ++x) {
                const int edge = std::min(x, visible_child_width - 1 - x);
                child_edge_x[static_cast<size_t>(x)] =
                    std::min(1.0F, static_cast<float>(edge) / static_cast<float>(feather));
            }
            for (int y = 0; y < visible_child_height; ++y) {
                const int edge = std::min(y, visible_child_height - 1 - y);
                child_edge_y[static_cast<size_t>(y)] =
                    std::min(1.0F, static_cast<float>(edge) / static_cast<float>(feather));
            }
        }

        auto render_parent_span = [&](int output_y, int begin_x, int end_x) {
            for (int output_x = begin_x; output_x < end_x; ++output_x) {
                bool parent_inside = false;
                const float smooth = sample_bilinear_mapped_preserving_interior(
                    parent,
                    parent_width,
                    parent_x_axis,
                    parent_y_axis,
                    output_x,
                    output_y,
                    parent_max_iter,
                    parent_inside);
                std::uint8_t* destination = output + static_cast<size_t>(
                    output_y * output_width + output_x) * 3U;
                write_colour_pixel(
                    parent_inside
                        ? static_cast<float>(effective_palette_iter)
                        : smooth,
                    effective_palette_iter,
                    palette,
                    palette_index_scale,
                    destination);
            }
        };

#ifdef _OPENMP
        if (threads > 0) omp_set_num_threads(threads);
#pragma omp parallel for schedule(static)
#endif
        for (int output_y = 0; output_y < output_height; ++output_y) {
            if (full_child) {
                for (int output_x = 0; output_x < output_width; ++output_x) {
                    bool child_inside = false;
                    const float smooth = sample_bilinear_mapped_preserving_interior(
                        child,
                        child_width,
                        child_x_axis,
                        child_y_axis,
                        output_x,
                        output_y,
                        child_max_iter,
                        child_inside);
                    std::uint8_t* destination = output + static_cast<size_t>(
                        output_y * output_width + output_x) * 3U;
                    write_colour_pixel(
                        child_inside
                            ? static_cast<float>(effective_palette_iter)
                            : smooth,
                        effective_palette_iter,
                        palette,
                        palette_index_scale,
                        destination);
                }
                continue;
            }

            if (!use_child || output_y < child_top || output_y >= child_top + visible_child_height) {
                render_parent_span(output_y, 0, output_width);
                continue;
            }

            render_parent_span(output_y, 0, child_left);
            const int child_right = child_left + visible_child_width;
            for (int output_x = child_left; output_x < child_right; ++output_x) {
                const int child_x = output_x - child_left;
                const int child_y = output_y - child_top;
                bool child_inside = false;
                const float child_smooth = sample_bilinear_mapped_preserving_interior(
                    child,
                    child_width,
                    child_x_axis,
                    child_y_axis,
                    child_x,
                    child_y,
                    child_max_iter,
                    child_inside);
                const float alpha = feather >= 2
                    ? std::min(
                        child_edge_x[static_cast<size_t>(child_x)],
                        child_edge_y[static_cast<size_t>(child_y)])
                    : 1.0F;
                std::uint8_t* destination = output + static_cast<size_t>(
                    output_y * output_width + output_x) * 3U;
                // Away from the feather band the child completely replaces
                // the parent.  Avoid sampling the parent for those pixels:
                // the deeper child is authoritative for both escape and
                // interior classification in its visible region.
                if (alpha >= 0.999999F) {
                    write_colour_pixel(
                        child_inside
                            ? static_cast<float>(effective_palette_iter)
                            : child_smooth,
                        effective_palette_iter,
                        palette,
                        palette_index_scale,
                        destination);
                    continue;
                }
                bool parent_inside = false;
                const float parent_smooth = sample_bilinear_mapped_preserving_interior(
                    parent,
                    parent_width,
                    parent_x_axis,
                    parent_y_axis,
                    output_x,
                    output_y,
                    parent_max_iter,
                    parent_inside);
                // The deeper child is authoritative inside its visible
                // rectangle. Falling back to an escaped parent value when
                // only the child is interior creates a hard rectangular fill.
                const float blended_smooth = child_inside
                    ? static_cast<float>(effective_palette_iter)
                    : parent_inside
                        ? child_smooth
                        : parent_smooth * (1.0F - alpha) + child_smooth * alpha;
                // Blend the scalar iteration field before colourisation. RGB
                // blending mixed two independently indexed hues and produced
                // a visible seam whenever adjacent tiles had different
                // iteration budgets.
                write_colour_pixel(
                    blended_smooth,
                    effective_palette_iter,
                    palette,
                    palette_index_scale,
                    destination);
            }
            render_parent_span(output_y, child_right, output_width);
        }
        set_error("");
        return 0;
    } catch (const std::exception& error) {
        set_error(error.what());
        return 1;
    } catch (...) {
        set_error("native atlas colouriser failed with an unknown exception");
        return 1;
    }
}

void* fractal_create_reference(
    const char* x_center,
    const char* y_center,
    const char* viewport_zoom,
    int max_iter,
    int precision_bits,
    int series_order
) {
    try {
        if (!x_center || !y_center || !viewport_zoom
            || !valid_iteration_count(max_iter)
            || !valid_precision_bits(precision_bits)
            || !valid_series_parameters(series_order, 2)) {
            throw std::runtime_error("invalid reference configuration");
        }
        auto context = create_reference_context(
            x_center, y_center, viewport_zoom, max_iter, precision_bits,
            series_order, false);
        set_error("");
        return register_reference(std::move(context));
    } catch (const std::exception& error) {
        set_error(error.what());
        return nullptr;
    } catch (...) {
        set_error("native reference creation failed with an unknown exception");
        return nullptr;
    }
}

void* fractal_create_reference_reusable(
    const char* x_center,
    const char* y_center,
    const char* viewport_zoom,
    int max_iter,
    int precision_bits,
    int series_order
) {
    try {
        if (!x_center || !y_center || !viewport_zoom
            || !valid_iteration_count(max_iter)
            || !valid_precision_bits(precision_bits)
            || !valid_series_parameters(series_order, 2)) {
            throw std::runtime_error("invalid reusable reference configuration");
        }
        auto context = create_reference_context(
            x_center, y_center, viewport_zoom, max_iter, precision_bits,
            series_order, true);
        set_error("");
        return register_reference(std::move(context));
    } catch (const std::exception& error) {
        set_error(error.what());
        return nullptr;
    } catch (...) {
        set_error("native reusable reference creation failed with an unknown exception");
        return nullptr;
    }
}

void* fractal_clone_reference(void* source_handle, const char* viewport_zoom) {
    try {
        if (!source_handle || !viewport_zoom) {
            throw std::runtime_error("invalid reference tier clone configuration");
        }
        const auto source = acquire_reference(source_handle);
        if (!source) {
            throw std::runtime_error("invalid or already-destroyed reference handle");
        }
#ifdef FRACTAL_HAVE_MPFR
        auto context = clone_reference_context(
            *source, viewport_zoom);
        set_error("");
        return register_reference(std::move(context));
#else
        throw std::runtime_error("deep reference tier cloning requires MPFR/GMP");
#endif
    } catch (const std::exception& error) {
        set_error(error.what());
        return nullptr;
    } catch (...) {
        set_error("native reference clone failed with an unknown exception");
        return nullptr;
    }
}

void fractal_destroy_reference(void* handle) {
    try {
        if (!handle) {
            set_error("");
            return;
        }
        const auto context = remove_reference(handle);
        if (!context) {
            set_error("invalid or already-destroyed reference handle");
            return;
        }
        set_error("");
    } catch (const std::exception& error) {
        set_error(error.what());
    } catch (...) {
        set_error("native reference destruction failed with an unknown exception");
    }
}

int fractal_get_reference_stats(
    void* handle,
    std::uint64_t* values,
    int capacity
) {
    try {
        if (!values || capacity < 5) {
            set_error("reference statistics buffer is too small or null");
            return -1;
        }
        const auto context = acquire_reference(handle);
        if (!context) {
            set_error("invalid or already-destroyed reference handle");
            return -1;
        }
        values[0] = context->reference_build_ns;
        values[1] = context->series_build_ns;
        values[2] = context->bla_build_ns;
        values[3] = context->image_series.enabled
            ? static_cast<std::uint64_t>(context->image_series.iteration)
            : 0;
        values[4] = context->image_series.enabled
            ? static_cast<std::uint64_t>(context->image_series.order)
            : 0;
        set_error("");
        return 5;
    } catch (const std::exception& error) {
        set_error(error.what());
        return -1;
    } catch (...) {
        set_error("reference statistics lookup failed with an unknown exception");
        return -1;
    }
}

int fractal_render_mandelbrot_reference_ex(
    float* output,
    int width,
    int height,
    const char* zoom_text,
    void* handle,
    int max_iter,
    int threads,
    int series_order,
    int series_block,
    const FractalRenderOptions* supplied_options
) {
    try {
        if (!output || !zoom_text || !handle || !valid_pixel_dimensions(width, height)
            || !valid_iteration_count(max_iter) || !valid_thread_count(threads)
            || !valid_series_parameters(series_order, series_block)) {
            throw std::runtime_error("invalid native render dimensions or handle");
        }
        const FractalRenderOptions options = checked_render_options(supplied_options);
        if (options.backend != 0 && options.backend != 1 && options.backend != 2) {
            throw std::runtime_error("unknown native render backend");
        }
        const auto context = acquire_reference(handle);
        if (!context) {
            throw std::runtime_error("invalid or already-destroyed reference handle");
        }
        if (max_iter > context->requested_max_iter) {
            throw std::runtime_error("render iteration count exceeds prepared reference");
        }
        // series_order selects the active polynomial degree.  Values above
        // three remain accepted for ABI compatibility and are clamped by the
        // renderer because this table stores terms through degree three.
        (void)series_order;
        (void)series_block;
        long double zoom_log10 = 0.0L;
#ifdef FRACTAL_HAVE_MPFR
        const FloatExp zoom_float_exp = parse_zoom_float_exp(zoom_text, context->precision_bits);
        zoom_log10 = fe_log(zoom_float_exp) / std::log(10.0L);
#else
        const long double zoom = parse_zoom(zoom_text);
        zoom_log10 = std::log10(zoom);
#endif
        if (zoom_log10 >= 6.0L) {
#ifdef FRACTAL_HAVE_MPFR
            if (options.backend == 2) {
                throw std::runtime_error(
                    "OpenCL backend currently supports only direct zooms below 1e6; "
                    "deep perturbation remains on the validated CPU backend");
            }
            RenderStats stats;
            if (render_stats_enabled.load(std::memory_order_relaxed)) {
                render_bla_dispatch<true>(output, width, height, zoom_text, *context, max_iter,
                                          threads, series_order, series_block, options, &stats);
                publish_render_stats(stats);
            } else {
                render_bla_dispatch<false>(output, width, height, zoom_text, *context, max_iter,
                                           threads, series_order, series_block, options, nullptr);
            }
#else
            (void)zoom;
            throw std::runtime_error("deep rendering requires MPFR/GMP; rebuild with make");
#endif
        } else {
#ifdef FRACTAL_HAVE_MPFR
            const long double zoom = parse_zoom(zoom_text);
#endif
            render_direct(
                output,
                width,
                height,
                zoom,
                context->x_center,
                context->y_center,
                max_iter,
                threads,
                options.backend);
        }
        set_error("");
        return 0;
    } catch (const std::exception& error) {
        set_error(error.what());
        return 1;
    } catch (...) {
        set_error("native reference render failed with an unknown exception");
        return 1;
    }
}

// Render an arbitrary list of scaled perturbations.  The Python exp-map
// layer uses this to sample (log radius, angle) coordinates directly; the
// numerical core and its validated series/BLA machinery remain shared with
// rectangular atlas tiles.
int fractal_render_points(
    float* output,
    int point_count,
    const char* zoom_text,
    const double* real_mantissa,
    const double* imag_mantissa,
    const std::int32_t* exponents,
    void* handle,
    int max_iter,
    int threads,
    int series_order,
    int series_block,
    const FractalRenderOptions* supplied_options
) {
    try {
        if (!output || point_count <= 0 || point_count > MAX_NATIVE_POINTS || !zoom_text
            || !real_mantissa || !imag_mantissa || !exponents || !handle) {
            throw std::runtime_error("invalid native point-render arguments");
        }
        if (!valid_iteration_count(max_iter) || !valid_thread_count(threads)
            || !valid_series_parameters(series_order, series_block)) {
            throw std::runtime_error("invalid native point-render limits");
        }
        const FractalRenderOptions options = checked_render_options(supplied_options);
        if (options.backend != 0 && options.backend != 1) {
            throw std::runtime_error("unknown native render backend");
        }
        const auto context = acquire_reference(handle);
        if (!context) {
            throw std::runtime_error("invalid or already-destroyed reference handle");
        }
        if (max_iter > context->requested_max_iter) {
            throw std::runtime_error("point render iteration count exceeds prepared reference");
        }
        // Reject malformed zoom text before copying a potentially large point
        // array into native memory.
#ifdef FRACTAL_HAVE_MPFR
        (void)parse_zoom_float_exp(zoom_text, context->precision_bits);
#else
        (void)parse_zoom(zoom_text);
#endif
        std::vector<ScaledComplex> points;
        points.reserve(static_cast<size_t>(point_count));
        ScaledNorm maximum_radius{};
        for (int index = 0; index < point_count; ++index) {
            ScaledComplex point{
                real_mantissa[index],
                imag_mantissa[index],
                exponents[index],
            };
            if (!sc_finite(point)) throw std::runtime_error("non-finite point offset");
            point.normalize();
            points.push_back(point);
            const ScaledNorm radius = sc_norm_squared(point);
            if (sc_compare_norm(radius, maximum_radius) > 0) maximum_radius = radius;
        }
#ifdef FRACTAL_HAVE_MPFR
        const FloatExp point_radius_squared{
            maximum_radius.mantissa,
            maximum_radius.exponent,
        };
        const FloatExp point_radius = fe_sqrt(point_radius_squared);
        RenderStats stats;
        if (render_stats_enabled.load(std::memory_order_relaxed)) {
            render_bla_dispatch<true>(
                output,
                point_count,
                1,
                zoom_text,
                *context,
                max_iter,
                threads,
                series_order,
                series_block,
                options,
                &stats,
                &points,
                &point_radius);
            publish_render_stats(stats);
        } else {
            render_bla_dispatch<false>(
                output,
                point_count,
                1,
                zoom_text,
                *context,
                max_iter,
                threads,
                series_order,
                series_block,
                options,
                nullptr,
                &points,
                &point_radius);
        }
        set_error("");
        return 0;
#else
        throw std::runtime_error("point rendering requires MPFR/GMP; rebuild with make");
#endif
    } catch (const std::exception& error) {
        set_error(error.what());
        return 1;
    } catch (...) {
        set_error("native point render failed with an unknown exception");
        return 1;
    }
}

// Stable compatibility entry point.  New callers should use the `_ex`
// variant so every render has explicit, per-call options.  Keeping this
// wrapper preserves the old C ABI shape for small external experiments.
int render_mandelbrot_reference(
    float* output,
    int width,
    int height,
    const char* zoom_text,
    void* handle,
    int max_iter,
    int threads,
    int series_order,
    int series_block
) {
    return fractal_render_mandelbrot_reference_ex(
        output,
        width,
        height,
        zoom_text,
        handle,
        max_iter,
        threads,
        series_order,
        series_block,
        nullptr);
}

// Versioned one-shot entry point for all supported formulas. Alternate
// formulas use the direct scalar path; the reusable MPFR/BLA reference
// machinery remains deliberately specific to the Mandelbrot parameter plane.
int render_fractal_ex(
    float* output,
    int width,
    int height,
    const char* zoom_text,
    const char* x_center,
    const char* y_center,
    int max_iter,
    int precision_bits,
    int use_perturbation,
    int threads,
    int formula,
    double julia_real,
    double julia_imag,
    const FractalRenderOptions* supplied_options
) {
    try {
        if (!output || !zoom_text || !x_center || !y_center
            || !valid_pixel_dimensions(width, height)
            || !valid_iteration_count(max_iter)
            || !valid_precision_bits(precision_bits)
            || use_perturbation < 0 || use_perturbation > 1
            || !valid_thread_count(threads)
            || !valid_formula(formula)
            || !std::isfinite(julia_real) || !std::isfinite(julia_imag)) {
            throw std::runtime_error("invalid native render dimensions, formula, or argument");
        }
        const FractalRenderOptions options = checked_render_options(supplied_options);
        if (options.backend != 0 && options.backend != 1 && options.backend != 2) {
            throw std::runtime_error("unknown native render backend");
        }
        if (formula != FRACTAL_FORMULA_MANDELBROT && use_perturbation) {
            throw std::runtime_error(
                "native deep perturbation is currently Mandelbrot-only; "
                "alternate formulas use the Python high-precision fallback");
        }
        if (!use_perturbation) {
            const long double zoom = parse_zoom(zoom_text);
            render_direct(
                output,
                width,
                height,
                zoom,
                parse_coordinate(x_center, "real"),
                parse_coordinate(y_center, "imaginary"),
                max_iter,
                threads,
                options.backend,
                formula,
                julia_real,
                julia_imag);
        } else {
            if (options.backend == 2) {
                throw std::runtime_error(
                    "OpenCL backend is only valid for direct one-shot renders");
            }
            void* context_handle = fractal_create_reference(
                x_center, y_center, zoom_text, max_iter, precision_bits, 8);
            if (!context_handle) throw std::runtime_error(last_error);
            const int status = fractal_render_mandelbrot_reference_ex(
                output,
                width,
                height,
                zoom_text,
                context_handle,
                max_iter,
                threads,
                8,
                32,
                &options);
            const std::string render_error = last_error;
            fractal_destroy_reference(context_handle);
            if (status != 0) throw std::runtime_error(render_error);
        }
        set_error("");
        return 0;
    } catch (const std::exception& error) {
        set_error(error.what());
        return 1;
    } catch (...) {
        set_error("native render failed with an unknown exception");
        return 1;
    }
}

// Versioned Mandelbrot compatibility entry point.  New callers should use
// render_fractal_ex when they need an alternate formula.
int render_mandelbrot_ex(
    float* output,
    int width,
    int height,
    const char* zoom_text,
    const char* x_center,
    const char* y_center,
    int max_iter,
    int precision_bits,
    int use_perturbation,
    int threads,
    const FractalRenderOptions* supplied_options
) {
    return render_fractal_ex(
        output,
        width,
        height,
        zoom_text,
        x_center,
        y_center,
        max_iter,
        precision_bits,
        use_perturbation,
        threads,
        FRACTAL_FORMULA_MANDELBROT,
        0.0,
        0.0,
        supplied_options);
}

// Compatibility entry point for shallow one-off renders and old callers.
int render_mandelbrot(
    float* output,
    int width,
    int height,
    const char* zoom_text,
    const char* x_center,
    const char* y_center,
    int max_iter,
    int precision_bits,
    int use_perturbation,
    int threads
) {
    return render_mandelbrot_ex(
        output,
        width,
        height,
        zoom_text,
        x_center,
        y_center,
        max_iter,
        precision_bits,
        use_perturbation,
        threads,
        nullptr);
}

} // extern "C"
