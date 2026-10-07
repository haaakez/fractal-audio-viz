// Mandelbrot renderer with reusable MPFR reference orbits, scaled
// mantissa/exponent perturbation, and hierarchical BLA maps. The exported
// functions intentionally use a C ABI so Python can drive the native core
// without knowing its C++ types.

#include <algorithm>
#include <atomic>
#include <array>
#include <cstdint>
#include <cmath>
#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <locale>
#include <stdexcept>
#include <string>
#include <sstream>
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
// MPFR direct recovery is deliberately restricted to bounded point cells. A
// large unresolved region must be subdivided by the atlas caller first; doing
// an exact arbitrary-precision iteration for every pixel in a frame-sized
// point request would turn a bounded repair into an accidental full render.
// Keep this in step with the atlas-side cell limit so the native fast path is
// available for every cell the caller deliberately permits.
constexpr std::size_t MAX_EXACT_POINT_REPAIR_PIXELS = 8192;
constexpr int MAX_SAFE_DEEP_LINEAR_BLA_LENGTH = 1024;
constexpr int ESCAPE_RADIUS_MODE_CLASSIC = 0;
constexpr int ESCAPE_RADIUS_MODE_KALLES_HIGH = 1;
constexpr int COORDINATE_MODE_PROJECT = FRACTAL_COORDINATE_MODE_PROJECT;
constexpr int COORDINATE_MODE_KALLES = FRACTAL_COORDINATE_MODE_KALLES;
constexpr long double ESCAPE_RADIUS_SQUARED = 4.0L;
constexpr long double KALLES_HIGH_BAILOUT_RADIUS = 10000.0L;
constexpr long double KALLES_HIGH_BAILOUT_SQUARED =
    KALLES_HIGH_BAILOUT_RADIUS * KALLES_HIGH_BAILOUT_RADIUS;
constexpr long double LOG_TWO = 0.693147180559945309417232121458176568L;
constexpr long double LOG_TEN = 2.302585092994045684017991454684364208L;
constexpr int MAX_NATIVE_ITERATIONS = 10'000'000;
constexpr int MAX_NATIVE_THREADS = 4096;
constexpr int MAX_NATIVE_PRECISION_BITS = 131'072;
constexpr int MAX_NATIVE_PIXELS = 100'000'000;
constexpr int MAX_NATIVE_POINTS = 100'000'000;
constexpr std::size_t MAX_NATIVE_TEXT_LENGTH = 50'000;
constexpr int KFP_ATLAS_SEAM_FEATHER_MAX = 64;
constexpr int KFP_ATLAS_SEAM_FEATHER_DIVISOR = 2;
constexpr long double MIN_NATIVE_LOG10_ZOOM = -300.0L;
constexpr long double MAX_NATIVE_LOG10_ZOOM = 9800.0L;

inline int kfp_atlas_seam_feather(int width, int height) noexcept {
    const int minimum = std::min(width, height);
    if (minimum <= 0) return 0;
    const int requested = std::max(
        2,
        std::min(
            KFP_ATLAS_SEAM_FEATHER_MAX,
            minimum / KFP_ATLAS_SEAM_FEATHER_DIVISOR));
    // Leave a fully-owned sample in the middle of even child rectangles.
    // With minimum=32 the greatest edge distance is 15, so a feather of 16
    // would blend the whole child and hide its interior sentinel.
    const int plateau_limit = std::max(0, (minimum - 1) / 2);
    return std::min(requested, plateau_limit);
}

// Store a large iteration field relative to a nearby bias when the caller is
// going to quantise it to float32.  Keeping the fractional part close to zero
// avoids losing Kalles' smooth escape transition once the absolute iteration
// count reaches tens of thousands.  A zero bias is exactly the historical
// representation.
inline float encode_render_value(long double value, double output_bias) noexcept {
    return static_cast<float>(value - static_cast<long double>(output_bias));
}

inline float encode_render_iteration(int iteration, double output_bias) noexcept {
    return encode_render_value(static_cast<long double>(iteration), output_bias);
}

bool valid_formula(int formula) noexcept {
    return formula >= FRACTAL_FORMULA_MANDELBROT
        && formula <= FRACTAL_FORMULA_TRICORN;
}

bool valid_escape_radius_mode(int mode) noexcept {
    return mode == ESCAPE_RADIUS_MODE_CLASSIC
        || mode == ESCAPE_RADIUS_MODE_KALLES_HIGH;
}

bool valid_coordinate_mode(int mode) noexcept {
    return mode == COORDINATE_MODE_PROJECT
        || mode == COORDINATE_MODE_KALLES;
}

inline double pixel_axis_offset(int index, int dimension, int coordinate_mode) noexcept {
    // CFraktalSFT::GetPixelCoordinates uses `i - m_nX / 2` and
    // `j - m_nY / 2`; both divisions are integer divisions because the
    // dimensions are ints. The project mode retains the historical
    // pixel-centred `(dimension - 1) / 2` convention.
    const double origin = coordinate_mode == COORDINATE_MODE_KALLES
        ? static_cast<double>(dimension / 2)
        : static_cast<double>(dimension - 1) / 2.0;
    return static_cast<double>(index) - origin;
}

inline double viewport_height_factor(int coordinate_mode) noexcept {
    // FraktalSFT stores Zoom as 2 / m_ZoomRadius and derives pixel spacing as
    // (m_ZoomRadius * 2) / height. Therefore a Kalles view spans 4 / Zoom
    // vertically. The project renderer's historical view remains 2.8 / Zoom.
    return coordinate_mode == COORDINATE_MODE_KALLES ? 4.0 : 2.8;
}

inline long double escape_radius_squared_for_mode(int mode) noexcept {
    return mode == ESCAPE_RADIUS_MODE_KALLES_HIGH
        ? KALLES_HIGH_BAILOUT_SQUARED
        : ESCAPE_RADIUS_SQUARED;
}

inline double escape_radius_squared_double(int mode) noexcept {
    return static_cast<double>(escape_radius_squared_for_mode(mode));
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

bool parse_classic_long_double(const char* text, long double& value) {
    if (!valid_c_string(text)) return false;

    // The GTK process inherits the user's desktop locale.  std::strtold()
    // follows LC_NUMERIC, so a locale using a comma decimal separator rejects
    // the ASCII coordinate strings exported by the Python/GUI side.  Keep the
    // wire format locale-neutral by parsing with the classic C++ locale on a
    // local stream instead of changing the process-global locale.
    std::istringstream stream(text);
    stream.imbue(std::locale::classic());
    stream >> value;
    if (stream.fail()) return false;
    stream >> std::ws;
    return stream.eof();
}

long double parse_coordinate(const char* text, const char* label) {
    if (!valid_c_string(text) || !label) {
        throw std::runtime_error("native coordinate text is too long or null");
    }
    long double value = 0.0L;
    if (!parse_classic_long_double(text, value) || !std::isfinite(value)) {
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

template<int Formula>
inline void iterate_direct_formula_static(
    double zr,
    double zi,
    double parameter_real,
    double parameter_imag,
    double& next_real,
    double& next_imag
) noexcept;

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

/*
 * Direct orbit step plus the first-order derivative used by Kalles' analytic
 * distance/slope path.  For a parameter-plane render the derivative is with
 * respect to c; for Julia it is with respect to the pixel's initial z.
 * Burning Ship and Tricorn are not holomorphic, so the real Jacobian is the
 * useful conservative analogue of the complex derivative there.
 *
 * Formula and parameter-plane status are compile-time values here. The old
 * version accepted both as runtime arguments and paid for two predictable
 * formula branches on every orbit iteration. Keeping the arithmetic in the
 * same branches (including the volatile alternate-map products) preserves
 * the reference values while letting the compiler specialize the hot loop.
 */
template<int Formula, bool ParameterPlane, bool NeedDerivative>
inline void iterate_direct_formula_with_derivative(
    double zr,
    double zi,
    double parameter_real,
    double parameter_imag,
    double derivative_real,
    double derivative_imag,
    double& next_real,
    double& next_imag,
    double& next_derivative_real,
    double& next_derivative_imag
) noexcept {
    if constexpr (!NeedDerivative) {
        iterate_direct_formula_static<Formula>(
            zr,
            zi,
            parameter_real,
            parameter_imag,
            next_real,
            next_imag);
        next_derivative_real = 0.0;
        next_derivative_imag = 0.0;
        return;
    }
    const double add_real = ParameterPlane ? 1.0 : 0.0;
    const double add_imag = ParameterPlane ? 0.0 : 0.0;
    if constexpr (Formula == FRACTAL_FORMULA_BURNING_SHIP) {
        const double absolute_real = std::abs(zr);
        const double absolute_imag = std::abs(zi);
        // Keep the piecewise alternate maps reproducible with the NumPy
        // fallback. A fused multiply-add changes a late orbit by a few ulps
        // and can move a boundary pixel to the other side of escape.
        const volatile double real_square = absolute_real * absolute_real;
        const volatile double imag_square = absolute_imag * absolute_imag;
        const volatile double cross = 2.0 * absolute_real * absolute_imag;
        const volatile double real_difference = real_square - imag_square;
        const volatile double imaginary_product = cross + parameter_imag;
        next_real = real_difference + parameter_real;
        next_imag = imaginary_product;
        const double sign_real = zr > 0.0 ? 1.0 : zr < 0.0 ? -1.0 : 0.0;
        const double sign_imag = zi > 0.0 ? 1.0 : zi < 0.0 ? -1.0 : 0.0;
        const double jacobian00 = 2.0 * absolute_real * sign_real;
        const double jacobian01 = -2.0 * absolute_imag * sign_imag;
        const double jacobian10 = 2.0 * absolute_imag * sign_real;
        const double jacobian11 = 2.0 * absolute_real * sign_imag;
        next_derivative_real = jacobian00 * derivative_real
            + jacobian01 * derivative_imag + add_real;
        next_derivative_imag = jacobian10 * derivative_real
            + jacobian11 * derivative_imag + add_imag;
        return;
    }
    if constexpr (Formula == FRACTAL_FORMULA_TRICORN) {
        const volatile double real_square = zr * zr;
        const volatile double imag_square = zi * zi;
        const volatile double cross = -2.0 * zr * zi;
        const volatile double real_difference = real_square - imag_square;
        const volatile double imaginary_product = cross + parameter_imag;
        next_real = real_difference + parameter_real;
        next_imag = imaginary_product;
        next_derivative_real = 2.0 * zr * derivative_real
            - 2.0 * zi * derivative_imag + add_real;
        next_derivative_imag = -2.0 * zi * derivative_real
            - 2.0 * zr * derivative_imag + add_imag;
        return;
    }
    if constexpr (Formula == FRACTAL_FORMULA_JULIA) {
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
    next_derivative_real = 2.0 * zr * derivative_real
        - 2.0 * zi * derivative_imag + add_real;
    next_derivative_imag = 2.0 * zr * derivative_imag
        + 2.0 * zi * derivative_real + add_imag;
}

#ifdef FRACTAL_HAVE_OPENCL

// OpenCL reports collection sizes through the driver. Treat those values as
// untrusted metadata: a broken ICD must not turn capability probing into an
// arbitrarily large host allocation or diagnostic string.
constexpr cl_uint MAX_OPENCL_PLATFORMS = 256;
constexpr cl_uint MAX_OPENCL_DEVICES = 256;
constexpr size_t MAX_OPENCL_INFO_BYTES = 1U << 20;

// OpenCL handles ordinary direct fields and the shared scaled perturbation
// path. KFP plane output remains on the exact scalar implementation; the
// alternate deep formulas use the same device recurrence without pretending
// that their real-Jacobian BLA tables are interchangeable with Mandelbrot's.
constexpr const char* OPENCL_DIRECT_KERNEL = R"CLC(
#if defined(cl_khr_fp64)
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#elif defined(cl_amd_fp64)
#pragma OPENCL EXTENSION cl_amd_fp64 : enable
#endif

#if !defined(FRACTAL_OPENCL_FAST_MATH)
#pragma OPENCL FP_CONTRACT OFF
#endif

__kernel void mandelbrot_direct(
    __global float* output,
    const int width,
    const int height,
    const double center_real,
    const double center_imag,
    const double width_span,
    const double height_span,
    const int max_iter,
    const double output_bias,
    const int formula,
    const double julia_real,
    const double julia_imag,
    const double escape_squared,
    const int coordinate_mode
) {
    const size_t pixel = get_global_id(0);
    const size_t count = (size_t)width * (size_t)height;
    if (pixel >= count) return;

    const int py = (int)(pixel / (size_t)width);
    const int px = (int)(pixel - (size_t)py * (size_t)width);
    // Kalles uses integer half-dimensions; the historical project viewport
    // is pixel-centred. Keep both coordinate contracts on the GPU so the
    // OpenCL field is suitable for every ordinary (non-plane) profile.
    const double x_origin = coordinate_mode != 0
        ? (double)(width / 2) : (double)(width - 1) * 0.5;
    const double y_origin = coordinate_mode != 0
        ? (double)(height / 2) : (double)(height - 1) * 0.5;
    const double cx = center_real
        + ((double)px - x_origin) * width_span / (double)width;
    const double cy = center_imag
        + (y_origin - (double)py) * height_span / (double)height;

    if (formula == 0) {
        const double q = (cx - 0.25) * (cx - 0.25) + cy * cy;
        const int in_cardioid = q * (q + cx - 0.25) <= 0.25 * cy * cy;
        const int in_bulb = (cx + 1.0) * (cx + 1.0) + cy * cy <= 0.0625;
        if (in_cardioid || in_bulb) {
            output[pixel] = (float)((double)max_iter - output_bias);
            return;
        }
    }

    double zr = formula == 1 ? cx : 0.0;
    double zi = formula == 1 ? cy : 0.0;
    const double parameter_real = formula == 1 ? julia_real : cx;
    const double parameter_imag = formula == 1 ? julia_imag : cy;
    int iteration = 0;
    for (; iteration < max_iter; ++iteration) {
        const double source_real = formula == 2 ? fabs(zr) : zr;
        const double source_imag = formula == 2 ? fabs(zi) : zi;
        const double next_real = source_real * source_real - source_imag * source_imag
            + parameter_real;
        const double next_imag = (formula == 3 ? -2.0 : 2.0)
            * source_real * source_imag + parameter_imag;
        zr = next_real;
        zi = next_imag;
        const double magnitude_squared = zr * zr + zi * zi;
        if (magnitude_squared > escape_squared || !isfinite(magnitude_squared)) {
            const double safe_squared = isfinite(magnitude_squared)
                ? fmax(magnitude_squared, escape_squared + 1.0e-7)
                : 1.7976931348623157e308;
            const double magnitude = sqrt(safe_squared);
            output[pixel] = (float)((double)(iteration + 1)
                - log(log(magnitude)) / log(2.0) - output_bias);
            return;
        }
    }
    output[pixel] = (float)((double)max_iter - output_bias);
}

)CLC";

// The compatibility kernel above accepts a formula id so older callers can
// keep one ABI. Production callers know the formula before launch, though,
// so give OpenCL's JIT a literal formula through these wrappers. The helper
// is inlined by the device compiler and the predictable alternate-formula
// branches disappear from the hot orbit loop. Keep the argument lists
// identical to the compatibility kernel: selecting a specialized entry
// point remains a host-side detail and the public ABI does not change.
constexpr const char* OPENCL_SPECIALIZED_DIRECT_KERNEL = R"CLC(
inline void direct_render_pixel_specialized(
    __global float* output,
    const size_t pixel,
    const int width,
    const int height,
    const double center_real,
    const double center_imag,
    const double width_span,
    const double height_span,
    const int max_iter,
    const double output_bias,
    const int formula,
    const double julia_real,
    const double julia_imag,
    const double escape_squared,
    const int coordinate_mode
) {
    const int py = (int)(pixel / (size_t)width);
    const int px = (int)(pixel - (size_t)py * (size_t)width);
    const double x_origin = coordinate_mode != 0
        ? (double)(width / 2) : (double)(width - 1) * 0.5;
    const double y_origin = coordinate_mode != 0
        ? (double)(height / 2) : (double)(height - 1) * 0.5;
    const double cx = center_real
        + ((double)px - x_origin) * width_span / (double)width;
    const double cy = center_imag
        + (y_origin - (double)py) * height_span / (double)height;

    if (formula == 0) {
        const double q = (cx - 0.25) * (cx - 0.25) + cy * cy;
        const int in_cardioid = q * (q + cx - 0.25) <= 0.25 * cy * cy;
        const int in_bulb = (cx + 1.0) * (cx + 1.0) + cy * cy <= 0.0625;
        if (in_cardioid || in_bulb) {
            output[pixel] = (float)((double)max_iter - output_bias);
            return;
        }
    }

    double zr = formula == 1 ? cx : 0.0;
    double zi = formula == 1 ? cy : 0.0;
    const double parameter_real = formula == 1 ? julia_real : cx;
    const double parameter_imag = formula == 1 ? julia_imag : cy;
    int iteration = 0;
    for (; iteration < max_iter; ++iteration) {
        const double source_real = formula == 2 ? fabs(zr) : zr;
        const double source_imag = formula == 2 ? fabs(zi) : zi;
        const double next_real = source_real * source_real - source_imag * source_imag
            + parameter_real;
        const double next_imag = (formula == 3 ? -2.0 : 2.0)
            * source_real * source_imag + parameter_imag;
        zr = next_real;
        zi = next_imag;
        const double magnitude_squared = zr * zr + zi * zi;
        if (magnitude_squared > escape_squared || !isfinite(magnitude_squared)) {
            const double safe_squared = isfinite(magnitude_squared)
                ? fmax(magnitude_squared, escape_squared + 1.0e-7)
                : 1.7976931348623157e308;
            const double magnitude = sqrt(safe_squared);
            output[pixel] = (float)((double)(iteration + 1)
                - log(log(magnitude)) / log(2.0) - output_bias);
            return;
        }
    }
    output[pixel] = (float)((double)max_iter - output_bias);
}

#define DIRECT_SPECIALIZED_KERNEL(kernel_name, literal_formula) \
__kernel void kernel_name( \
    __global float* output, const int width, const int height, \
    const double center_real, const double center_imag, \
    const double width_span, const double height_span, const int max_iter, \
    const double output_bias, const int unused_formula, \
    const double julia_real, const double julia_imag, \
    const double escape_squared, const int coordinate_mode \
) { \
    const size_t pixel = get_global_id(0); \
    const size_t count = (size_t)width * (size_t)height; \
    if (pixel >= count) return; \
    direct_render_pixel_specialized( \
        output, pixel, width, height, center_real, center_imag, \
        width_span, height_span, max_iter, output_bias, literal_formula, \
        julia_real, julia_imag, escape_squared, coordinate_mode); \
}

DIRECT_SPECIALIZED_KERNEL(mandelbrot_direct_mandelbrot, 0)
DIRECT_SPECIALIZED_KERNEL(mandelbrot_direct_julia, 1)
DIRECT_SPECIALIZED_KERNEL(mandelbrot_direct_burning_ship, 2)
DIRECT_SPECIALIZED_KERNEL(mandelbrot_direct_tricorn, 3)
#undef DIRECT_SPECIALIZED_KERNEL
)CLC";

// Deep perturbation uses the same shared-exponent complex layout as the CPU
// renderer. In particular, the pixel delta is never converted to a double,
// which is the crucial difference between a useful e150 device pass and a
// shallow-preview shortcut. Mandelbrot can use its compact linear BLA table;
// Julia, Burning Ship, and Tricorn take formula-specific exact scaled steps.
// The kernel deliberately reports Mandelbrot cancellation as NaN so the
// existing reference-repair machinery can choose a new centre.
constexpr const char* OPENCL_DEEP_PERTURBATION_KERNEL = R"CLC(
#if defined(cl_khr_fp64)
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#elif defined(cl_amd_fp64)
#pragma OPENCL EXTENSION cl_amd_fp64 : enable
#endif

typedef struct { double r; double i; int e; int pad; } sc;
typedef struct { sc A; sc B; double radius_m; int radius_e; int length; } bla_step;

inline sc sc_normalize(sc a) {
    const double magnitude = fmax(fabs(a.r), fabs(a.i));
    if (magnitude == 0.0) return (sc){0.0, 0.0, 0, 0};
    // Products and sums of normalized values overwhelmingly need only a
    // one-bit shift.  Keep this in the device kernel just as the host
    // ScaledComplex path does: frexp/ldexp in every perturbation iteration
    // is much more expensive on consumer GPUs than exact powers of two.
    if (magnitude >= 2.0) {
        int shift = 0;
        (void)frexp(magnitude, &shift);
        a.r = ldexp(a.r, -shift);
        a.i = ldexp(a.i, -shift);
        a.e += shift;
        return a;
    }
    if (magnitude >= 1.0) {
        a.r *= 0.5;
        a.i *= 0.5;
        a.e += 1;
        return a;
    }
    if (magnitude >= 0.5) return a;
    if (magnitude >= 0.25) {
        a.r *= 2.0;
        a.i *= 2.0;
        a.e -= 1;
        return a;
    }
    int shift = 0;
    const double ignored = frexp(magnitude, &shift);
    (void)ignored;
    a.r = ldexp(a.r, -shift);
    a.i = ldexp(a.i, -shift);
    a.e += shift;
    return a;
}

inline sc sc_add_device(sc a, sc b) {
    if (a.r == 0.0 && a.i == 0.0) return b;
    if (b.r == 0.0 && b.i == 0.0) return a;
    if (b.e > a.e) { const sc temporary = a; a = b; b = temporary; }
    const int difference = a.e - b.e;
    if (difference > 60) return a;
    return sc_normalize((sc){a.r + ldexp(b.r, -difference),
                              a.i + ldexp(b.i, -difference), a.e, 0});
}

// The scalar deep renderer checks the escape margin in a small mantissa /
// exponent type instead of squaring the rounded ``reference + delta`` value.
// Keep the same compensated norm on the device: at a Kalles-coordinate
// boundary, those last few bits decide whether a pixel is an escape or a
// perturbation glitch.
typedef struct { double m; int e; } fe;

inline fe fe_normalize_device(double mantissa, int exponent) {
    if (mantissa == 0.0) return (fe){0.0, 0};
    int shift = 0;
    const double magnitude = frexp(fabs(mantissa), &shift);
    if (!(magnitude > 0.0) || !isfinite(magnitude)) {
        return (fe){mantissa, exponent};
    }
    return (fe){ldexp(mantissa, -shift), exponent + shift};
}

inline fe fe_add_device(fe a, fe b) {
    if (a.m == 0.0) return b;
    if (b.m == 0.0) return a;
    fe larger = a;
    fe smaller = b;
    if (b.e > a.e) {
        larger = b;
        smaller = a;
    }
    const int difference = larger.e - smaller.e;
    if (difference > 60) return larger;
    return fe_normalize_device(
        larger.m + ldexp(smaller.m, -difference), larger.e);
}

inline fe fe_mul_device(fe a, fe b) {
    if (a.m == 0.0 || b.m == 0.0) return (fe){0.0, 0};
    return fe_normalize_device(a.m * b.m, a.e + b.e);
}

inline fe fe_scale_device(fe a, const double scale) {
    return fe_normalize_device(a.m * scale, a.e);
}

// The perturbation glitch threshold is always the same positive scalar.  A
// general ``frexp(a.m * 1e-7)`` in every reference iteration is needlessly
// expensive on GPUs.  ``frexp(1e-7)`` is exactly (0.8388608, -23) in the
// device double format, and reference norms are already normalized to
// [0.5, 1), so the product needs at most one known two-power adjustment.
inline fe fe_glitch_threshold_device(fe a) {
    if (a.m == 0.0) return (fe){0.0, 0};
    const double mantissa = a.m * 0.8388608;
    if (mantissa < 0.5) return (fe){mantissa * 2.0, a.e - 24};
    return (fe){mantissa, a.e - 23};
}

inline fe fe_component_device(sc value, const int imaginary) {
    return fe_normalize_device(imaginary ? value.i : value.r, value.e);
}

inline fe fe_norm_device(sc value) {
    return fe_add_device(
        fe_mul_device(fe_component_device(value, 0),
                      fe_component_device(value, 0)),
        fe_mul_device(fe_component_device(value, 1),
                      fe_component_device(value, 1)));
}

// ``sc_normalize`` keeps the larger component in [0.5, 1), so the squared
// mantissa is bounded to [0.25, 2).  That means its normalized exponent can
// only be -1, 0, or +1.  Avoiding frexp here matters on consumer GPUs: the
// deep kernel performs this check once for the reconstructed value and once
// for the reference on every perturbation iteration.  The branch layout is
// intentionally identical to the host ``sc_norm_squared`` implementation.
inline fe sc_norm_fast_device(sc value, const double norm) {
    if (norm == 0.0) return (fe){0.0, 0};
    if (!isfinite(norm)) return (fe){INFINITY, 2147483647};
    const int squared_exponent = 2 * value.e;
    if (norm >= 1.0) return (fe){norm * 0.5, squared_exponent + 1};
    if (norm < 0.5) return (fe){norm * 2.0, squared_exponent - 1};
    return (fe){norm, squared_exponent};
}

inline fe sc_escape_margin_with_delta_device(
    sc reference,
    sc delta,
    fe bailout
) {
    const fe reference_real = fe_component_device(reference, 0);
    const fe reference_imag = fe_component_device(reference, 1);
    const fe delta_real = fe_component_device(delta, 0);
    const fe delta_imag = fe_component_device(delta, 1);
    const fe cross = fe_scale_device(
        fe_add_device(
            fe_mul_device(reference_real, delta_real),
            fe_mul_device(reference_imag, delta_imag)),
        2.0);
    // Keep the same operation order as the scalar renderer.  Computing
    // reference_norm + cross + delta_norm directly loses the tiny signed
    // margin when reference_norm is close to the bailout radius.  Subtract
    // the bailout first, then add the margin back around zero.
    const fe reference_margin = fe_add_device(
        fe_norm_device(reference),
        fe_scale_device(bailout, -1.0));
    return fe_add_device(
        fe_add_device(reference_margin, cross),
        fe_norm_device(delta));
}

inline fe sc_norm_squared_with_delta_device(
    sc reference,
    sc delta,
    fe bailout
) {
    return fe_add_device(
        bailout,
        sc_escape_margin_with_delta_device(reference, delta, bailout));
}

inline int fe_compare_device(fe a, fe b) {
    if (a.m == 0.0 && b.m == 0.0) return 0;
    if (a.m == 0.0) return b.m > 0.0 ? -1 : 1;
    if (b.m == 0.0) return a.m > 0.0 ? 1 : -1;
    if ((a.m < 0.0) != (b.m < 0.0)) return a.m < 0.0 ? -1 : 1;
    if (a.e != b.e) {
        const int sign = a.e < b.e ? -1 : 1;
        return a.m < 0.0 ? -sign : sign;
    }
    if (a.m == b.m) return 0;
    const int sign = a.m < b.m ? -1 : 1;
    return a.m < 0.0 ? -sign : sign;
}

inline int fe_is_finite_device(fe value) {
    return isfinite(value.m) && value.e != 2147483647;
}

inline sc sc_mul_device(sc a, sc b) {
    if ((a.r == 0.0 && a.i == 0.0) || (b.r == 0.0 && b.i == 0.0))
        return (sc){0.0, 0.0, 0, 0};
    return sc_normalize((sc){a.r * b.r - a.i * b.i,
                              a.r * b.i + a.i * b.r, a.e + b.e, 0});
}

inline sc sc_neg_device(sc value) {
    value.r = -value.r;
    value.i = -value.i;
    return value;
}

inline sc sc_sub_device(sc a, sc b) {
    return sc_add_device(a, sc_neg_device(b));
}

inline sc sc_conjugate_device(sc value) {
    value.i = -value.i;
    return value;
}

inline sc sc_double_device(sc value) {
    if (value.r == 0.0 && value.i == 0.0)
        return (sc){0.0, 0.0, 0, 0};
    value.e += 1;
    return value;
}

inline sc sc_abs_device(sc value) {
    value.r = fabs(value.r);
    value.i = fabs(value.i);
    return value;
}

inline int sc_outside_norm_device(sc value, const double norm,
                                  const int bailout_exponent,
                                  const double bailout_mantissa) {
    if (!isfinite(norm)) return 1;
    if (norm == 0.0) return 0;
    int shift = 0;
    const double normalized = frexp(norm, &shift);
    const int exponent = 2 * value.e + shift;
    return exponent > bailout_exponent
        || (exponent == bailout_exponent && normalized > bailout_mantissa);
}

inline int sc_norm_less_device(sc a, const double an, sc b) {
    const double bn = b.r * b.r + b.i * b.i;
    if (an == 0.0) return bn != 0.0;
    if (bn == 0.0) return 0;
    if (!isfinite(an)) return 0;
    if (!isfinite(bn)) return 1;
    // sc_normalize keeps both component pairs in [0.5, 1), so their squared
    // mantissas are already in [0.25, 2).  The shared exponent is therefore
    // sufficient to order different scales; only equal exponents need the
    // mantissa comparison.  The old implementation called frexp twice here
    // on every perturbation iteration.
    if (a.e != b.e) return a.e < b.e;
    return an < bn;
}

inline int sc_norm_below_radius_device(sc value, const double radius_m,
                                       const int radius_e) {
    const double norm = value.r * value.r + value.i * value.i;
    if (norm == 0.0) return 1;
    if (!isfinite(norm) || !(radius_m > 0.0)) return 0;
    int shift = 0;
    double mantissa = norm;
    if (mantissa >= 1.0) {
        mantissa *= 0.5;
        shift = 1;
    } else if (mantissa < 0.5) {
        mantissa *= 2.0;
        shift = -1;
    }
    const int exponent = 2 * value.e + shift;
    return exponent < radius_e || (exponent == radius_e && mantissa < radius_m);
}

inline double sc_component_device(sc value, const int imaginary) {
    const double component = imaginary ? value.i : value.r;
    const double result = ldexp(component, value.e);
    return isfinite(result) ? result : 0.0;
}

inline void sc_store_de_device(sc total, sc derivative,
                               const double spacing_mantissa,
                               const int spacing_exponent,
                               __global double *out_r,
                               __global double *out_i) {
    *out_r = 0.0; *out_i = 0.0;
    const double magnitude = hypot(total.r, total.i);
    const double log_magnitude = log(magnitude) + (double)total.e * 0.69314718055994530942;
    if (!(magnitude > 0.0) || !(log_magnitude > 0.0) || !isfinite(log_magnitude)) return;
    const sc unit = (sc){total.r / magnitude, total.i / magnitude, 0, 0};
    const sc spacing = sc_normalize((sc){spacing_mantissa, 0.0, spacing_exponent, 0});
    sc conjugate_derivative = derivative; conjugate_derivative.i = -conjugate_derivative.i;
    const sc denominator = sc_mul_device(unit, sc_mul_device(conjugate_derivative, spacing));
    const double denominator_norm = denominator.r * denominator.r + denominator.i * denominator.i;
    if (!(denominator_norm > 0.0) || !isfinite(denominator_norm)) return;
    const sc result = sc_normalize((sc){
        magnitude * log_magnitude * denominator.r / denominator_norm,
        -magnitude * log_magnitude * denominator.i / denominator_norm,
        total.e - denominator.e, 0});
    *out_r = sc_component_device(result, 0);
    *out_i = sc_component_device(result, 1);
}

__kernel void mandelbrot_deep_perturbation(
    __global float* output,
    __global const sc* reference,
    const int reference_count,
    const int width,
    const int height,
    const double view_width_mantissa,
    const int view_width_exponent,
    const double view_height_mantissa,
    const int view_height_exponent,
    const int max_iter,
    const double output_bias,
    const int bailout_exponent,
    const double bailout_mantissa,
    const int coordinate_mode,
    const double pixel_spacing_mantissa,
    const int pixel_spacing_exponent,
    const int write_planes,
    __global long* orbit_iteration,
    __global double* phase,
    __global double* de_x,
    __global double* de_y,
    __global double* test1,
    __global double* test2,
    __global const bla_step* bla_steps,
    __global const int* bla_offsets,
    __global const int* bla_counts,
    const int bla_level_count,
    const int use_linear_bla,
    const int formula,
    const double parameter_real,
    const double parameter_imag,
    const int parameter_exponent,
    __global const sc* point_offsets,
    const int point_mode,
    __global const fe* reference_norms
) {
    const size_t pixel = get_global_id(0);
    const size_t count = (size_t)width * (size_t)height;
    if (pixel >= count) return;
    const int py = (int)(pixel / (size_t)width);
    const int px = (int)(pixel - (size_t)py * (size_t)width);
    // Project mode is pixel-centred; Kalles uses integer half-dimensions.
    // The distinction matters only for even frames, which is why it is easy
    // to miss in square/odd deep-zoom probes.
    const double x_axis = coordinate_mode == 1
        ? (double)px - (double)width * 0.5 : (double)px - (double)(width - 1) * 0.5;
    const double y_axis = coordinate_mode == 1
        ? (double)height * 0.5 - (double)py : (double)(height - 1) * 0.5 - (double)py;
    // Forming the two values separately preserves subnormal viewport spans:
    // normalize after assigning their true binary exponents instead of
    // letting ldexp underflow an e150 delta first. Point repairs already have
    // the exact shared-exponent offset in ``point_offsets``; using it directly
    // avoids rebuilding the coordinate geometry a second time on the device.
    sc delta_c;
    if (point_mode) {
        delta_c = sc_normalize(point_offsets[pixel]);
    } else {
        sc dx = (sc){view_width_mantissa * x_axis / (double)width, 0.0,
                     view_width_exponent, 0};
        sc dy = (sc){0.0, view_height_mantissa * y_axis / (double)height,
                     view_height_exponent, 0};
        delta_c = sc_add_device(sc_normalize(dx), sc_normalize(dy));
    }
    if (reference_count < 2) { output[pixel] = nan((uint)0); return; }
    const int julia = formula == 1;
    const sc zero = (sc){0.0, 0.0, 0, 0};
    const sc parameter = (sc){parameter_real, parameter_imag,
                              parameter_exponent, 0};
    const sc parameter_delta = julia ? zero : delta_c;
    sc delta = delta_c;
    sc derivative = (sc){1.0, 0.0, 0, 0};
    int iteration = julia ? 0 : 1;
    int reference_index = julia ? 0 : 1;
    int rebase_count = 0;
    double previous_norm = 0.0;
    while (iteration < max_iter && reference_index < reference_count) {
        const sc reference_value = reference[reference_index];
        const sc total = sc_add_device(reference_value, delta);
        const double total_norm = total.r * total.r + total.i * total.i;
        // The device can only inspect the reconstructed scaled value here.
        // Do not rebuild the CPU bailout-centred margin: when the reference
        // is many binary exponents below the bailout, that subtraction
        // intentionally collapses to zero and would mark every interior
        // pixel as a glitch. The reconstructed norm is the right quantity
        // for the GPU's own recurrence and remains normalized in `fe`.
        const fe reconstructed_norm = sc_norm_fast_device(total, total_norm);
        // The reference orbit is immutable for the whole atlas.  Its norm
        // therefore belongs in a compact read-only table, not in the inner
        // pixel/iteration loop.  Recomputing this value used to perform two
        // extra multiplies and a branch for every perturbation step.
        const fe reference_norm = reference_norms[reference_index];
        // Kalles-style cancellation/glitch guard. Do not colour a broken
        // perturbation as an interior pixel.
        if (formula == 0
            && (reference_value.r != 0.0 || reference_value.i != 0.0)
            && fe_is_finite_device(reference_norm)
            && fe_compare_device(
                reconstructed_norm,
                fe_glitch_threshold_device(reference_norm)) < 0) {
            output[pixel] = nan((uint)0);
            return;
        }
        if (fe_compare_device(
                reconstructed_norm,
                (fe){bailout_mantissa, bailout_exponent}) > 0) {
            const double magnitude_log = 0.5 * (
                log(fmax(reconstructed_norm.m, 1.0e-300))
                + (double)reconstructed_norm.e * 0.69314718055994530942);
            output[pixel] = (float)((double)iteration
                - log(fmax(magnitude_log, 1.0e-300)) / 0.69314718055994530942
                - output_bias);
            if (write_planes) {
                orbit_iteration[pixel] = (long)(iteration - 1);
                double p = atan2(total.i, total.r) / 6.28318530717958647692;
                phase[pixel] = p - floor(p);
                sc_store_de_device(total, derivative, pixel_spacing_mantissa,
                                   pixel_spacing_exponent, &de_x[pixel], &de_y[pixel]);
                test1[pixel] = ldexp(total_norm, 2 * total.e);
                test2[pixel] = previous_norm;
            }
            return;
        }
        previous_norm = ldexp(total_norm, 2 * total.e);
        // Restart from z_0 when the perturbation becomes larger than the
        // reference state. This is the inexpensive Kalles rebase that keeps
        // a single reference useful through winding filaments; without it a
        // mathematically valid GPU perturbation drifts long before a true
        // cancellation/glitch is detected.
        if (formula == 0 && sc_norm_less_device(total, total_norm, delta)
            && ++rebase_count <= 64) {
            delta = total;
            reference_index = 0;
            continue;
        }
        // This is the same conservative linear BLA relation used by the
        // CPU's ultra-deep tier: d' = A*d + B*dc.  It is valid only inside
        // its uploaded radius; otherwise the kernel takes one exact scaled
        // perturbation step below.  KFP plane output deliberately disables
        // this until derivative composition is also ported.
        if (formula == 0 && use_linear_bla && !write_planes && reference_index > 0) {
            const int remaining = max_iter - iteration;
            int jumped = 0;
            const int offset = reference_index - 1;
            // Only levels up to the trailing-zero count of the current
            // reference offset can be aligned here. The old loop tested
            // every level from the top, rejecting most of them with the
            // same bit-mask before reaching the useful candidates.
            const int highest_aligned_level = offset == 0
                ? bla_level_count - 1
                : min(bla_level_count - 1, (int)ctz((uint)offset));
            for (int level = highest_aligned_level; level >= 0; --level) {
                const int span = 1 << level;
                if ((offset & (span - 1)) != 0) continue;
                const int item = offset >> level;
                if (item < 0 || item >= bla_counts[level]) continue;
                const bla_step candidate = bla_steps[bla_offsets[level] + item];
                if (candidate.length <= 1 || candidate.length > remaining
                    || !sc_norm_below_radius_device(
                        delta, candidate.radius_m, candidate.radius_e)) continue;
                delta = sc_add_device(sc_mul_device(candidate.A, delta),
                                      sc_mul_device(candidate.B, delta_c));
                reference_index += candidate.length;
                iteration += candidate.length;
                jumped = 1;
                break;
            }
            if (jumped) continue;
        }
        if (formula == 2) {
            // Burning Ship is smooth only while both absolute-value signs
            // stay fixed. Use its real 2x2 derivative in that region, and
            // reconstruct the next state from the actual total at a cusp so
            // the GPU does not smear an axis crossing into a rectangular
            // deep-zoom tile.
            const int crosses_real = (reference_value.r < 0.0)
                != (total.r < 0.0);
            const int crosses_imag = (reference_value.i < 0.0)
                != (total.i < 0.0);
            if ((crosses_real || crosses_imag)
                && reference_index + 1 < reference_count) {
                const sc absolute_total = sc_abs_device(total);
                const sc squared = sc_mul_device(absolute_total, absolute_total);
                delta = sc_add_device(
                    sc_sub_device(sc_add_device(squared, parameter),
                                  reference[reference_index + 1]),
                    parameter_delta);
            } else {
                const sc absolute_reference = sc_abs_device(reference_value);
                sc signed_delta = delta;
                signed_delta.r *= reference_value.r < 0.0 ? -1.0 : 1.0;
                signed_delta.i *= reference_value.i < 0.0 ? -1.0 : 1.0;
                const sc linear = sc_double_device(
                    sc_mul_device(absolute_reference, signed_delta));
                const sc square = sc_mul_device(signed_delta, signed_delta);
                delta = sc_add_device(sc_add_device(linear, square), parameter_delta);
            }
        } else if (formula == 3) {
            const sc conjugate_reference = sc_conjugate_device(reference_value);
            const sc conjugate_delta = sc_conjugate_device(delta);
            delta = sc_add_device(
                sc_double_device(sc_mul_device(conjugate_reference, conjugate_delta)),
                sc_add_device(sc_mul_device(conjugate_delta, conjugate_delta),
                              parameter_delta));
        } else {
            // Mandelbrot and Julia are holomorphic z² maps. Julia keeps c
            // fixed; Mandelbrot carries the pixel offset as dc.
            const sc linear = sc_double_device(sc_mul_device(reference_value, delta));
            const sc square = sc_mul_device(delta, delta);
            if (write_planes && formula == 0) {
                derivative = sc_add_device(
                    sc_double_device(sc_mul_device(total, derivative)),
                    (sc){1.0, 0.0, 0, 0});
            }
            delta = sc_add_device(sc_add_device(linear, square), parameter_delta);
        }
        ++reference_index;
        ++iteration;
    }
    output[pixel] = (float)((double)max_iter - output_bias);
    if (write_planes) {
        orbit_iteration[pixel] = (long)max_iter;
        phase[pixel] = 0.0; de_x[pixel] = 0.0; de_y[pixel] = 0.0;
        test1[pixel] = 0.0; test2[pixel] = 0.0;
    }
}
)CLC";

// The ordinary atlas spends most of its time in the exact Mandelbrot
// perturbation interval before the linear-BLA tiers become valid.  Keep a
// separate scalar entry point for that hot case: the generic kernel above
// must support four formulas, KFP planes, point repairs, and BLA, but none of
// those branches are needed for a normal field tile.  The recurrence and
// bailout order intentionally match the Mandelbrot branch above.
constexpr const char* OPENCL_DEEP_SCALAR_KERNEL = R"CLC(
__kernel void mandelbrot_deep_perturbation_scalar(
    __global float* output,
    __global const sc* reference,
    const int reference_count,
    const int width,
    const int height,
    const double view_width_mantissa,
    const int view_width_exponent,
    const double view_height_mantissa,
    const int view_height_exponent,
    const int max_iter,
    const double output_bias,
    const int bailout_exponent,
    const double bailout_mantissa,
    const int coordinate_mode,
    __global const fe* reference_norms
) {
    const size_t pixel = get_global_id(0);
    const size_t count = (size_t)width * (size_t)height;
    if (pixel >= count) return;
    const int py = (int)(pixel / (size_t)width);
    const int px = (int)(pixel - (size_t)py * (size_t)width);
    const double x_axis = coordinate_mode == 1
        ? (double)px - (double)width * 0.5
        : (double)px - (double)(width - 1) * 0.5;
    const double y_axis = coordinate_mode == 1
        ? (double)height * 0.5 - (double)py
        : (double)(height - 1) * 0.5 - (double)py;
    const sc dx = sc_normalize((sc){
        view_width_mantissa * x_axis / (double)width, 0.0,
        view_width_exponent, 0});
    const sc dy = sc_normalize((sc){
        0.0, view_height_mantissa * y_axis / (double)height,
        view_height_exponent, 0});
    const sc delta_c = sc_add_device(dx, dy);
    sc delta = delta_c;
    if (reference_count < 2) {
        output[pixel] = nan((uint)0);
        return;
    }
    int iteration = 1;
    int reference_index = 1;
    int rebase_count = 0;
    while (iteration < max_iter && reference_index < reference_count) {
        const sc reference_value = reference[reference_index];
        const sc total = sc_add_device(reference_value, delta);
        const double total_norm = total.r * total.r + total.i * total.i;
        const fe reconstructed_norm = sc_norm_fast_device(total, total_norm);
        const fe reference_norm = reference_norms[reference_index];
        if ((reference_value.r != 0.0 || reference_value.i != 0.0)
            && fe_is_finite_device(reference_norm)
            && fe_compare_device(
                reconstructed_norm,
                fe_glitch_threshold_device(reference_norm)) < 0) {
            output[pixel] = nan((uint)0);
            return;
        }
        if (fe_compare_device(
                reconstructed_norm,
                (fe){bailout_mantissa, bailout_exponent}) > 0) {
            const double magnitude_log = 0.5 * (
                log(fmax(reconstructed_norm.m, 1.0e-300))
                + (double)reconstructed_norm.e * 0.69314718055994530942);
            output[pixel] = (float)((double)iteration
                - log(fmax(magnitude_log, 1.0e-300))
                    / 0.69314718055994530942
                - output_bias);
            return;
        }
        // Both scaled values keep their largest component in [0.5, 1), so
        // a two-or-more exponent lead proves that |total| cannot be smaller
        // than |delta|. Avoid the two frexp calls in the common non-glitch
        // case; retain the exact comparison at the only exponents where
        // cancellation is possible.
        if (total.e <= delta.e + 1
            && sc_norm_less_device(total, total_norm, delta)
            && ++rebase_count <= 64) {
            delta = total;
            reference_index = 0;
            continue;
        }
        const sc linear = sc_double_device(sc_mul_device(reference_value, delta));
        const sc square = sc_mul_device(delta, delta);
        delta = sc_add_device(sc_add_device(linear, square), delta_c);
        ++reference_index;
        ++iteration;
    }
    output[pixel] = (float)((double)max_iter - output_bias);
}
)CLC";

// The ordinary deep path uses the generic kernel once a linear-BLA table is
// available, which keeps the implementation compact but leaves predictable
// formula/plane branches in every pixel.  This variant is the same
// Mandelbrot recurrence with the validated linear-BLA jump, without those
// optional branches.  It is an optional entry point: the generic kernel stays
// available as the correctness fallback for restrictive OpenCL compilers.
constexpr const char* OPENCL_DEEP_SCALAR_BLA_KERNEL = R"CLC(
__kernel void mandelbrot_deep_perturbation_scalar_bla(
    __global float* output,
    __global const sc* reference,
    const int reference_count,
    const int width,
    const int height,
    const double view_width_mantissa,
    const int view_width_exponent,
    const double view_height_mantissa,
    const int view_height_exponent,
    const int max_iter,
    const double output_bias,
    const int bailout_exponent,
    const double bailout_mantissa,
    const int coordinate_mode,
    __global const fe* reference_norms,
    __global const bla_step* bla_steps,
    __global const int* bla_offsets,
    __global const int* bla_counts,
    const int bla_level_count
) {
    const size_t pixel = get_global_id(0);
    const size_t count = (size_t)width * (size_t)height;
    if (pixel >= count) return;
    const int py = (int)(pixel / (size_t)width);
    const int px = (int)(pixel - (size_t)py * (size_t)width);
    const double x_axis = coordinate_mode == 1
        ? (double)px - (double)width * 0.5
        : (double)px - (double)(width - 1) * 0.5;
    const double y_axis = coordinate_mode == 1
        ? (double)height * 0.5 - (double)py
        : (double)(height - 1) * 0.5 - (double)py;
    const sc dx = sc_normalize((sc){
        view_width_mantissa * x_axis / (double)width, 0.0,
        view_width_exponent, 0});
    const sc dy = sc_normalize((sc){
        0.0, view_height_mantissa * y_axis / (double)height,
        view_height_exponent, 0});
    const sc delta_c = sc_add_device(dx, dy);
    sc delta = delta_c;
    if (reference_count < 2) {
        output[pixel] = nan((uint)0);
        return;
    }
    int iteration = 1;
    int reference_index = 1;
    int rebase_count = 0;
    while (iteration < max_iter && reference_index < reference_count) {
        const sc reference_value = reference[reference_index];
        const sc total = sc_add_device(reference_value, delta);
        const double total_norm = total.r * total.r + total.i * total.i;
        const fe reconstructed_norm = sc_norm_fast_device(total, total_norm);
        const fe reference_norm = reference_norms[reference_index];
        if ((reference_value.r != 0.0 || reference_value.i != 0.0)
            && fe_is_finite_device(reference_norm)
            && fe_compare_device(
                reconstructed_norm,
                fe_glitch_threshold_device(reference_norm)) < 0) {
            output[pixel] = nan((uint)0);
            return;
        }
        if (fe_compare_device(
                reconstructed_norm,
                (fe){bailout_mantissa, bailout_exponent}) > 0) {
            const double magnitude_log = 0.5 * (
                log(fmax(reconstructed_norm.m, 1.0e-300))
                + (double)reconstructed_norm.e * 0.69314718055994530942);
            output[pixel] = (float)((double)iteration
                - log(fmax(magnitude_log, 1.0e-300))
                    / 0.69314718055994530942
                - output_bias);
            return;
        }
        if (total.e <= delta.e + 1
            && sc_norm_less_device(total, total_norm, delta)
            && ++rebase_count <= 64) {
            delta = total;
            reference_index = 0;
            continue;
        }

        const int remaining = max_iter - iteration;
        int jumped = 0;
        const int offset = reference_index - 1;
        const int highest_aligned_level = offset == 0
            ? bla_level_count - 1
            : min(bla_level_count - 1, (int)ctz((uint)offset));
        for (int level = highest_aligned_level; level >= 0; --level) {
            const int span = 1 << level;
            if ((offset & (span - 1)) != 0) continue;
            const int item = offset >> level;
            if (item < 0 || item >= bla_counts[level]) continue;
            const bla_step candidate = bla_steps[bla_offsets[level] + item];
            if (candidate.length <= 1 || candidate.length > remaining
                || !sc_norm_below_radius_device(
                    delta, candidate.radius_m, candidate.radius_e)) continue;
            delta = sc_add_device(sc_mul_device(candidate.A, delta),
                                  sc_mul_device(candidate.B, delta_c));
            reference_index += candidate.length;
            iteration += candidate.length;
            jumped = 1;
            break;
        }
        if (jumped) continue;

        const sc linear = sc_double_device(sc_mul_device(reference_value, delta));
        const sc square = sc_mul_device(delta, delta);
        delta = sc_add_device(sc_add_device(linear, square), delta_c);
        ++reference_index;
        ++iteration;
    }
    output[pixel] = (float)((double)max_iter - output_bias);
}
)CLC";

// A large 4K deep field is dominated by the scaled recurrence, and consumer
// NVIDIA GPUs execute fp64 at a small fraction of their fp32 rate.  This
// kernel keeps the exponent-separated representation (so e150 does not
// underflow) while storing the normalized mantissas as floats.  It is used
// only for large scalar fields; strict double precision remains the default
// for small probes, point repairs, and every orbit-dependent plane path.
constexpr const char* OPENCL_DEEP_MIXED_KERNEL = R"CLC(
typedef struct { float r; float i; int e; int pad; } fsc;
typedef struct { float m; int e; } ffe;
typedef struct { fsc A; fsc B; float radius_m; int radius_e; int length; } fbla_step;

inline fsc fsc_normalize_device(fsc a) {
    const float magnitude = fmax(fabs(a.r), fabs(a.i));
    if (magnitude == 0.0f) return (fsc){0.0f, 0.0f, 0, 0};
    if (magnitude >= 1.0f) {
        if (magnitude < 2.0f) {
            a.r *= 0.5f; a.i *= 0.5f; a.e += 1;
            return a;
        }
        int shift = 0;
        (void)frexp(magnitude, &shift);
        a.r = ldexp(a.r, -shift);
        a.i = ldexp(a.i, -shift);
        a.e += shift;
        return a;
    }
    if (magnitude >= 0.5f) return a;
    if (magnitude >= 0.25f) {
        a.r *= 2.0f; a.i *= 2.0f; a.e -= 1;
        return a;
    }
    int shift = 0;
    (void)frexp(magnitude, &shift);
    a.r = ldexp(a.r, -shift);
    a.i = ldexp(a.i, -shift);
    a.e += shift;
    return a;
}

inline fsc fsc_add_device(fsc a, fsc b) {
    if (a.r == 0.0f && a.i == 0.0f) return b;
    if (b.r == 0.0f && b.i == 0.0f) return a;
    if (b.e > a.e) { const fsc temporary = a; a = b; b = temporary; }
    const int difference = a.e - b.e;
    if (difference > 24) return a;
    return fsc_normalize_device((fsc){
        a.r + ldexp(b.r, -difference),
        a.i + ldexp(b.i, -difference), a.e, 0});
}

inline fsc fsc_mul_device(fsc a, fsc b) {
    if ((a.r == 0.0f && a.i == 0.0f)
        || (b.r == 0.0f && b.i == 0.0f)) {
        return (fsc){0.0f, 0.0f, 0, 0};
    }
    return fsc_normalize_device((fsc){
        a.r * b.r - a.i * b.i,
        a.r * b.i + a.i * b.r, a.e + b.e, 0});
}

inline fsc fsc_neg_device(fsc value) {
    value.r = -value.r; value.i = -value.i; return value;
}

inline fsc fsc_sub_device(fsc a, fsc b) {
    return fsc_add_device(a, fsc_neg_device(b));
}

inline fsc fsc_conjugate_device(fsc value) {
    value.i = -value.i; return value;
}

inline fsc fsc_double_device(fsc value) {
    if (value.r == 0.0f && value.i == 0.0f)
        return (fsc){0.0f, 0.0f, 0, 0};
    value.e += 1; return value;
}

inline fsc fsc_abs_device(fsc value) {
    value.r = fabs(value.r); value.i = fabs(value.i); return value;
}

inline ffe ffe_normalize_device(float mantissa, int exponent) {
    if (mantissa == 0.0f) return (ffe){0.0f, 0};
    int shift = 0;
    (void)frexp(fabs(mantissa), &shift);
    return (ffe){ldexp(mantissa, -shift), exponent + shift};
}

inline ffe ffe_add_device(ffe a, ffe b) {
    if (a.m == 0.0f) return b;
    if (b.m == 0.0f) return a;
    ffe larger = a;
    ffe smaller = b;
    if (b.e > a.e) { larger = b; smaller = a; }
    const int difference = larger.e - smaller.e;
    if (difference > 24) return larger;
    return ffe_normalize_device(
        larger.m + ldexp(smaller.m, -difference), larger.e);
}

inline ffe ffe_mul_device(ffe a, ffe b) {
    if (a.m == 0.0f || b.m == 0.0f) return (ffe){0.0f, 0};
    return ffe_normalize_device(a.m * b.m, a.e + b.e);
}

inline ffe ffe_scale_device(ffe a, float scale) {
    return ffe_normalize_device(a.m * scale, a.e);
}

inline ffe ffe_component_device(fsc value, int imaginary) {
    return ffe_normalize_device(imaginary ? value.i : value.r, value.e);
}

inline ffe ffe_norm_device(fsc value) {
    return ffe_add_device(
        ffe_mul_device(ffe_component_device(value, 0),
                       ffe_component_device(value, 0)),
        ffe_mul_device(ffe_component_device(value, 1),
                       ffe_component_device(value, 1)));
}

inline ffe fsc_norm_fast_device(fsc value, float norm) {
    if (norm == 0.0f) return (ffe){0.0f, 0};
    if (!isfinite(norm)) return (ffe){INFINITY, 2147483647};
    const int squared_exponent = 2 * value.e;
    if (norm >= 1.0f) return (ffe){norm * 0.5f, squared_exponent + 1};
    if (norm < 0.5f) return (ffe){norm * 2.0f, squared_exponent - 1};
    return (ffe){norm, squared_exponent};
}

inline ffe fsc_glitch_threshold_device(ffe value) {
    if (value.m == 0.0f) return (ffe){0.0f, 0};
    const float mantissa = value.m * 0.8388608f;
    if (mantissa < 0.5f) return (ffe){mantissa * 2.0f, value.e - 24};
    return (ffe){mantissa, value.e - 23};
}

inline int ffe_compare_device(ffe a, ffe b) {
    if (a.m == 0.0f && b.m == 0.0f) return 0;
    if (a.m == 0.0f) return b.m > 0.0f ? -1 : 1;
    if (b.m == 0.0f) return a.m > 0.0f ? 1 : -1;
    if ((a.m < 0.0f) != (b.m < 0.0f)) return a.m < 0.0f ? -1 : 1;
    if (a.e != b.e) {
        const int sign = a.e < b.e ? -1 : 1;
        return a.m < 0.0f ? -sign : sign;
    }
    if (a.m == b.m) return 0;
    const int sign = a.m < b.m ? -1 : 1;
    return a.m < 0.0f ? -sign : sign;
}

inline int ffe_is_finite_device(ffe value) {
    return isfinite(value.m) && value.e != 2147483647;
}

inline int fsc_norm_less_device(fsc a, float an, fsc b) {
    const float bn = b.r * b.r + b.i * b.i;
    if (an == 0.0f) return bn != 0.0f;
    if (bn == 0.0f) return 0;
    if (!isfinite(an)) return 0;
    if (!isfinite(bn)) return 1;
    if (a.e != b.e) return a.e < b.e;
    return an < bn;
}

inline int fsc_norm_below_radius_device(
    fsc value, float radius_m, int radius_e
) {
    const float norm = value.r * value.r + value.i * value.i;
    if (norm == 0.0f) return 1;
    if (!isfinite(norm) || !(radius_m > 0.0f)) return 0;
    int shift = 0;
    float mantissa = norm;
    if (mantissa >= 1.0f) { mantissa *= 0.5f; shift = 1; }
    else if (mantissa < 0.5f) { mantissa *= 2.0f; shift = -1; }
    const int exponent = 2 * value.e + shift;
    return exponent < radius_e
        || (exponent == radius_e && mantissa < radius_m);
}

inline fsc fsc_parameter(float real, float imag, int exponent) {
    return fsc_normalize_device((fsc){real, imag, exponent, 0});
}

__kernel void mandelbrot_deep_perturbation_mixed(
    __global float* output,
    __global const fsc* reference,
    const int reference_count,
    const int width,
    const int height,
    const float view_width_scale,
    const int view_width_exponent,
    const float view_height_scale,
    const int view_height_exponent,
    const int max_iter,
    const float output_bias,
    const int bailout_exponent,
    const float bailout_mantissa,
    const int coordinate_mode,
    __global const ffe* reference_norms,
    __global const fbla_step* bla_steps,
    __global const int* bla_offsets,
    __global const int* bla_counts,
    const int bla_level_count,
    const int formula,
    const float parameter_real,
    const float parameter_imag,
    const int parameter_exponent
) {
    const size_t pixel = get_global_id(0);
    const size_t count = (size_t)width * (size_t)height;
    if (pixel >= count) return;
    const int py = (int)(pixel / (size_t)width);
    const int px = (int)(pixel - (size_t)py * (size_t)width);
    const float x_axis = coordinate_mode == 1
        ? (float)px - (float)width * 0.5f
        : (float)px - (float)(width - 1) * 0.5f;
    const float y_axis = coordinate_mode == 1
        ? (float)height * 0.5f - (float)py
        : (float)(height - 1) * 0.5f - (float)py;
    const fsc dx = fsc_normalize_device((fsc){
        view_width_scale * x_axis,
        0.0f, view_width_exponent, 0});
    const fsc dy = fsc_normalize_device((fsc){
        0.0f,
        view_height_scale * y_axis,
        view_height_exponent, 0});
    const fsc delta_c = fsc_add_device(dx, dy);
    if (reference_count < 2) {
        output[pixel] = nan((uint)0);
        return;
    }
    const int julia = formula == 1;
    const fsc zero = (fsc){0.0f, 0.0f, 0, 0};
    const fsc parameter = fsc_parameter(
        parameter_real, parameter_imag, parameter_exponent);
    const fsc parameter_delta = julia ? zero : delta_c;
    fsc delta = delta_c;
    int iteration = julia ? 0 : 1;
    int reference_index = julia ? 0 : 1;
    int rebase_count = 0;
    while (iteration < max_iter && reference_index < reference_count) {
        const fsc reference_value = reference[reference_index];
        const fsc total = fsc_add_device(reference_value, delta);
        const float total_norm = total.r * total.r + total.i * total.i;
        const ffe reconstructed_norm = fsc_norm_fast_device(total, total_norm);
        // The host stores the already-scaled glitch threshold in this table.
        // It is immutable for the reference orbit, so doing the multiply and
        // exponent adjustment once during upload avoids another normalization
        // branch in every perturbation iteration.
        const ffe reference_norm = reference_norms[reference_index];
        if (formula == 0
            && (reference_value.r != 0.0f || reference_value.i != 0.0f)
            && ffe_is_finite_device(reference_norm)
            && ffe_compare_device(reconstructed_norm, reference_norm) < 0) {
            output[pixel] = nan((uint)0);
            return;
        }
        if (ffe_compare_device(
                reconstructed_norm,
                (ffe){(float)bailout_mantissa, bailout_exponent}) > 0) {
            const float magnitude_log = 0.5f * (
                log(fmax(reconstructed_norm.m, 1.0e-30f))
                + (float)reconstructed_norm.e * 0.6931471805599453f);
            output[pixel] = (float)iteration
                - log(fmax(magnitude_log, 1.0e-30f))
                    / 0.6931471805599453f
                - (float)output_bias;
            return;
        }
        if (formula == 0
            && total.e <= delta.e + 1
            && fsc_norm_less_device(total, total_norm, delta)
            && ++rebase_count <= 64) {
            delta = total;
            reference_index = 0;
            continue;
        }
        if (formula == 0 && bla_level_count > 0 && reference_index > 0) {
            const int remaining = max_iter - iteration;
            int jumped = 0;
            const int offset = reference_index - 1;
            const int highest_aligned_level = offset == 0
                ? bla_level_count - 1
                : min(bla_level_count - 1, (int)ctz((uint)offset));
            for (int level = highest_aligned_level; level >= 0; --level) {
                const int span = 1 << level;
                if ((offset & (span - 1)) != 0) continue;
                const int item = offset >> level;
                if (item < 0 || item >= bla_counts[level]) continue;
                const fbla_step candidate = bla_steps[bla_offsets[level] + item];
                if (candidate.length <= 1 || candidate.length > remaining
                    || !fsc_norm_below_radius_device(
                        delta, candidate.radius_m, candidate.radius_e)) continue;
                delta = fsc_add_device(
                    fsc_mul_device(candidate.A, delta),
                    fsc_mul_device(candidate.B, delta_c));
                reference_index += candidate.length;
                iteration += candidate.length;
                jumped = 1;
                break;
            }
            if (jumped) continue;
        }
        if (formula == 2) {
            const int crosses_real = (reference_value.r < 0.0f)
                != (total.r < 0.0f);
            const int crosses_imag = (reference_value.i < 0.0f)
                != (total.i < 0.0f);
            if ((crosses_real || crosses_imag)
                && reference_index + 1 < reference_count) {
                const fsc absolute_total = fsc_abs_device(total);
                const fsc squared = fsc_mul_device(absolute_total, absolute_total);
                delta = fsc_add_device(
                    fsc_sub_device(fsc_add_device(squared, parameter),
                                   reference[reference_index + 1]),
                    parameter_delta);
            } else {
                const fsc absolute_reference = fsc_abs_device(reference_value);
                fsc signed_delta = delta;
                signed_delta.r *= reference_value.r < 0.0f ? -1.0f : 1.0f;
                signed_delta.i *= reference_value.i < 0.0f ? -1.0f : 1.0f;
                const fsc linear = fsc_double_device(
                    fsc_mul_device(absolute_reference, signed_delta));
                const fsc square = fsc_mul_device(signed_delta, signed_delta);
                delta = fsc_add_device(
                    fsc_add_device(linear, square), parameter_delta);
            }
        } else if (formula == 3) {
            const fsc conjugate_reference = fsc_conjugate_device(reference_value);
            const fsc conjugate_delta = fsc_conjugate_device(delta);
            delta = fsc_add_device(
                fsc_double_device(
                    fsc_mul_device(conjugate_reference, conjugate_delta)),
                fsc_add_device(
                    fsc_mul_device(conjugate_delta, conjugate_delta),
                    parameter_delta));
        } else {
            const fsc linear = fsc_double_device(
                fsc_mul_device(reference_value, delta));
            const fsc square = fsc_mul_device(delta, delta);
            delta = fsc_add_device(
                fsc_add_device(linear, square), parameter_delta);
        }
        ++reference_index;
        ++iteration;
    }
    output[pixel] = (float)max_iter - (float)output_bias;
}

// The normal export is Mandelbrot-only and has no orbit planes or point
// repairs.  Keeping that formula in a separate entry point lets the OpenCL
// compiler remove the Julia/Burning-Ship/Tricorn branches from the hot loop.
// The argument layout intentionally matches the prefix of the mixed kernel so
// the host-side setup stays straightforward.
__kernel void mandelbrot_deep_perturbation_mixed_mandelbrot(
    __global float* output,
    __global const fsc* reference,
    const int reference_count,
    const int width,
    const int height,
    const float view_width_scale,
    const int view_width_exponent,
    const float view_height_scale,
    const int view_height_exponent,
    const int max_iter,
    const float output_bias,
    const int bailout_exponent,
    const float bailout_mantissa,
    const int coordinate_mode,
    __global const ffe* reference_norms,
    __global const fbla_step* bla_steps,
    __global const int* bla_offsets,
    __global const int* bla_counts,
    const int bla_level_count
) {
    const size_t pixel = get_global_id(0);
    const size_t count = (size_t)width * (size_t)height;
    if (pixel >= count) return;
    const int py = (int)(pixel / (size_t)width);
    const int px = (int)(pixel - (size_t)py * (size_t)width);
    const float x_axis = coordinate_mode == 1
        ? (float)px - (float)width * 0.5f
        : (float)px - (float)(width - 1) * 0.5f;
    const float y_axis = coordinate_mode == 1
        ? (float)height * 0.5f - (float)py
        : (float)(height - 1) * 0.5f - (float)py;
    const fsc dx = fsc_normalize_device((fsc){
        view_width_scale * x_axis,
        0.0f, view_width_exponent, 0});
    const fsc dy = fsc_normalize_device((fsc){
        0.0f,
        view_height_scale * y_axis,
        view_height_exponent, 0});
    const fsc delta_c = fsc_add_device(dx, dy);
    if (reference_count < 2) {
        output[pixel] = nan((uint)0);
        return;
    }
    fsc delta = delta_c;
    int iteration = 1;
    int reference_index = 1;
    int rebase_count = 0;
    while (iteration < max_iter && reference_index < reference_count) {
        const fsc reference_value = reference[reference_index];
        const fsc total = fsc_add_device(reference_value, delta);
        const float total_norm = total.r * total.r + total.i * total.i;
        const ffe reconstructed_norm = fsc_norm_fast_device(total, total_norm);
        const ffe glitch_threshold = reference_norms[reference_index];
        if ((reference_value.r != 0.0f || reference_value.i != 0.0f)
            && ffe_is_finite_device(glitch_threshold)
            && ffe_compare_device(reconstructed_norm, glitch_threshold) < 0) {
            output[pixel] = nan((uint)0);
            return;
        }
        if (ffe_compare_device(
                reconstructed_norm,
                (ffe){bailout_mantissa, bailout_exponent}) > 0) {
            const float magnitude_log = 0.5f * (
                log(fmax(reconstructed_norm.m, 1.0e-30f))
                + (float)reconstructed_norm.e * 0.6931471805599453f);
            output[pixel] = (float)iteration
                - log(fmax(magnitude_log, 1.0e-30f))
                    / 0.6931471805599453f
                - output_bias;
            return;
        }
        if (total.e <= delta.e + 1
            && fsc_norm_less_device(total, total_norm, delta)
            && ++rebase_count <= 64) {
            delta = total;
            reference_index = 0;
            continue;
        }
        if (bla_level_count > 0 && reference_index > 0) {
            const int remaining = max_iter - iteration;
            int jumped = 0;
            const int offset = reference_index - 1;
            const int highest_aligned_level = offset == 0
                ? bla_level_count - 1
                : min(bla_level_count - 1, (int)ctz((uint)offset));
            for (int level = highest_aligned_level; level >= 0; --level) {
                const int span = 1 << level;
                if ((offset & (span - 1)) != 0) continue;
                const int item = offset >> level;
                if (item < 0 || item >= bla_counts[level]) continue;
                const fbla_step candidate = bla_steps[bla_offsets[level] + item];
                if (candidate.length <= 1 || candidate.length > remaining
                    || !fsc_norm_below_radius_device(
                        delta, candidate.radius_m, candidate.radius_e)) continue;
                delta = fsc_add_device(
                    fsc_mul_device(candidate.A, delta),
                    fsc_mul_device(candidate.B, delta_c));
                reference_index += candidate.length;
                iteration += candidate.length;
                jumped = 1;
                break;
            }
            if (jumped) continue;
        }
        const fsc linear = fsc_double_device(
            fsc_mul_device(reference_value, delta));
        const fsc square = fsc_mul_device(delta, delta);
        delta = fsc_add_device(
            fsc_add_device(linear, square), delta_c);
        ++reference_index;
        ++iteration;
    }
    output[pixel] = (float)max_iter - output_bias;
}
)CLC";

// Ordinary Aurora colourisation is a table lookup once the CPU has built the
// tiny audio-reactive palette. Keeping that lookup on the device avoids a
// second full OpenMP pass when the field was rendered by OpenCL.
constexpr const char* OPENCL_AURORA_COLOUR_KERNEL = R"CLC(
#if defined(cl_khr_fp64)
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#elif defined(cl_amd_fp64)
#pragma OPENCL EXTENSION cl_amd_fp64 : enable
#endif

__kernel void aurora_colourise(
    __global const float* field,
    __global const uchar* palette,
    __global uchar* output,
    const int pixel_count,
    const int max_iter,
    const int palette_size,
    const double palette_index_scale
) {
    const size_t pixel = get_global_id(0);
    if (pixel >= (size_t)pixel_count) return;
    const float smooth = field[pixel];
    const size_t destination = pixel * (size_t)3;
    if (!isfinite(smooth) || smooth >= (float)max_iter) {
        output[destination] = 0;
        output[destination + 1] = 0;
        output[destination + 2] = 0;
        return;
    }
    const double scaled = (double)smooth * palette_index_scale;
    int index = scaled >= (double)(palette_size - 1)
        ? palette_size - 1
        : (!isfinite(scaled) || scaled <= 0.0 ? 0 : (int)scaled);
    const size_t source = (size_t)index * (size_t)3;
    output[destination] = palette[source];
    output[destination + 1] = palette[source + 1];
    output[destination + 2] = palette[source + 2];
}
)CLC";

// Scalar KFP profiles use the same finite-difference stencil as the native
// SetColor path. Keep this as a separate kernel instead of trying to express
// KFP through the ordinary Aurora lookup: the distance transfer, cyclic LUT
// interpolation, slope relief, and ordered dither are all part of the KFP
// image contract. Plane-aware, textured, and multi-colour profiles continue
// to use the exact CPU implementation because they need data which is not in
// the scalar field ABI.
constexpr const char* OPENCL_KFP_COLOUR_KERNEL = R"CLC(
#if defined(cl_khr_fp64)
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#elif defined(cl_amd_fp64)
#pragma OPENCL EXTENSION cl_amd_fp64 : enable
#endif

inline double kfp_gpu_clamp(double value, double lower, double upper) {
    return fmin(fmax(value, lower), upper);
}

inline double kfp_gpu_sample(
    __global const float* field,
    int width,
    int height,
    int x,
    int y,
    int max_iter,
    double field_bias,
    double smooth_offset
) {
    x = max(0, min(width - 1, x));
    y = max(0, min(height - 1, y));
    const float raw = field[(size_t)y * (size_t)width + (size_t)x];
    if (!isfinite(raw)) return fmax(0.0, smooth_offset);
    if ((double)raw >= (double)max_iter - field_bias) {
        return (double)max_iter + 1.0;
    }
    return fmax(
        0.0,
        kfp_gpu_clamp((double)raw + field_bias, 0.0, (double)max_iter)
            + smooth_offset);
}

inline double kfp_gpu_reflected_sample(
    __global const float* field,
    int width,
    int height,
    int x,
    int y,
    int offset_x,
    int offset_y,
    int max_iter,
    double centre,
    double field_bias,
    double smooth_offset
) {
    const int sample_x = x + offset_x;
    const int sample_y = y + offset_y;
    if (sample_x >= 0 && sample_x < width
        && sample_y >= 0 && sample_y < height) {
        return kfp_gpu_sample(
            field, width, height, sample_x, sample_y, max_iter,
            field_bias, smooth_offset);
    }
    const int opposite_x = x - offset_x;
    const int opposite_y = y - offset_y;
    if (opposite_x >= 0 && opposite_x < width
        && opposite_y >= 0 && opposite_y < height) {
        return 2.0 * centre - kfp_gpu_sample(
            field, width, height, opposite_x, opposite_y, max_iter,
            field_bias, smooth_offset);
    }
    return centre;
}

inline double kfp_gpu_difference(
    int differences,
    double centre,
    double left,
    double right,
    double up,
    double down,
    double top_left,
    double top_right,
    double bottom_left,
    double bottom_right
) {
    const double inverse_sqrt_two = 0.7071067811865475244;
    if (differences == 0) {
        return fabs(left - centre) * 1.414
            + fabs(up - centre) * 1.414
            + fabs(top_left - centre)
            + fabs(bottom_left - centre);
    }
    if (differences == 1) {
        const double squared =
            (left - centre) * (left - centre)
            + (right - centre) * (right - centre)
            + (up - centre) * (up - centre)
            + (down - centre) * (down - centre)
            + ((top_left - centre) * (top_left - centre)
                + (bottom_right - centre) * (bottom_right - centre))
                * inverse_sqrt_two * inverse_sqrt_two
            + ((bottom_left - centre) * (bottom_left - centre)
                + (top_right - centre) * (top_right - centre))
                * inverse_sqrt_two * inverse_sqrt_two;
        return sqrt(fmax(0.0, squared * 0.25)) * 2.8284271247461903;
    }
    if (differences == 2) {
        const double squared =
            (right - left) * (right - left) * 0.25
            + (down - up) * (down - up) * 0.25
            + (bottom_right - top_left) * (bottom_right - top_left) * 0.125
            + (top_right - bottom_left) * (top_right - bottom_left) * 0.125;
        return sqrt(fmax(0.0, squared * 0.5)) * 2.8284271247461903;
    }
    if (differences == 3) {
        const double squared =
            (top_left - centre) * (top_left - centre) * 0.5
            + (left - up) * (left - up) * 0.5;
        return sqrt(fmax(0.0, squared)) * 2.8284271247461903;
    }
    if (differences == 4) {
        const double dx = ((up - top_left) + (centre - left)) * 0.5;
        const double dy = ((left - top_left) + (centre - up)) * 0.5;
        return sqrt(dx * dx + dy * dy) * 2.8284271247461903;
    }
    if (differences == 5) {
        const double dx = (right + top_right + bottom_right
            - left - top_left - bottom_left) / 6.0;
        const double dy = (down + bottom_left + bottom_right
            - up - top_left - top_right) / 6.0;
        return sqrt(dx * dx + dy * dy) * 2.8284271247461903;
    }
    if (differences == 6) {
        const double laplacian = top_left + 4.0 * up + top_right
            + 4.0 * left - 20.0 * centre + 4.0 * right
            + bottom_left + 4.0 * down + bottom_right;
        return sqrt(fabs(laplacian / 6.0 * 1.4426950408889634))
            * 2.8284271247461903;
    }
    // Differences_Analytic needs the derivative planes, which are deliberately
    // outside this scalar kernel. The CPU-compatible scalar path defines it as
    // zero when those planes are absent.
    return 0.0;
}

inline uchar kfp_gpu_byte(float value, int x, int y, int channel) {
    if (!isfinite(value)) value = 0.0f;
    const uint coordinate = ((uint)x + (uint)channel * 67u + (uint)y * 236u) * 119u;
    const float mask = (float)(coordinate & 255u) / 256.0f;
    const int quantized = (int)floor(255.0f * value + mask);
    return (uchar)max(0, min(255, quantized));
}

__kernel void kfp_colourise(
    __global const float* field,
    __global const uchar* lut,
    __global uchar* output,
    const int width,
    const int height,
    const int max_iter,
    const double field_bias,
    const double smooth_offset,
    const int iter_division,
    const double iter_div,
    const double color_offset,
    const int color_method,
    const int smooth,
    const int inverse_transition,
    const int flat,
    const int slopes,
    const double slope_power,
    const double slope_ratio,
    const double slope_cosine,
    const double slope_sine,
    const int differences,
    const int interior_red,
    const int interior_green,
    const int interior_blue,
    const double transfer_minimum,
    const double transfer_maximum,
    const int lut_size
) {
    const size_t pixel = get_global_id(0);
    const size_t pixel_count = (size_t)width * (size_t)height;
    if (pixel >= pixel_count) return;
    const int y = (int)(pixel / (size_t)width);
    const int x = (int)(pixel - (size_t)y * (size_t)width);
    const float raw = field[pixel];
    const bool inside = isfinite(raw)
        && (double)raw >= (double)max_iter - field_bias;
    uchar* destination = output + pixel * (size_t)3;
    if (inside) {
        destination[0] = (uchar)interior_red;
        destination[1] = (uchar)interior_green;
        destination[2] = (uchar)interior_blue;
        return;
    }

    const double centre = kfp_gpu_sample(
        field, width, height, x, y, max_iter, field_bias, smooth_offset);
    const double colour_iter = flat ? floor(centre) : centre;
    const double left = kfp_gpu_reflected_sample(
        field, width, height, x, y, -1, 0, max_iter, centre,
        field_bias, smooth_offset);
    const double right = kfp_gpu_reflected_sample(
        field, width, height, x, y, 1, 0, max_iter, centre,
        field_bias, smooth_offset);
    const double up = kfp_gpu_reflected_sample(
        field, width, height, x, y, 0, -1, max_iter, centre,
        field_bias, smooth_offset);
    const double down = kfp_gpu_reflected_sample(
        field, width, height, x, y, 0, 1, max_iter, centre,
        field_bias, smooth_offset);
    const double top_left = kfp_gpu_reflected_sample(
        field, width, height, x, y, -1, -1, max_iter, centre,
        field_bias, smooth_offset);
    const double top_right = kfp_gpu_reflected_sample(
        field, width, height, x, y, 1, -1, max_iter, centre,
        field_bias, smooth_offset);
    const double bottom_left = kfp_gpu_reflected_sample(
        field, width, height, x, y, -1, 1, max_iter, centre,
        field_bias, smooth_offset);
    const double bottom_right = kfp_gpu_reflected_sample(
        field, width, height, x, y, 1, 1, max_iter, centre,
        field_bias, smooth_offset);

    double gradient = 0.0;
    if (color_method >= 5 && color_method <= 8) {
        gradient = kfp_gpu_difference(
            differences, centre, left, right, up, down,
            top_left, top_right, bottom_left, bottom_right);
    }
    double distance = gradient * (double)width / 640.0;
    if (isnan(distance) || distance < 0.0) distance = 0.0;

    double transfer = colour_iter;
    if (color_method == 1) transfer = sqrt(fmax(0.0, colour_iter));
    else if (color_method == 2) transfer = pow(fmax(0.0, colour_iter), 1.0 / 3.0);
    else if (color_method == 3) transfer = log(fmax(1.0, colour_iter));
    else if (color_method == 4) transfer = 1024.0
        * (colour_iter - transfer_minimum)
        / fmax(transfer_maximum - transfer_minimum, 1.0e-12);
    else if (color_method == 5) transfer = fmin(distance, 1024.0);
    else if (color_method == 6) {
        const double distance_transfer = fmin(sqrt(fmax(0.0, distance)), 1024.0);
        transfer = distance_transfer > iter_div ? centre : distance_transfer;
    } else if (color_method == 7) transfer = log(fmax(1.0, distance + 1.0));
    else if (color_method == 8) transfer = sqrt(fmax(0.0, distance));
    else if (color_method == 9) transfer = log(1.0 + log(1.0 + fmax(0.0, colour_iter)));
    else if (color_method == 10) transfer = atan(colour_iter);
    else if (color_method == 11) transfer = sqrt(sqrt(fmax(0.0, colour_iter)));
    if (isnan(transfer) || transfer < 0.0) transfer = 0.0;
    if (color_method == 5 || color_method == 7 || color_method == 8) {
        transfer = kfp_gpu_clamp(transfer, 0.0, 1024.0);
    }

    double palette_value = transfer;
    if (iter_division) palette_value /= iter_div;
    double position = fmod(palette_value + color_offset, (double)lut_size);
    if (position < 0.0) position += (double)lut_size;
    const double lower_position = floor(position);
    const int lower = max(0, min(lut_size - 1, (int)lower_position));
    double fraction = position - lower_position;
    if (smooth && inverse_transition) fraction = 1.0 - fraction;
    const int upper = (lower + 1) % lut_size;
    float red;
    float green;
    float blue;
    if (smooth) {
        const double inverse_fraction = 1.0 - fraction;
        red = (float)(((double)lut[lower * 3] * inverse_fraction
            + (double)lut[upper * 3] * fraction) / 255.0f);
        green = (float)(((double)lut[lower * 3 + 1] * inverse_fraction
            + (double)lut[upper * 3 + 1] * fraction) / 255.0f);
        blue = (float)(((double)lut[lower * 3 + 2] * inverse_fraction
            + (double)lut[upper * 3 + 2] * fraction) / 255.0f);
    } else {
        red = (float)lut[lower * 3] / 255.0f;
        green = (float)lut[lower * 3 + 1] / 255.0f;
        blue = (float)lut[lower * 3 + 2] / 255.0f;
    }

    if (slopes && slope_power > 0.0 && slope_ratio > 0.0) {
        const double horizontal = x > 0 ? left - centre : centre - right;
        const double vertical = y == 0 ? centre - down : up - centre;
        const double projected = (
            horizontal * slope_cosine + vertical * slope_sine)
            * slope_power * (double)width / 640.0;
        const double strength = atan(fabs(projected))
            / (3.14159265358979323846 / 2.0) * slope_ratio / 100.0;
        const double factor = 1.0 - strength;
        if (projected >= 0.0) {
            red = (float)((double)red * factor);
            green = (float)((double)green * factor);
            blue = (float)((double)blue * factor);
        } else {
            red = (float)((double)red * factor + strength);
            green = (float)((double)green * factor + strength);
            blue = (float)((double)blue * factor + strength);
        }
    }
    destination[0] = kfp_gpu_byte(red, x, y, 0);
    destination[1] = kfp_gpu_byte(green, x, y, 1);
    destination[2] = kfp_gpu_byte(blue, x, y, 2);
}
)CLC";

// RGB atlas composition is also a per-video-frame operation.  Keeping this
// sampler on the same device as the KFP colour pass avoids downloading two
// 1080p tiles to the CPU, running a second bilinear loop there, and uploading
// the result again for every frame.  The kernel deliberately mirrors the
// native RGB compositor's centred crop, child rectangle, and narrow feather.
constexpr const char* OPENCL_RGB_ATLAS_KERNEL = R"CLC(
#if defined(cl_khr_fp64)
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#elif defined(cl_amd_fp64)
#pragma OPENCL EXTENSION cl_amd_fp64 : enable
#endif

inline float rgb_clamp_float(float value, float lower, float upper) {
    return fmin(fmax(value, lower), upper);
}

inline void rgb_axis_sample(
    int destination,
    int destination_size,
    int source_size,
    float zoom,
    int* index0,
    int* index1,
    float* weight
) {
    zoom = fmax(zoom, 1.0f);
    const float crop_size = (float)source_size / zoom;
    const float left = ((float)source_size - crop_size) * 0.5f;
    float source = left
        + ((float)destination + 0.5f) * crop_size / (float)destination_size
        - 0.5f;
    source = fmin(fmax(source, 0.0f), (float)(source_size - 1));
    const int lower = (int)floor(source);
    *index0 = lower;
    *index1 = min(lower + 1, source_size - 1);
    *weight = source - (float)lower;
}

inline uchar rgb_sample_channel(
    __global const uchar* source,
    int source_width,
    int x0,
    int x1,
    int y0,
    int y1,
    float x_weight,
    float y_weight,
    int channel
) {
    const size_t top = (size_t)y0 * (size_t)source_width * 3U;
    const size_t bottom = (size_t)y1 * (size_t)source_width * 3U;
    const float top_value =
        (float)source[top + (size_t)x0 * 3U + (size_t)channel]
            * (1.0f - x_weight)
        + (float)source[top + (size_t)x1 * 3U + (size_t)channel]
            * x_weight;
    const float bottom_value =
        (float)source[bottom + (size_t)x0 * 3U + (size_t)channel]
            * (1.0f - x_weight)
        + (float)source[bottom + (size_t)x1 * 3U + (size_t)channel]
            * x_weight;
    const float value = bottom_value * y_weight
        + top_value * (1.0f - y_weight);
    return (uchar)max(0, min(255, (int)floor(value + 0.5f)));
}

inline uchar rgb_sample_channel_at(
    __global const uchar* source,
    int source_width,
    int source_height,
    int destination_x,
    int destination_y,
    int destination_width,
    int destination_height,
    double zoom,
    int channel
) {
    int x0, x1, y0, y1;
    float x_weight, y_weight;
    rgb_axis_sample(
        destination_x, destination_width, source_width, zoom,
        &x0, &x1, &x_weight);
    rgb_axis_sample(
        destination_y, destination_height, source_height, zoom,
        &y0, &y1, &y_weight);
    return rgb_sample_channel(
        source, source_width, x0, x1, y0, y1,
        x_weight, y_weight, channel);
}

inline void rgb_sample_rgb_at(
    __global const uchar* source,
    int source_width,
    int source_height,
    int destination_x,
    int destination_y,
    int destination_width,
    int destination_height,
    float zoom,
    __private uchar* output
) {
    int x0, x1, y0, y1;
    float x_weight, y_weight;
    rgb_axis_sample(
        destination_x, destination_width, source_width, zoom,
        &x0, &x1, &x_weight);
    rgb_axis_sample(
        destination_y, destination_height, source_height, zoom,
        &y0, &y1, &y_weight);
    for (int channel = 0; channel < 3; ++channel) {
        output[channel] = rgb_sample_channel(
            source, source_width, x0, x1, y0, y1,
            x_weight, y_weight, channel);
    }
}

__kernel void rgb_crop(
    __global const uchar* source,
    const int source_width,
    const int source_height,
    __global uchar* output,
    const int output_width,
    const int output_height,
    const float zoom
) {
    const size_t pixel = get_global_id(0);
    const size_t pixel_count = (size_t)output_width * (size_t)output_height;
    if (pixel >= pixel_count) return;
    const int y = (int)(pixel / (size_t)output_width);
    const int x = (int)(pixel - (size_t)y * (size_t)output_width);
    const size_t destination = pixel * 3U;
    uchar rgb[3];
    rgb_sample_rgb_at(
        source, source_width, source_height, x, y,
        output_width, output_height, (float)zoom, rgb);
    output[destination] = rgb[0];
    output[destination + 1] = rgb[1];
    output[destination + 2] = rgb[2];
}

__kernel void rgb_atlas_composite(
    __global const uchar* parent,
    const int parent_width,
    const int parent_height,
    __global const uchar* child,
    const int child_width,
    const int child_height,
    __global uchar* output,
    const int output_width,
    const int output_height,
    const float parent_zoom,
    const float child_fraction,
    const float child_zoom,
    const int feather,
    const int use_child
) {
    // The host launches this kernel as a one-dimensional work list, just
    // like rgb_crop.  Derive both coordinates from that linear pixel index;
    // asking OpenCL for get_global_id(1) from a 1-D launch leaves the second
    // coordinate at zero on several drivers and produces only the top row.
    const size_t pixel = get_global_id(0);
    const size_t pixel_count = (size_t)output_width * (size_t)output_height;
    if (pixel >= pixel_count) return;
    const int y = (int)(pixel / (size_t)output_width);
    const int x = (int)(pixel - (size_t)y * (size_t)output_width);
    const size_t destination = pixel * 3U;
    uchar parent_rgb[3];
    rgb_sample_rgb_at(
        parent, parent_width, parent_height, x, y,
        output_width, output_height, (float)parent_zoom, parent_rgb);
    output[destination] = parent_rgb[0];
    output[destination + 1] = parent_rgb[1];
    output[destination + 2] = parent_rgb[2];
    if (!use_child) return;

    const int visible_width = max(
        1, (int)floor((double)output_width * child_fraction + 0.5));
    const int visible_height = max(
        1, (int)floor((double)output_height * child_fraction + 0.5));
    const int child_left = (output_width - visible_width) / 2;
    const int child_top = (output_height - visible_height) / 2;
    if (x < child_left || x >= child_left + visible_width
        || y < child_top || y >= child_top + visible_height) return;

    const int child_x = x - child_left;
    const int child_y = y - child_top;
    uchar child_rgb[3];
    rgb_sample_rgb_at(
        child, child_width, child_height, child_x, child_y,
        visible_width, visible_height, (float)child_zoom, child_rgb);

    float alpha = 1.0f;
    const int effective_feather = min(
        feather, min(visible_width, visible_height));
    if (effective_feather >= 2) {
        const int edge_distance = min(
            min(child_x, visible_width - 1 - child_x),
            min(child_y, visible_height - 1 - child_y));
        const float linear = rgb_clamp_float(
            (float)edge_distance / (float)effective_feather,
            0.0f, 1.0f);
        alpha = linear * linear * (3.0f - 2.0f * linear);
    }
    for (int channel = 0; channel < 3; ++channel) {
        const float value = (float)parent_rgb[channel] * (1.0f - alpha)
            + (float)child_rgb[channel] * alpha;
        output[destination + (size_t)channel] =
            (uchar)max(0, min(255, (int)floor(value + 0.5f)));
    }
}
)CLC";

// Ordinary Aurora frames used to compose a complete scalar atlas on the CPU,
// upload that 8 MB surface, and then run the very small palette lookup kernel.
// Keep the source tiles resident instead: this kernel performs the same
// bilinear crop/interior handling and narrow child seam directly on the
// device, then immediately applies the Aurora LUT.  The double arithmetic is
// intentional; it matches the native atlas coordinate calculation and keeps
// very deep, narrow crops from accumulating visible pixel drift.
constexpr const char* OPENCL_ATLAS_AURORA_KERNEL = R"CLC(
#if defined(cl_khr_fp64)
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#elif defined(cl_amd_fp64)
#pragma OPENCL EXTENSION cl_amd_fp64 : enable
#endif

#if defined(FRACTAL_OPENCL_ATLAS_MAPS)
typedef struct {
    int index0;
    int index1;
    float weight;
    float padding;
} atlas_axis;
#endif

inline void atlas_aurora_axis(
    int destination,
    int destination_size,
    int source_size,
    double zoom,
    __private int* index0,
    __private int* index1,
    __private float* weight
) {
    zoom = fmax(zoom, 1.0);
    const double crop_size = (double)source_size / zoom;
    const double left = ((double)source_size - crop_size) * 0.5;
    double source = left
        + ((double)destination + 0.5) * crop_size
            / (double)destination_size
        - 0.5;
    source = fmin(fmax(source, 0.0), (double)(source_size - 1));
    const int lower = (int)floor(source);
    *index0 = lower;
    *index1 = min(lower + 1, source_size - 1);
    *weight = (float)(source - (double)lower);
}

inline float atlas_aurora_sample(
    __global const float* source,
    int source_width,
    int source_height,
    int destination_x,
    int destination_y,
    int destination_width,
    int destination_height,
    double zoom,
    int source_max_iter,
    double source_field_bias,
    __private int* inside
) {
    int x0, x1, y0, y1;
    float x_weight, y_weight;
    atlas_aurora_axis(
        destination_x, destination_width, source_width, zoom,
        &x0, &x1, &x_weight);
    atlas_aurora_axis(
        destination_y, destination_height, source_height, zoom,
        &y0, &y1, &y_weight);
    const double xw = (double)x_weight;
    const double yw = (double)y_weight;
    const float values[4] = {
        source[(size_t)y0 * (size_t)source_width + (size_t)x0],
        source[(size_t)y0 * (size_t)source_width + (size_t)x1],
        source[(size_t)y1 * (size_t)source_width + (size_t)x0],
        source[(size_t)y1 * (size_t)source_width + (size_t)x1],
    };
    const double weights[4] = {
        (1.0 - yw) * (1.0 - xw),
        (1.0 - yw) * xw,
        yw * (1.0 - xw),
        yw * xw,
    };
    double interior_weight = 0.0;
    double exterior_weight = 0.0;
    double exterior_value = 0.0;
    for (int index = 0; index < 4; ++index) {
        const float value = values[index];
        const double weight = weights[index];
        if (isfinite(value)
            && (double)value >= (double)source_max_iter - source_field_bias) {
            interior_weight += weight;
        } else if (isfinite(value)) {
            exterior_weight += weight;
            exterior_value += ((double)value + source_field_bias) * weight;
        }
    }
    if (interior_weight >= 0.5) {
        *inside = 1;
        return (float)source_max_iter;
    }
    *inside = 0;
    if (exterior_weight <= 1.0e-12) return 0.0f;
    return (float)(exterior_value / exterior_weight);
}

inline float atlas_aurora_value(
    __global const float* source,
    int source_width,
    int source_height,
    int destination_x,
    int destination_y,
    int destination_width,
    int destination_height,
    double zoom,
    int source_max_iter,
    double source_field_bias,
    int effective_iter,
    double output_field_bias
) {
    int inside = 0;
    const float sample = atlas_aurora_sample(
        source, source_width, source_height,
        destination_x, destination_y,
        destination_width, destination_height,
        zoom, source_max_iter, source_field_bias, &inside);
    return inside
        ? (float)((double)effective_iter - output_field_bias)
        : sample - (float)output_field_bias;
}

// The source atlas is already float-valued and the destination palette index
// is quantised to a small integer LUT.  On GPUs with weak fp64 throughput,
// doing the crop/interpolation in float is substantially cheaper while still
// retaining the exact source texel selection and the same interior rule.
// This entry point is compiled only for the explicit fast OpenCL atlas mode;
// the strict path above remains the default and the reference implementation.
inline float atlas_aurora_value_float(
    __global const float* source,
    int source_width,
    int source_height,
    int destination_x,
    int destination_y,
    int destination_width,
    int destination_height,
    double zoom,
    int source_max_iter,
    double source_field_bias,
    int effective_iter,
    double output_field_bias
) {
    const float effective_zoom = fmax((float)zoom, 1.0f);
    const float x_crop = (float)source_width / effective_zoom;
    const float y_crop = (float)source_height / effective_zoom;
    const float x_left = ((float)source_width - x_crop) * 0.5f;
    const float y_top = ((float)source_height - y_crop) * 0.5f;
    float x_source = x_left
        + ((float)destination_x + 0.5f) * x_crop
            / (float)destination_width
        - 0.5f;
    float y_source = y_top
        + ((float)destination_y + 0.5f) * y_crop
            / (float)destination_height
        - 0.5f;
    x_source = fmin(fmax(x_source, 0.0f), (float)(source_width - 1));
    y_source = fmin(fmax(y_source, 0.0f), (float)(source_height - 1));
    const int x0 = (int)floor(x_source);
    const int y0 = (int)floor(y_source);
    const int x1 = min(x0 + 1, source_width - 1);
    const int y1 = min(y0 + 1, source_height - 1);
    const float x_weight = x_source - (float)x0;
    const float y_weight = y_source - (float)y0;
    const float values[4] = {
        source[(size_t)y0 * (size_t)source_width + (size_t)x0],
        source[(size_t)y0 * (size_t)source_width + (size_t)x1],
        source[(size_t)y1 * (size_t)source_width + (size_t)x0],
        source[(size_t)y1 * (size_t)source_width + (size_t)x1],
    };
    const float weights[4] = {
        (1.0f - y_weight) * (1.0f - x_weight),
        (1.0f - y_weight) * x_weight,
        y_weight * (1.0f - x_weight),
        y_weight * x_weight,
    };
    const float field_bias = (float)source_field_bias;
    const float interior_threshold =
        (float)source_max_iter - field_bias;
    float interior_weight = 0.0f;
    float exterior_weight = 0.0f;
    float exterior_value = 0.0f;
    for (int index = 0; index < 4; ++index) {
        const float value = values[index];
        const float weight = weights[index];
        if (isfinite(value) && value >= interior_threshold) {
            interior_weight += weight;
        } else if (isfinite(value)) {
            exterior_weight += weight;
            exterior_value += (value + field_bias) * weight;
        }
    }
    if (interior_weight >= 0.5f) {
        return (float)effective_iter - (float)output_field_bias;
    }
    if (exterior_weight <= 1.0e-6f) return 0.0f;
    return exterior_value / exterior_weight - (float)output_field_bias;
}

#if defined(FRACTAL_OPENCL_ATLAS_FLOAT)
#define atlas_aurora_value_render atlas_aurora_value_float
#else
#define atlas_aurora_value_render atlas_aurora_value
#endif

#if defined(FRACTAL_OPENCL_ATLAS_MAPS)
inline float atlas_aurora_value_mapped(
    __global const float* source,
    int source_width,
    int destination_x,
    int destination_y,
    __global const atlas_axis* x_map,
    __global const atlas_axis* y_map,
    int source_max_iter,
    double source_field_bias,
    int effective_iter,
    double output_field_bias
) {
    const atlas_axis x_axis = x_map[destination_x];
    const atlas_axis y_axis = y_map[destination_y];
    const double xw = (double)x_axis.weight;
    const double yw = (double)y_axis.weight;
    const float values[4] = {
        source[(size_t)y_axis.index0 * (size_t)source_width
               + (size_t)x_axis.index0],
        source[(size_t)y_axis.index0 * (size_t)source_width
               + (size_t)x_axis.index1],
        source[(size_t)y_axis.index1 * (size_t)source_width
               + (size_t)x_axis.index0],
        source[(size_t)y_axis.index1 * (size_t)source_width
               + (size_t)x_axis.index1],
    };
    const double weights[4] = {
        (1.0 - yw) * (1.0 - xw),
        (1.0 - yw) * xw,
        yw * (1.0 - xw),
        yw * xw,
    };
    double interior_weight = 0.0;
    double exterior_weight = 0.0;
    double exterior_value = 0.0;
    const double interior_threshold = (double)source_max_iter
        - source_field_bias;
    for (int index = 0; index < 4; ++index) {
        const float value = values[index];
        const double weight = weights[index];
        if (isfinite(value) && (double)value >= interior_threshold) {
            interior_weight += weight;
        } else if (isfinite(value)) {
            exterior_weight += weight;
            exterior_value += ((double)value + source_field_bias) * weight;
        }
    }
    if (interior_weight >= 0.5) {
        return (float)((double)effective_iter - output_field_bias);
    }
    if (exterior_weight <= 1.0e-12) return 0.0f;
    return (float)(exterior_value / exterior_weight - output_field_bias);
}
#if defined(FRACTAL_OPENCL_ATLAS_FLOAT)
inline float atlas_aurora_value_mapped_float(
    __global const float* source,
    int source_width,
    int destination_x,
    int destination_y,
    __global const atlas_axis* x_map,
    __global const atlas_axis* y_map,
    int source_max_iter,
    double source_field_bias,
    int effective_iter,
    double output_field_bias
) {
    const atlas_axis x_axis = x_map[destination_x];
    const atlas_axis y_axis = y_map[destination_y];
    const float xw = x_axis.weight;
    const float yw = y_axis.weight;
    const float values[4] = {
        source[(size_t)y_axis.index0 * (size_t)source_width
               + (size_t)x_axis.index0],
        source[(size_t)y_axis.index0 * (size_t)source_width
               + (size_t)x_axis.index1],
        source[(size_t)y_axis.index1 * (size_t)source_width
               + (size_t)x_axis.index0],
        source[(size_t)y_axis.index1 * (size_t)source_width
               + (size_t)x_axis.index1],
    };
    const float weights[4] = {
        (1.0f - yw) * (1.0f - xw),
        (1.0f - yw) * xw,
        yw * (1.0f - xw),
        yw * xw,
    };
    const float field_bias = (float)source_field_bias;
    const float interior_threshold =
        (float)source_max_iter - field_bias;
    float interior_weight = 0.0f;
    float exterior_weight = 0.0f;
    float exterior_value = 0.0f;
    for (int index = 0; index < 4; ++index) {
        const float value = values[index];
        const float weight = weights[index];
        if (isfinite(value) && value >= interior_threshold) {
            interior_weight += weight;
        } else if (isfinite(value)) {
            exterior_weight += weight;
            exterior_value += (value + field_bias) * weight;
        }
    }
    if (interior_weight >= 0.5f) {
        return (float)effective_iter - (float)output_field_bias;
    }
    if (exterior_weight <= 1.0e-6f) return 0.0f;
    return exterior_value / exterior_weight - (float)output_field_bias;
}
#define atlas_aurora_value_mapped_render atlas_aurora_value_mapped_float
#else
#define atlas_aurora_value_mapped_render atlas_aurora_value_mapped
#endif
#endif

inline uchar4 atlas_aurora_lookup(
    float smooth,
    int effective_iter,
    __global const uchar* palette,
    ulong palette_offset,
    int palette_size,
    float palette_index_scale,
    int interior_red,
    int interior_green,
    int interior_blue
) {
    if (!isfinite(smooth) || smooth >= (float)effective_iter) {
        return (uchar4)(
            (uchar)clamp(interior_red, 0, 255),
            (uchar)clamp(interior_green, 0, 255),
            (uchar)clamp(interior_blue, 0, 255),
            (uchar)0);
    }
    // The palette scale is generated as float on the host and the field is
    // already float-valued. Keep this hot lookup in single precision; the
    // old double promotion made every 4K pixel pay the RTX FP64 penalty.
    const float scaled = smooth * palette_index_scale;
    const int index = scaled >= (float)(palette_size - 1)
        ? palette_size - 1
        : (!isfinite(scaled) || scaled <= 0.0 ? 0 : (int)scaled);
    const size_t source = palette_offset + (size_t)index * (size_t)3;
    return (uchar4)(
        palette[source], palette[source + 1], palette[source + 2], (uchar)0);
}

inline uchar atlas_aurora_blend_channel(
    uchar parent,
    uchar child,
    float alpha
) {
    const double value = (double)parent * (1.0 - (double)alpha)
        + (double)child * (double)alpha;
    return (uchar)rint(fmin(fmax(value, 0.0), 255.0));
}

#if defined(FRACTAL_OPENCL_ATLAS_IMAGES)
// The ordinary Aurora source is a scalar field with an interior marker.  The
// image path stores three bilinearly sampled quantities in an RGBA image:
// exterior value, exterior weight, and interior weight.  That preserves the
// scalar compositor's treatment of NaNs and interior pixels while allowing
// the hardware sampler to do the four-tap interpolation in one operation.
__constant sampler_t atlas_aurora_sampler =
    CLK_NORMALIZED_COORDS_FALSE
    | CLK_ADDRESS_CLAMP
    | CLK_FILTER_LINEAR;

inline float atlas_aurora_image_value(
    read_only image2d_t source,
    int source_width,
    int source_height,
    int destination_x,
    int destination_y,
    int destination_width,
    int destination_height,
    double zoom,
    int source_max_iter,
    int effective_iter
) {
    const float effective_zoom = fmax((float)zoom, 1.0f);
    const float crop_width = (float)source_width / effective_zoom;
    const float crop_height = (float)source_height / effective_zoom;
    const float left = ((float)source_width - crop_width) * 0.5f;
    const float top = ((float)source_height - crop_height) * 0.5f;
    float source_x = left
        + ((float)destination_x + 0.5f) * crop_width
            / (float)destination_width
        - 0.5f;
    float source_y = top
        + ((float)destination_y + 0.5f) * crop_height
            / (float)destination_height
        - 0.5f;
    source_x = fmin(fmax(source_x, 0.0f), (float)(source_width - 1));
    source_y = fmin(fmax(source_y, 0.0f), (float)(source_height - 1));
    const float4 sample = read_imagef(
        source,
        atlas_aurora_sampler,
        (float2)(source_x + 0.5f, source_y + 0.5f));
    if (sample.z >= 0.5f) {
        return (float)effective_iter;
    }
    if (sample.y <= 1.0e-6f) return 0.0f;
    (void)source_max_iter;
    return sample.x / sample.y;
}

__kernel void aurora_atlas_colourise_image(
    read_only image2d_t parent,
    const int parent_width,
    const int parent_height,
    const int parent_max_iter,
    read_only image2d_t child,
    const int child_width,
    const int child_height,
    const int child_max_iter,
    __global const uchar* palette,
    __global uchar* output,
    const int output_width,
    const int output_height,
    const double parent_zoom,
    const double child_fraction,
    const double child_zoom,
    const double parent_field_bias,
    const double child_field_bias,
    const double output_field_bias,
    const int effective_iter,
    const int feather,
    const int palette_size,
    const float palette_index_scale,
    const int interior_red,
    const int interior_green,
    const int interior_blue,
    const int use_child,
    const ulong output_offset,
    const ulong palette_offset
#if defined(FRACTAL_OPENCL_ATLAS_MAPS)
    , __global const atlas_axis* parent_x_map
    , __global const atlas_axis* parent_y_map
    , __global const atlas_axis* child_x_map
    , __global const atlas_axis* child_y_map
    , const int parent_x_map_offset
    , const int parent_y_map_offset
    , const int child_x_map_offset
    , const int child_y_map_offset
#endif
) {
    const size_t pixel_count = (size_t)output_width * (size_t)output_height;
    const int linear_launch = get_work_dim() == 1;
    const size_t linear_id = get_global_id(0);
    const int x = linear_launch
        ? (int)(linear_id % (size_t)output_width)
        : (int)linear_id;
    const int y = linear_launch
        ? (int)(linear_id / (size_t)output_width)
        : (int)get_global_id(1);
    if (linear_launch
        ? linear_id >= pixel_count
        : (x >= output_width || y >= output_height)) return;
    const size_t pixel = (size_t)y * (size_t)output_width + (size_t)x;
    const size_t destination = output_offset + pixel * (size_t)3;
    const int full_child = use_child && child_fraction >= 0.999999;
    float smooth;
    if (full_child) {
        smooth = atlas_aurora_image_value(
            child, child_width, child_height, x, y,
            output_width, output_height, child_zoom, child_max_iter,
            effective_iter);
        const uchar4 colour = atlas_aurora_lookup(
            smooth, effective_iter, palette, palette_offset, palette_size,
            palette_index_scale, interior_red, interior_green, interior_blue);
        output[destination] = colour.x;
        output[destination + 1] = colour.y;
        output[destination + 2] = colour.z;
        return;
    }

    if (use_child) {
        const int visible_width = max(
            1, (int)floor((double)output_width * child_fraction + 0.5));
        const int visible_height = max(
            1, (int)floor((double)output_height * child_fraction + 0.5));
        const int child_left = (output_width - visible_width) / 2;
        const int child_top = (output_height - visible_height) / 2;
        if (x >= child_left && x < child_left + visible_width
            && y >= child_top && y < child_top + visible_height) {
            const int child_x = x - child_left;
            const int child_y = y - child_top;
            const float child_value = atlas_aurora_image_value(
                child, child_width, child_height, child_x, child_y,
                visible_width, visible_height, child_zoom, child_max_iter,
                effective_iter);
            const int seam_feather = min(
                feather, min(visible_width / 8, visible_height / 8));
            float alpha = 1.0f;
            if (seam_feather >= 2) {
                const int edge_distance = min(
                    min(child_x, visible_width - 1 - child_x),
                    min(child_y, visible_height - 1 - child_y));
                const float linear = fmin(
                    1.0f,
                    (float)edge_distance / (float)seam_feather);
                const float eased = linear * linear
                    * (3.0f - 2.0f * linear);
                alpha = eased;
            }
            if (alpha >= 0.999999f) {
                const uchar4 colour = atlas_aurora_lookup(
                    child_value, effective_iter, palette, palette_offset,
                    palette_size, palette_index_scale, interior_red,
                    interior_green, interior_blue);
                output[destination] = colour.x;
                output[destination + 1] = colour.y;
                output[destination + 2] = colour.z;
                return;
            }
            smooth = atlas_aurora_image_value(
                parent, parent_width, parent_height, x, y,
                output_width, output_height, parent_zoom, parent_max_iter,
                effective_iter);
            const uchar4 parent_colour = atlas_aurora_lookup(
                smooth, effective_iter, palette, palette_offset, palette_size,
                palette_index_scale, interior_red, interior_green, interior_blue);
            const uchar4 child_colour = atlas_aurora_lookup(
                child_value, effective_iter, palette, palette_offset,
                palette_size, palette_index_scale, interior_red,
                interior_green, interior_blue);
            output[destination] = atlas_aurora_blend_channel(
                parent_colour.x, child_colour.x, alpha);
            output[destination + 1] = atlas_aurora_blend_channel(
                parent_colour.y, child_colour.y, alpha);
            output[destination + 2] = atlas_aurora_blend_channel(
                parent_colour.z, child_colour.z, alpha);
            return;
        }
    }

    smooth = atlas_aurora_image_value(
        parent, parent_width, parent_height, x, y,
        output_width, output_height, parent_zoom, parent_max_iter,
        effective_iter);
    const uchar4 colour = atlas_aurora_lookup(
        smooth, effective_iter, palette, palette_offset, palette_size,
        palette_index_scale, interior_red, interior_green, interior_blue);
    output[destination] = colour.x;
    output[destination + 1] = colour.y;
    output[destination + 2] = colour.z;
    (void)parent_field_bias;
    (void)child_field_bias;
    (void)output_field_bias;
}
#endif

__kernel void aurora_atlas_colourise(
    __global const float* parent,
    const int parent_width,
    const int parent_height,
    const int parent_max_iter,
    __global const float* child,
    const int child_width,
    const int child_height,
    const int child_max_iter,
    __global const uchar* palette,
    __global uchar* output,
    const int output_width,
    const int output_height,
    const double parent_zoom,
    const double child_fraction,
    const double child_zoom,
    const double parent_field_bias,
    const double child_field_bias,
    const double output_field_bias,
    const int effective_iter,
    const int feather,
    const int palette_size,
    const float palette_index_scale,
    const int interior_red,
    const int interior_green,
    const int interior_blue,
    const int use_child,
    const ulong output_offset,
    const ulong palette_offset
#if defined(FRACTAL_OPENCL_ATLAS_MAPS)
    , __global const atlas_axis* parent_x_map
    , __global const atlas_axis* parent_y_map
    , __global const atlas_axis* child_x_map
    , __global const atlas_axis* child_y_map
    , const int parent_x_map_offset
    , const int parent_y_map_offset
    , const int child_x_map_offset
    , const int child_y_map_offset
#endif
) {
    const size_t pixel_count = (size_t)output_width * (size_t)output_height;
    const int linear_launch = get_work_dim() == 1;
    const size_t linear_id = get_global_id(0);
    const int x = linear_launch
        ? (int)(linear_id % (size_t)output_width)
        : (int)linear_id;
    const int y = linear_launch
        ? (int)(linear_id / (size_t)output_width)
        : (int)get_global_id(1);
    if (linear_launch
        ? linear_id >= pixel_count
        : (x >= output_width || y >= output_height)) return;
    const size_t pixel = (size_t)y * (size_t)output_width + (size_t)x;

    const int full_child = use_child && child_fraction >= 0.999999;
    float smooth;
    const size_t destination = output_offset + pixel * (size_t)3;
    if (full_child) {
#if defined(FRACTAL_OPENCL_ATLAS_MAPS)
        smooth = atlas_aurora_value_mapped_render(
            child, child_width, x, y,
            child_x_map + child_x_map_offset,
            child_y_map + child_y_map_offset,
            child_max_iter, child_field_bias, effective_iter, output_field_bias);
#else
        smooth = atlas_aurora_value_render(
            child, child_width, child_height, x, y,
            output_width, output_height, child_zoom, child_max_iter,
            child_field_bias, effective_iter, output_field_bias);
#endif
        const uchar4 colour = atlas_aurora_lookup(
            smooth, effective_iter, palette, palette_offset, palette_size,
            palette_index_scale, interior_red, interior_green, interior_blue);
        output[destination] = colour.x;
        output[destination + 1] = colour.y;
        output[destination + 2] = colour.z;
        return;
    }

    if (use_child) {
        const int visible_width = max(
            1, (int)floor((double)output_width * child_fraction + 0.5));
        const int visible_height = max(
            1, (int)floor((double)output_height * child_fraction + 0.5));
        const int child_left = (output_width - visible_width) / 2;
        const int child_top = (output_height - visible_height) / 2;
        if (x >= child_left && x < child_left + visible_width
            && y >= child_top && y < child_top + visible_height) {
            const int child_x = x - child_left;
            const int child_y = y - child_top;
#if defined(FRACTAL_OPENCL_ATLAS_MAPS)
            const float child_value = atlas_aurora_value_mapped_render(
                child, child_width, child_x, child_y,
                child_x_map + child_x_map_offset,
                child_y_map + child_y_map_offset,
                child_max_iter,
                child_field_bias, effective_iter, output_field_bias);
#else
            const float child_value = atlas_aurora_value_render(
                child, child_width, child_height, child_x, child_y,
                visible_width, visible_height, child_zoom,
                child_max_iter, child_field_bias,
                effective_iter, output_field_bias);
#endif
            const int seam_feather = min(
                feather, min(visible_width / 8, visible_height / 8));
            float alpha = 1.0f;
            if (seam_feather >= 2) {
                const int edge_distance = min(
                    min(child_x, visible_width - 1 - child_x),
                    min(child_y, visible_height - 1 - child_y));
                const float linear = fmin(
                    1.0f,
                    (float)edge_distance / (float)seam_feather);
                const float eased = linear * linear
                    * (3.0f - 2.0f * linear);
                alpha = eased;
            }
            if (alpha >= 0.999999f) {
                const uchar4 colour = atlas_aurora_lookup(
                    child_value, effective_iter, palette, palette_offset,
                    palette_size,
                    palette_index_scale, interior_red, interior_green,
                    interior_blue);
                output[destination] = colour.x;
                output[destination + 1] = colour.y;
                output[destination + 2] = colour.z;
                return;
            }
#if defined(FRACTAL_OPENCL_ATLAS_MAPS)
            smooth = atlas_aurora_value_mapped_render(
                parent, parent_width, x, y,
                parent_x_map + parent_x_map_offset,
                parent_y_map + parent_y_map_offset,
                parent_max_iter, parent_field_bias,
                effective_iter, output_field_bias);
#else
            smooth = atlas_aurora_value_render(
                parent, parent_width, parent_height, x, y,
                output_width, output_height, parent_zoom, parent_max_iter,
                parent_field_bias, effective_iter, output_field_bias);
#endif
            const uchar4 parent_colour = atlas_aurora_lookup(
                smooth, effective_iter, palette, palette_offset, palette_size,
                palette_index_scale, interior_red, interior_green,
                interior_blue);
            const uchar4 child_colour = atlas_aurora_lookup(
                child_value, effective_iter, palette, palette_offset,
                palette_size,
                palette_index_scale, interior_red, interior_green,
                interior_blue);
            output[destination] = atlas_aurora_blend_channel(
                parent_colour.x, child_colour.x, alpha);
            output[destination + 1] = atlas_aurora_blend_channel(
                parent_colour.y, child_colour.y, alpha);
            output[destination + 2] = atlas_aurora_blend_channel(
                parent_colour.z, child_colour.z, alpha);
            return;
        }
    }

#if defined(FRACTAL_OPENCL_ATLAS_MAPS)
    smooth = atlas_aurora_value_mapped_render(
        parent, parent_width, x, y,
        parent_x_map + parent_x_map_offset,
        parent_y_map + parent_y_map_offset,
        parent_max_iter, parent_field_bias, effective_iter, output_field_bias);
#else
    smooth = atlas_aurora_value_render(
        parent, parent_width, parent_height, x, y,
        output_width, output_height, parent_zoom, parent_max_iter,
        parent_field_bias, effective_iter, output_field_bias);
#endif

    const uchar4 colour = atlas_aurora_lookup(
        smooth, effective_iter, palette, palette_offset, palette_size,
        palette_index_scale, interior_red, interior_green, interior_blue
    );
    output[destination] = colour.x;
    output[destination + 1] = colour.y;
    output[destination + 2] = colour.z;
}
#undef atlas_aurora_value_render
#if defined(FRACTAL_OPENCL_ATLAS_MAPS)
#undef atlas_aurora_value_mapped_render
#endif
)CLC";

// The ordinary atlas crop uses separable coordinates: the horizontal source
// coordinate depends only on x and the vertical source coordinate only on y.
// The original fused kernel recalculated both axes for every pixel, including
// the same fp64 divisions and floors for every pixel in one workgroup.  This
// variant computes the two axes once per local row/column, then reuses them
// from local memory.  It keeps the old kernel available for drivers that do
// not accept the fixed 16-column launch shape.
constexpr const char* OPENCL_ATLAS_TILED_KERNEL = R"CLC(
#define ATLAS_TILED_LOCAL_X 16
#define ATLAS_TILED_LOCAL_Y 16

inline float atlas_aurora_value_from_axes_tiled(
    __global const float* source,
    int source_width,
    int x0,
    int x1,
    float x_weight,
    int y0,
    int y1,
    float y_weight,
    int source_max_iter,
    double source_field_bias,
    int effective_iter,
    double output_field_bias
) {
    const float values[4] = {
        source[(size_t)y0 * (size_t)source_width + (size_t)x0],
        source[(size_t)y0 * (size_t)source_width + (size_t)x1],
        source[(size_t)y1 * (size_t)source_width + (size_t)x0],
        source[(size_t)y1 * (size_t)source_width + (size_t)x1],
    };
#if defined(FRACTAL_OPENCL_ATLAS_FLOAT)
    const float xw = x_weight;
    const float yw = y_weight;
    const float weights[4] = {
        (1.0f - yw) * (1.0f - xw),
        (1.0f - yw) * xw,
        yw * (1.0f - xw),
        yw * xw,
    };
    const float field_bias = (float)source_field_bias;
    const float interior_threshold =
        (float)source_max_iter - field_bias;
    float interior_weight = 0.0f;
    float exterior_weight = 0.0f;
    float exterior_value = 0.0f;
    for (int index = 0; index < 4; ++index) {
        const float value = values[index];
        const float weight = weights[index];
        if (isfinite(value) && value >= interior_threshold) {
            interior_weight += weight;
        } else if (isfinite(value)) {
            exterior_weight += weight;
            exterior_value += (value + field_bias) * weight;
        }
    }
    if (interior_weight >= 0.5f) {
        return (float)effective_iter - (float)output_field_bias;
    }
    if (exterior_weight <= 1.0e-6f) return 0.0f;
    return exterior_value / exterior_weight - (float)output_field_bias;
#else
    const double xw = (double)x_weight;
    const double yw = (double)y_weight;
    const double weights[4] = {
        (1.0 - yw) * (1.0 - xw),
        (1.0 - yw) * xw,
        yw * (1.0 - xw),
        yw * xw,
    };
    double interior_weight = 0.0;
    double exterior_weight = 0.0;
    double exterior_value = 0.0;
    for (int index = 0; index < 4; ++index) {
        const float value = values[index];
        const double weight = weights[index];
        if (isfinite(value)
            && (double)value >= (double)source_max_iter - source_field_bias) {
            interior_weight += weight;
        } else if (isfinite(value)) {
            exterior_weight += weight;
            exterior_value += ((double)value + source_field_bias) * weight;
        }
    }
    if (interior_weight >= 0.5) {
        return (float)((double)effective_iter - output_field_bias);
    }
    if (exterior_weight <= 1.0e-12) return 0.0f;
    return (float)(exterior_value / exterior_weight - output_field_bias);
#endif
}

__kernel void aurora_atlas_colourise_tiled(
    __global const float* parent,
    const int parent_width,
    const int parent_height,
    const int parent_max_iter,
    __global const float* child,
    const int child_width,
    const int child_height,
    const int child_max_iter,
    __global const uchar* palette,
    __global uchar* output,
    const int output_width,
    const int output_height,
    const double parent_zoom,
    const double child_fraction,
    const double child_zoom,
    const double parent_field_bias,
    const double child_field_bias,
    const double output_field_bias,
    const int effective_iter,
    const int feather,
    const int palette_size,
    const float palette_index_scale,
    const int interior_red,
    const int interior_green,
    const int interior_blue,
    const int use_child,
    const ulong output_offset,
    const ulong palette_offset
#if defined(FRACTAL_OPENCL_ATLAS_MAPS)
    , __global const atlas_axis* parent_x_map
    , __global const atlas_axis* parent_y_map
    , __global const atlas_axis* child_x_map
    , __global const atlas_axis* child_y_map
    , const int parent_x_map_offset
    , const int parent_y_map_offset
    , const int child_x_map_offset
    , const int child_y_map_offset
#endif
) {
    const int local_x = (int)get_local_id(0);
    const int local_y = (int)get_local_id(1);
    const int x = (int)get_global_id(0);
    const int y = (int)get_global_id(1);
    const int clamped_x = min(max(x, 0), output_width - 1);
    const int clamped_y = min(max(y, 0), output_height - 1);
    const int full_child = use_child && child_fraction >= 0.999999;
    const int visible_width = max(
        1, (int)floor((double)output_width * child_fraction + 0.5));
    const int visible_height = max(
        1, (int)floor((double)output_height * child_fraction + 0.5));
    const int child_left = (output_width - visible_width) / 2;
    const int child_top = (output_height - visible_height) / 2;
    const int safe_child_width = child_width > 0 ? child_width : parent_width;
    const int safe_child_height = child_height > 0 ? child_height : parent_height;
    const int child_x = full_child
        ? clamped_x
        : min(max(clamped_x - child_left, 0), visible_width - 1);
    const int child_y = full_child
        ? clamped_y
        : min(max(clamped_y - child_top, 0), visible_height - 1);
    const int child_destination_width = full_child
        ? output_width : visible_width;
    const int child_destination_height = full_child
        ? output_height : visible_height;

    __local int parent_x0[ATLAS_TILED_LOCAL_X];
    __local int parent_x1[ATLAS_TILED_LOCAL_X];
    __local float parent_x_weight[ATLAS_TILED_LOCAL_X];
    __local int parent_y0[ATLAS_TILED_LOCAL_Y];
    __local int parent_y1[ATLAS_TILED_LOCAL_Y];
    __local float parent_y_weight[ATLAS_TILED_LOCAL_Y];
    __local int child_x0[ATLAS_TILED_LOCAL_X];
    __local int child_x1[ATLAS_TILED_LOCAL_X];
    __local float child_x_weight[ATLAS_TILED_LOCAL_X];
    __local int child_y0[ATLAS_TILED_LOCAL_Y];
    __local int child_y1[ATLAS_TILED_LOCAL_Y];
    __local float child_y_weight[ATLAS_TILED_LOCAL_Y];

    if (local_y == 0) {
#if defined(FRACTAL_OPENCL_ATLAS_MAPS)
        const atlas_axis parent_axis =
            parent_x_map[parent_x_map_offset + clamped_x];
        const atlas_axis child_axis =
            child_x_map[child_x_map_offset + child_x];
        parent_x0[local_x] = parent_axis.index0;
        parent_x1[local_x] = parent_axis.index1;
        parent_x_weight[local_x] = parent_axis.weight;
        child_x0[local_x] = child_axis.index0;
        child_x1[local_x] = child_axis.index1;
        child_x_weight[local_x] = child_axis.weight;
#else
        int x0, x1;
        float weight;
        atlas_aurora_axis(
            clamped_x, output_width, parent_width, parent_zoom,
            &x0, &x1, &weight);
        parent_x0[local_x] = x0;
        parent_x1[local_x] = x1;
        parent_x_weight[local_x] = weight;
        atlas_aurora_axis(
            child_x, child_destination_width, safe_child_width, child_zoom,
            &x0, &x1, &weight);
        child_x0[local_x] = x0;
        child_x1[local_x] = x1;
        child_x_weight[local_x] = weight;
#endif
    }
    if (local_x == 0) {
#if defined(FRACTAL_OPENCL_ATLAS_MAPS)
        const atlas_axis parent_axis =
            parent_y_map[parent_y_map_offset + clamped_y];
        const atlas_axis child_axis =
            child_y_map[child_y_map_offset + child_y];
        parent_y0[local_y] = parent_axis.index0;
        parent_y1[local_y] = parent_axis.index1;
        parent_y_weight[local_y] = parent_axis.weight;
        child_y0[local_y] = child_axis.index0;
        child_y1[local_y] = child_axis.index1;
        child_y_weight[local_y] = child_axis.weight;
#else
        int y0, y1;
        float weight;
        atlas_aurora_axis(
            clamped_y, output_height, parent_height, parent_zoom,
            &y0, &y1, &weight);
        parent_y0[local_y] = y0;
        parent_y1[local_y] = y1;
        parent_y_weight[local_y] = weight;
        atlas_aurora_axis(
            child_y, child_destination_height, safe_child_height, child_zoom,
            &y0, &y1, &weight);
        child_y0[local_y] = y0;
        child_y1[local_y] = y1;
        child_y_weight[local_y] = weight;
#endif
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (x >= output_width || y >= output_height) return;

    const size_t pixel = (size_t)y * (size_t)output_width + (size_t)x;
    const size_t destination = output_offset + pixel * (size_t)3;
    float smooth;
    if (full_child) {
        smooth = atlas_aurora_value_from_axes_tiled(
            child, safe_child_width,
            child_x0[local_x], child_x1[local_x], child_x_weight[local_x],
            child_y0[local_y], child_y1[local_y], child_y_weight[local_y],
            child_max_iter, child_field_bias, effective_iter, output_field_bias);
        const uchar4 colour = atlas_aurora_lookup(
            smooth, effective_iter, palette, palette_offset, palette_size,
            palette_index_scale, interior_red, interior_green, interior_blue);
        output[destination] = colour.x;
        output[destination + 1] = colour.y;
        output[destination + 2] = colour.z;
        return;
    }

    if (use_child
        && x >= child_left && x < child_left + visible_width
        && y >= child_top && y < child_top + visible_height) {
        const int local_child_x = x - child_left;
        const int local_child_y = y - child_top;
        const float child_value = atlas_aurora_value_from_axes_tiled(
            child, safe_child_width,
            child_x0[local_x], child_x1[local_x], child_x_weight[local_x],
            child_y0[local_y], child_y1[local_y], child_y_weight[local_y],
            child_max_iter, child_field_bias, effective_iter, output_field_bias);
        const int seam_feather = min(
            feather, min(visible_width / 8, visible_height / 8));
        float alpha = 1.0f;
        if (seam_feather >= 2) {
            const int edge_distance = min(
                min(local_child_x, visible_width - 1 - local_child_x),
                min(local_child_y, visible_height - 1 - local_child_y));
            const float linear = fmin(
                1.0f, (float)edge_distance / (float)seam_feather);
            const float eased = linear * linear * (3.0f - 2.0f * linear);
            alpha = eased;
        }
        if (alpha >= 0.999999f) {
            const uchar4 colour = atlas_aurora_lookup(
                child_value, effective_iter, palette, palette_offset,
                palette_size, palette_index_scale, interior_red,
                interior_green, interior_blue);
            output[destination] = colour.x;
            output[destination + 1] = colour.y;
            output[destination + 2] = colour.z;
            return;
        }
        smooth = atlas_aurora_value_from_axes_tiled(
            parent, parent_width,
            parent_x0[local_x], parent_x1[local_x], parent_x_weight[local_x],
            parent_y0[local_y], parent_y1[local_y], parent_y_weight[local_y],
            parent_max_iter, parent_field_bias, effective_iter, output_field_bias);
        const uchar4 parent_colour = atlas_aurora_lookup(
            smooth, effective_iter, palette, palette_offset, palette_size,
            palette_index_scale, interior_red, interior_green, interior_blue);
        const uchar4 child_colour = atlas_aurora_lookup(
            child_value, effective_iter, palette, palette_offset, palette_size,
            palette_index_scale, interior_red, interior_green, interior_blue);
        output[destination] = atlas_aurora_blend_channel(
            parent_colour.x, child_colour.x, alpha);
        output[destination + 1] = atlas_aurora_blend_channel(
            parent_colour.y, child_colour.y, alpha);
        output[destination + 2] = atlas_aurora_blend_channel(
            parent_colour.z, child_colour.z, alpha);
        return;
    }

    smooth = atlas_aurora_value_from_axes_tiled(
        parent, parent_width,
        parent_x0[local_x], parent_x1[local_x], parent_x_weight[local_x],
        parent_y0[local_y], parent_y1[local_y], parent_y_weight[local_y],
        parent_max_iter, parent_field_bias, effective_iter, output_field_bias);
    const uchar4 colour = atlas_aurora_lookup(
        smooth, effective_iter, palette, palette_offset, palette_size,
        palette_index_scale, interior_red, interior_green, interior_blue);
    output[destination] = colour.x;
    output[destination + 1] = colour.y;
    output[destination + 2] = colour.z;
}

#undef ATLAS_TILED_LOCAL_X
#undef ATLAS_TILED_LOCAL_Y
)CLC";

struct OpenClAtlasAxis {
    std::int32_t index0 = 0;
    std::int32_t index1 = 0;
    float weight = 0.0f;
    float padding = 0.0f;
};

static_assert(sizeof(OpenClAtlasAxis) == 16,
              "OpenCL atlas axis layout must remain a 16-byte record");

void fill_opencl_atlas_axis(
    std::vector<OpenClAtlasAxis>& output,
    int destination_size,
    int source_size,
    double zoom
) {
    output.resize(static_cast<size_t>(destination_size));
    const double effective_zoom = std::max(zoom, 1.0);
    const double crop_size = static_cast<double>(source_size) / effective_zoom;
    const double left = (static_cast<double>(source_size) - crop_size) * 0.5;
    for (int index = 0; index < destination_size; ++index) {
        double source = left
            + (static_cast<double>(index) + 0.5) * crop_size
                / static_cast<double>(destination_size)
            - 0.5;
        source = std::min(
            std::max(source, 0.0), static_cast<double>(source_size - 1));
        const int lower = static_cast<int>(std::floor(source));
        OpenClAtlasAxis& axis = output[static_cast<size_t>(index)];
        axis.index0 = lower;
        axis.index1 = std::min(lower + 1, source_size - 1);
        axis.weight = static_cast<float>(source - static_cast<double>(lower));
    }
}

std::vector<float> pack_opencl_atlas_image(
    const float* source,
    int width,
    int height,
    int max_iter
) {
    const size_t pixel_count = static_cast<size_t>(width)
        * static_cast<size_t>(height);
    if (pixel_count > std::numeric_limits<size_t>::max() / 4U) {
        throw std::runtime_error("OpenCL atlas image is too large");
    }
    std::vector<float> packed(pixel_count * 4U, 0.0f);
    for (size_t index = 0; index < pixel_count; ++index) {
        const float value = source[index];
        if (!std::isfinite(value)) continue;
        float* pixel = packed.data() + index * 4U;
        if (value >= static_cast<float>(max_iter)) {
            pixel[2] = 1.0f;
        } else {
            pixel[0] = value;
            pixel[1] = 1.0f;
        }
    }
    return packed;
}

bool opencl_atlas_force_2d_launch() {
    const char* setting = std::getenv("FRACTAL_OPENCL_ATLAS_2D");
    if (!setting || setting[0] == '\0') return false;
    return std::strcmp(setting, "0") != 0
        && std::strcmp(setting, "false") != 0
        && std::strcmp(setting, "off") != 0
        && std::strcmp(setting, "no") != 0;
}

size_t opencl_atlas_tiled_workgroup(
    size_t workgroup_limit,
    size_t pixel_count
) {
    if (workgroup_limit < 16) return 0;
    const size_t target = pixel_count <= static_cast<size_t>(1920) * 1080
        ? 256U : 128U;
    size_t workgroup = std::min(workgroup_limit, target);
    workgroup = std::min(workgroup, static_cast<size_t>(256));
    workgroup -= workgroup % 16U;
    return workgroup >= 16 ? workgroup : 0;
}

bool opencl_atlas_use_secondary_queue() {
    const char* setting = std::getenv("FRACTAL_OPENCL_ATLAS_SINGLE_QUEUE");
    if (!setting || setting[0] == '\0') return true;
    return std::strcmp(setting, "0") == 0
        || std::strcmp(setting, "false") == 0
        || std::strcmp(setting, "off") == 0
        || std::strcmp(setting, "no") == 0;
}

bool opencl_atlas_use_pipelined_batch() {
    const char* setting = std::getenv("FRACTAL_OPENCL_ATLAS_PIPELINE");
    // The complete-batch schedule is the measured fast path on the current
    // NVIDIA implementation. Keep an explicit opt-out for older ICDs that
    // need the pairwise compatibility schedule.
    if (!setting || setting[0] == '\0') return true;
    return std::strcmp(setting, "0") != 0
        && std::strcmp(setting, "false") != 0
        && std::strcmp(setting, "off") != 0
        && std::strcmp(setting, "no") != 0;
}

struct OpenClRuntime {
    cl_context context = nullptr;
    cl_command_queue queue = nullptr;
    // The batched atlas compositor can use a second in-order queue to overlap
    // one frame's readback with the next frame's colour kernel.  It is
    // deliberately optional: older ICDs may expose only one usable queue.
    cl_command_queue atlas_queue_secondary = nullptr;
    cl_program program = nullptr;
    cl_kernel kernel = nullptr;
    std::array<cl_kernel, 4> direct_formula_kernels{};
    cl_kernel deep_kernel = nullptr;
    cl_kernel deep_scalar_kernel = nullptr;
    cl_kernel deep_scalar_bla_kernel = nullptr;
    cl_kernel deep_mixed_kernel = nullptr;
    cl_kernel deep_mixed_mandelbrot_kernel = nullptr;
    cl_kernel colour_kernel = nullptr;
    cl_kernel kfp_colour_kernel = nullptr;
    cl_kernel rgb_crop_kernel = nullptr;
    cl_kernel rgb_composite_kernel = nullptr;
    cl_kernel atlas_colour_kernel = nullptr;
    cl_kernel atlas_colour_kernel_secondary = nullptr;
    cl_kernel atlas_tiled_colour_kernel = nullptr;
    cl_kernel atlas_tiled_colour_kernel_secondary = nullptr;
    cl_kernel atlas_image_colour_kernel = nullptr;
    cl_kernel atlas_image_colour_kernel_secondary = nullptr;
    // Zero means that the driver could not validate a common explicit local
    // size for all field kernels; callers then let OpenCL choose one.
    size_t workgroup_size = 0;
    size_t deep_scalar_workgroup_size = 0;
    size_t deep_scalar_bla_workgroup_size = 0;
    size_t deep_mixed_workgroup_size = 0;
    size_t deep_mixed_mandelbrot_workgroup_size = 0;
    size_t colour_workgroup_size = 0;
    size_t kfp_colour_workgroup_size = 0;
    size_t rgb_workgroup_size = 0;
    size_t atlas_colour_workgroup_size = 0;
    size_t atlas_colour_workgroup_limit = 0;
    size_t atlas_tiled_colour_workgroup_limit = 0;
    bool device_is_gpu = false;
    // Ordinary live/export fields are rendered repeatedly at the same source
    // size. Keep their device destination alive between calls instead of
    // paying a driver allocation/free round-trip for every atlas tile.
    cl_mem direct_output = nullptr;
    size_t direct_output_capacity = 0;
    cl_mem deep_output = nullptr;
    size_t deep_output_capacity = 0;
    cl_mem colour_field = nullptr;
    size_t colour_field_capacity = 0;
    cl_mem colour_palette = nullptr;
    size_t colour_palette_capacity = 0;
    cl_mem colour_output = nullptr;
    size_t colour_output_capacity = 0;
    cl_mem rgb_parent = nullptr;
    size_t rgb_parent_capacity = 0;
    cl_mem rgb_child = nullptr;
    size_t rgb_child_capacity = 0;
    cl_mem rgb_output = nullptr;
    size_t rgb_output_capacity = 0;
    cl_mem atlas_parent = nullptr;
    size_t atlas_parent_capacity = 0;
    cl_mem atlas_child = nullptr;
    size_t atlas_child_capacity = 0;
    cl_mem atlas_output = nullptr;
    size_t atlas_output_capacity = 0;
    bool atlas_axis_maps = false;
    cl_mem atlas_parent_x_axis = nullptr;
    size_t atlas_parent_x_axis_capacity = 0;
    cl_mem atlas_parent_y_axis = nullptr;
    size_t atlas_parent_y_axis_capacity = 0;
    cl_mem atlas_child_x_axis = nullptr;
    size_t atlas_child_x_axis_capacity = 0;
    cl_mem atlas_child_y_axis = nullptr;
    size_t atlas_child_y_axis_capacity = 0;
    cl_mem atlas_parent_image = nullptr;
    cl_mem atlas_child_image = nullptr;
    size_t atlas_parent_image_width = 0;
    size_t atlas_parent_image_height = 0;
    size_t atlas_child_image_width = 0;
    size_t atlas_child_image_height = 0;
    std::uint64_t atlas_parent_image_cache_token = 0;
    std::uint64_t atlas_child_image_cache_token = 0;
    bool atlas_images = false;
    // The static KFP atlas keeps the same immutable RGB tiles for many video
    // frames.  The ordinary RGB ABI still uploads on every call; this token
    // pair is used only by the explicitly cached compositor entry point.
    std::uint64_t rgb_parent_cache_token = 0;
    size_t rgb_parent_cache_bytes = 0;
    std::uint64_t rgb_child_cache_token = 0;
    size_t rgb_child_cache_bytes = 0;
    std::uint64_t atlas_parent_cache_token = 0;
    size_t atlas_parent_cache_bytes = 0;
    std::uint64_t atlas_child_cache_token = 0;
    size_t atlas_child_cache_bytes = 0;
    cl_mem deep_reference = nullptr;
    size_t deep_reference_capacity = 0;
    std::uint64_t deep_reference_generation = 0;
    cl_mem deep_reference_norms = nullptr;
    size_t deep_reference_norms_capacity = 0;
    std::uint64_t deep_reference_norms_generation = 0;
    cl_mem mixed_reference = nullptr;
    size_t mixed_reference_capacity = 0;
    std::uint64_t mixed_reference_generation = 0;
    cl_mem mixed_reference_norms = nullptr;
    size_t mixed_reference_norms_capacity = 0;
    std::uint64_t mixed_reference_norms_generation = 0;
    cl_mem deep_point_offsets = nullptr;
    size_t deep_point_offsets_capacity = 0;
    cl_mem deep_bla = nullptr;
    cl_mem deep_bla_offsets = nullptr;
    cl_mem deep_bla_counts = nullptr;
    size_t deep_bla_capacity = 0;
    size_t deep_bla_levels_capacity = 0;
    std::uint64_t deep_bla_generation = 0;
    int deep_bla_level_count = 0;
    cl_mem mixed_bla = nullptr;
    cl_mem mixed_bla_offsets = nullptr;
    cl_mem mixed_bla_counts = nullptr;
    size_t mixed_bla_capacity = 0;
    size_t mixed_bla_levels_capacity = 0;
    std::uint64_t mixed_bla_generation = 0;
    int mixed_bla_level_count = 0;
    std::mutex mutex;
    std::string error;

    ~OpenClRuntime() {
        if (deep_bla_counts) clReleaseMemObject(deep_bla_counts);
        if (deep_bla_offsets) clReleaseMemObject(deep_bla_offsets);
        if (deep_bla) clReleaseMemObject(deep_bla);
        if (mixed_bla_counts) clReleaseMemObject(mixed_bla_counts);
        if (mixed_bla_offsets) clReleaseMemObject(mixed_bla_offsets);
        if (mixed_bla) clReleaseMemObject(mixed_bla);
        if (mixed_reference_norms) clReleaseMemObject(mixed_reference_norms);
        if (mixed_reference) clReleaseMemObject(mixed_reference);
        if (deep_reference_norms) clReleaseMemObject(deep_reference_norms);
        if (deep_reference) clReleaseMemObject(deep_reference);
        if (colour_output) clReleaseMemObject(colour_output);
        if (colour_palette) clReleaseMemObject(colour_palette);
        if (colour_field) clReleaseMemObject(colour_field);
        if (rgb_output) clReleaseMemObject(rgb_output);
        if (rgb_child) clReleaseMemObject(rgb_child);
        if (rgb_parent) clReleaseMemObject(rgb_parent);
        if (atlas_output) clReleaseMemObject(atlas_output);
        if (atlas_child_y_axis) clReleaseMemObject(atlas_child_y_axis);
        if (atlas_child_x_axis) clReleaseMemObject(atlas_child_x_axis);
        if (atlas_parent_y_axis) clReleaseMemObject(atlas_parent_y_axis);
        if (atlas_parent_x_axis) clReleaseMemObject(atlas_parent_x_axis);
        if (atlas_child) clReleaseMemObject(atlas_child);
        if (atlas_parent) clReleaseMemObject(atlas_parent);
        if (atlas_child_image) clReleaseMemObject(atlas_child_image);
        if (atlas_parent_image) clReleaseMemObject(atlas_parent_image);
        if (deep_output) clReleaseMemObject(deep_output);
        if (deep_point_offsets) clReleaseMemObject(deep_point_offsets);
        if (direct_output) clReleaseMemObject(direct_output);
        for (cl_kernel direct_kernel : direct_formula_kernels) {
            if (direct_kernel) clReleaseKernel(direct_kernel);
        }
        if (deep_kernel) clReleaseKernel(deep_kernel);
        if (deep_scalar_kernel) clReleaseKernel(deep_scalar_kernel);
        if (deep_scalar_bla_kernel) clReleaseKernel(deep_scalar_bla_kernel);
        if (deep_mixed_kernel) clReleaseKernel(deep_mixed_kernel);
        if (deep_mixed_mandelbrot_kernel) {
            clReleaseKernel(deep_mixed_mandelbrot_kernel);
        }
        if (colour_kernel) clReleaseKernel(colour_kernel);
        if (kfp_colour_kernel) clReleaseKernel(kfp_colour_kernel);
        if (rgb_crop_kernel) clReleaseKernel(rgb_crop_kernel);
        if (rgb_composite_kernel) clReleaseKernel(rgb_composite_kernel);
        if (atlas_tiled_colour_kernel_secondary) {
            clReleaseKernel(atlas_tiled_colour_kernel_secondary);
        }
        if (atlas_tiled_colour_kernel) {
            clReleaseKernel(atlas_tiled_colour_kernel);
        }
        if (atlas_colour_kernel_secondary) {
            clReleaseKernel(atlas_colour_kernel_secondary);
        }
        if (atlas_colour_kernel) clReleaseKernel(atlas_colour_kernel);
        if (atlas_image_colour_kernel_secondary) {
            clReleaseKernel(atlas_image_colour_kernel_secondary);
        }
        if (atlas_image_colour_kernel) {
            clReleaseKernel(atlas_image_colour_kernel);
        }
        if (kernel) clReleaseKernel(kernel);
        if (program) clReleaseProgram(program);
        if (atlas_queue_secondary) {
            clReleaseCommandQueue(atlas_queue_secondary);
        }
        if (queue) clReleaseCommandQueue(queue);
        if (context) clReleaseContext(context);
    }
};

bool ensure_opencl_atlas_image(
    OpenClRuntime& runtime,
    cl_mem& image,
    size_t& image_width,
    size_t& image_height,
    int width,
    int height,
    std::uint64_t& cache_token,
    cl_int& status
) {
    const size_t requested_width = static_cast<size_t>(width);
    const size_t requested_height = static_cast<size_t>(height);
    if (image
        && image_width == requested_width
        && image_height == requested_height) {
        return true;
    }
    cl_image_format format{};
    format.image_channel_order = CL_RGBA;
    format.image_channel_data_type = CL_FLOAT;
    cl_image_desc description{};
    description.image_type = CL_MEM_OBJECT_IMAGE2D;
    description.image_width = requested_width;
    description.image_height = requested_height;
    cl_mem replacement = clCreateImage(
        runtime.context,
        CL_MEM_READ_ONLY,
        &format,
        &description,
        nullptr,
        &status);
    if (status != CL_SUCCESS || !replacement) return false;
    if (image) clReleaseMemObject(image);
    image = replacement;
    image_width = requested_width;
    image_height = requested_height;
    cache_token = 0;
    return true;
}

std::once_flag opencl_once;
std::unique_ptr<OpenClRuntime> opencl_runtime;

std::string opencl_error_text(cl_int status) {
    return "OpenCL error " + std::to_string(static_cast<int>(status));
}

bool opencl_device_supports_fp64(cl_device_id device) {
    size_t extension_size = 0;
    const cl_int status = clGetDeviceInfo(
        device, CL_DEVICE_EXTENSIONS, 0, nullptr, &extension_size);
    if (status != CL_SUCCESS || extension_size == 0
        || extension_size > MAX_OPENCL_INFO_BYTES) {
        return false;
    }
    std::string extensions(extension_size, '\0');
    if (clGetDeviceInfo(
            device, CL_DEVICE_EXTENSIONS, extension_size,
            extensions.data(), nullptr) != CL_SUCCESS) {
        return false;
    }
    return extensions.find("cl_khr_fp64") != std::string::npos
        || extensions.find("cl_amd_fp64") != std::string::npos;
}

struct OpenClDeviceCandidate {
    cl_device_id device = nullptr;
    cl_platform_id platform = nullptr;
    cl_ulong global_memory = 0;
    std::uint64_t throughput = 0;
    cl_uint compute_units = 0;
    cl_uint clock_mhz = 0;
    bool is_gpu = false;
};

OpenClDeviceCandidate inspect_opencl_device(
    cl_platform_id platform,
    cl_device_id device,
    bool is_gpu
) {
    OpenClDeviceCandidate candidate;
    candidate.device = device;
    candidate.platform = platform;
    candidate.is_gpu = is_gpu;
    // These are performance hints only. A driver is allowed to omit or
    // reject individual queries, so retain the device with zero-valued
    // fields rather than making capability probing fail.
    clGetDeviceInfo(
        device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(candidate.global_memory),
        &candidate.global_memory, nullptr);
    clGetDeviceInfo(
        device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(candidate.compute_units),
        &candidate.compute_units, nullptr);
    clGetDeviceInfo(
        device, CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(candidate.clock_mhz),
        &candidate.clock_mhz, nullptr);
    const std::uint64_t clock = candidate.clock_mhz > 0
        ? static_cast<std::uint64_t>(candidate.clock_mhz) : 1U;
    candidate.throughput = static_cast<std::uint64_t>(candidate.compute_units)
        * clock;
    return candidate;
}

bool better_opencl_device(
    const OpenClDeviceCandidate& candidate,
    const OpenClDeviceCandidate& current
) {
    // Compute units × advertised clock is a conservative portable ordering
    // for devices exposed through different ICDs. Global memory breaks ties
    // in favour of a discrete card over a small integrated adapter, while
    // the final fields keep enumeration deterministic when a driver reports
    // no performance metadata.
    if (candidate.throughput != current.throughput) {
        return candidate.throughput > current.throughput;
    }
    if (candidate.global_memory != current.global_memory) {
        return candidate.global_memory > current.global_memory;
    }
    if (candidate.compute_units != current.compute_units) {
        return candidate.compute_units > current.compute_units;
    }
    return candidate.clock_mhz > current.clock_mhz;
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

        OpenClDeviceCandidate selected_candidate;
        for (cl_platform_id platform : platforms) {
            cl_uint device_count = 0;
            if (clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, nullptr, &device_count)
                    == CL_SUCCESS && device_count > 0) {
                if (device_count > MAX_OPENCL_DEVICES) continue;
                std::vector<cl_device_id> devices(device_count);
                if (clGetDeviceIDs(
                        platform, CL_DEVICE_TYPE_GPU, device_count,
                        devices.data(), nullptr) == CL_SUCCESS) {
                    for (cl_device_id device : devices) {
                        if (opencl_device_supports_fp64(device)) {
                            const OpenClDeviceCandidate candidate =
                                inspect_opencl_device(platform, device, true);
                            if (!selected_candidate.device
                                || better_opencl_device(candidate, selected_candidate)) {
                                selected_candidate = candidate;
                            }
                        }
                    }
                }
            }
        }
        if (!selected_candidate.device) {
            for (cl_platform_id platform : platforms) {
                cl_uint device_count = 0;
                if (clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 0, nullptr, &device_count)
                        == CL_SUCCESS && device_count > 0) {
                    if (device_count > MAX_OPENCL_DEVICES) continue;
                    std::vector<cl_device_id> devices(device_count);
                    if (clGetDeviceIDs(
                            platform, CL_DEVICE_TYPE_CPU, device_count,
                            devices.data(), nullptr) == CL_SUCCESS) {
                        for (cl_device_id device : devices) {
                            if (opencl_device_supports_fp64(device)) {
                                const OpenClDeviceCandidate candidate =
                                    inspect_opencl_device(platform, device, false);
                                if (!selected_candidate.device
                                    || better_opencl_device(candidate, selected_candidate)) {
                                    selected_candidate = candidate;
                                }
                            }
                        }
                    }
                }
            }
        }
        if (!selected_candidate.device || !selected_candidate.platform) {
            runtime->error = "no double-precision OpenCL GPU or CPU device is available";
            opencl_runtime = std::move(runtime);
            return;
        }
        cl_device_id selected = selected_candidate.device;
        cl_platform_id selected_platform = selected_candidate.platform;
        runtime->device_is_gpu = selected_candidate.is_gpu;

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
        // A separate in-order queue is enough to let the driver overlap
        // independent atlas kernel/readback ranges.  Failure is non-fatal;
        // the compositor keeps its proven single-queue implementation.
        cl_int secondary_queue_status = CL_SUCCESS;
        runtime->atlas_queue_secondary = clCreateCommandQueue(
            runtime->context, selected, 0, &secondary_queue_status);
        if (secondary_queue_status != CL_SUCCESS
            || !runtime->atlas_queue_secondary) {
            runtime->atlas_queue_secondary = nullptr;
        }
        const std::string combined_source = std::string(OPENCL_DIRECT_KERNEL)
            + "\n" + OPENCL_SPECIALIZED_DIRECT_KERNEL
            + "\n" + OPENCL_DEEP_PERTURBATION_KERNEL
            + "\n" + OPENCL_DEEP_SCALAR_KERNEL
            + "\n" + OPENCL_DEEP_SCALAR_BLA_KERNEL
            + "\n" + OPENCL_DEEP_MIXED_KERNEL
            + "\n" + OPENCL_AURORA_COLOUR_KERNEL
            + "\n" + OPENCL_KFP_COLOUR_KERNEL
            + "\n" + OPENCL_RGB_ATLAS_KERNEL
            + "\n" + OPENCL_ATLAS_AURORA_KERNEL
            + "\n" + OPENCL_ATLAS_TILED_KERNEL;
        const size_t source_length = combined_source.size();
        const char* kernel_source = combined_source.c_str();
        runtime->program = clCreateProgramWithSource(
            runtime->context, 1, &kernel_source, &source_length, &status);
        if (status != CL_SUCCESS || !runtime->program) {
            runtime->error = opencl_error_text(status);
            opencl_runtime = std::move(runtime);
            return;
        }
        // NVIDIA's double-precision OpenCL compiler can leave a sizeable
        // amount of throughput on the table when every multiply/add in the
        // scaled-complex recurrence is kept as a separately rounded
        // operation. Keep the strict program as the default, but allow a
        // complete export/live-view run to opt into the driver's fused,
        // relaxed scheduling for a measured comparison. The option is read
        // before the one-time runtime build, so all kernels in one process
        // use the same arithmetic contract.
        const bool fast_math = std::getenv("FRACTAL_OPENCL_FAST_MATH") != nullptr;
        const bool atlas_float = std::getenv("FRACTAL_OPENCL_ATLAS_FLOAT") != nullptr;
        const char* atlas_maps_setting =
            std::getenv("FRACTAL_OPENCL_ATLAS_MAPS");
        const bool atlas_axis_maps = atlas_maps_setting != nullptr
            && atlas_maps_setting[0] != '\0'
            && std::strcmp(atlas_maps_setting, "0") != 0
            && std::strcmp(atlas_maps_setting, "false") != 0
            && std::strcmp(atlas_maps_setting, "off") != 0
            && std::strcmp(atlas_maps_setting, "no") != 0;
        const char* atlas_images_setting =
            std::getenv("FRACTAL_OPENCL_ATLAS_IMAGES");
        const bool atlas_images = atlas_images_setting != nullptr
            && atlas_images_setting[0] != '\0'
            && std::strcmp(atlas_images_setting, "0") != 0
            && std::strcmp(atlas_images_setting, "false") != 0
            && std::strcmp(atlas_images_setting, "off") != 0
            && std::strcmp(atlas_images_setting, "no") != 0;
        runtime->atlas_images = atlas_images;
        // The image sampler owns the coordinate interpolation, so it is
        // mutually exclusive with the explicit axis-map path.
        runtime->atlas_axis_maps = atlas_axis_maps && !atlas_images;
        std::string build_options;
        if (fast_math) {
            build_options = "-cl-fast-relaxed-math";
            build_options += " -D FRACTAL_OPENCL_FAST_MATH=1";
        }
        if (atlas_float) {
            if (!build_options.empty()) {
                build_options += " ";
            }
            build_options += "-D FRACTAL_OPENCL_ATLAS_FLOAT=1";
        }
        if (runtime->atlas_axis_maps) {
            if (!build_options.empty()) {
                build_options += " ";
            }
            build_options += "-D FRACTAL_OPENCL_ATLAS_MAPS=1";
        }
        if (atlas_images) {
            if (!build_options.empty()) {
                build_options += " ";
            }
            build_options += "-D FRACTAL_OPENCL_ATLAS_IMAGES=1";
        }
        status = clBuildProgram(
            runtime->program, 1, &selected, build_options.c_str(), nullptr, nullptr);
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
        } else {
            constexpr std::array<const char*, 4> specialized_names{
                "mandelbrot_direct_mandelbrot",
                "mandelbrot_direct_julia",
                "mandelbrot_direct_burning_ship",
                "mandelbrot_direct_tricorn",
            };
            for (size_t index = 0; index < specialized_names.size(); ++index) {
                cl_int specialized_status = CL_SUCCESS;
                cl_kernel specialized = clCreateKernel(
                    runtime->program, specialized_names[index], &specialized_status);
                if (specialized_status == CL_SUCCESS && specialized) {
                    runtime->direct_formula_kernels[index] = specialized;
                }
                // A single driver may reject one optional entry point while
                // accepting the others. Keep the compatible kernel usable
                // instead of turning that partial capability into a global
                // OpenCL failure.
            }
            runtime->deep_kernel = clCreateKernel(
                runtime->program, "mandelbrot_deep_perturbation", &status);
            if (status != CL_SUCCESS || !runtime->deep_kernel) {
                runtime->error = opencl_error_text(status);
            } else {
                cl_int scalar_status = CL_SUCCESS;
                runtime->deep_scalar_kernel = clCreateKernel(
                    runtime->program, "mandelbrot_deep_perturbation_scalar",
                    &scalar_status);
                // This is an optional hot-path specialization. The generic
                // kernel remains the correctness fallback if an older or
                // unusually strict ICD rejects the extra entry point.
                if (scalar_status != CL_SUCCESS || !runtime->deep_scalar_kernel) {
                    runtime->deep_scalar_kernel = nullptr;
                }
                cl_int scalar_bla_status = CL_SUCCESS;
                runtime->deep_scalar_bla_kernel = clCreateKernel(
                    runtime->program, "mandelbrot_deep_perturbation_scalar_bla",
                    &scalar_bla_status);
                // The scalar+BLA entry point is another optional
                // specialization. Keep the generic BLA kernel available when
                // an older compiler cannot validate this longer argument list.
                if (scalar_bla_status != CL_SUCCESS
                    || !runtime->deep_scalar_bla_kernel) {
                    runtime->deep_scalar_bla_kernel = nullptr;
                }
                cl_int mixed_status = CL_SUCCESS;
                runtime->deep_mixed_kernel = clCreateKernel(
                    runtime->program, "mandelbrot_deep_perturbation_mixed",
                    &mixed_status);
                // Mixed precision is an optional throughput path.  A driver
                // that cannot compile it must retain the strict fp64 path.
                if (mixed_status != CL_SUCCESS || !runtime->deep_mixed_kernel) {
                    runtime->deep_mixed_kernel = nullptr;
                }
                cl_int mixed_mandelbrot_status = CL_SUCCESS;
                runtime->deep_mixed_mandelbrot_kernel = clCreateKernel(
                    runtime->program,
                    "mandelbrot_deep_perturbation_mixed_mandelbrot",
                    &mixed_mandelbrot_status);
                // The Mandelbrot-only mixed entry point is an optional
                // throughput specialization. Keep the formula-generic mixed
                // kernel as the fallback for older OpenCL compilers.
                if (mixed_mandelbrot_status != CL_SUCCESS
                    || !runtime->deep_mixed_mandelbrot_kernel) {
                    runtime->deep_mixed_mandelbrot_kernel = nullptr;
                }
            }
        }
        const auto kernel_workgroup_limit = [selected](cl_kernel kernel) {
            size_t limit = 0;
            if (!kernel || clGetKernelWorkGroupInfo(
                    kernel, selected, CL_KERNEL_WORK_GROUP_SIZE,
                    sizeof(limit), &limit, nullptr) != CL_SUCCESS) {
                return static_cast<size_t>(0);
            }
            return limit;
        };
        const auto choose_workgroup = [selected](
            cl_kernel kernel,
            size_t limit,
            size_t fallback
        ) {
            if (!kernel || limit == 0) return static_cast<size_t>(0);
            size_t preferred = 0;
            if (clGetKernelWorkGroupInfo(
                    kernel, selected,
                    CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE,
                    sizeof(preferred), &preferred, nullptr) != CL_SUCCESS
                || preferred == 0 || preferred > 256) {
                preferred = fallback;
            }
            if (const char* override_text = std::getenv(
                    "FRACTAL_OPENCL_WORKGROUP")) {
                char* end = nullptr;
                const unsigned long long requested = std::strtoull(
                    override_text, &end, 10);
                if (end != override_text && *end == '\0'
                    && requested > 0
                    && requested <= static_cast<unsigned long long>(limit)
                    && requested % preferred == 0) {
                    return static_cast<size_t>(requested);
                }
            }
            return std::min(preferred, limit);
        };
        if (runtime->error.empty() && runtime->kernel && runtime->deep_kernel) {
            size_t common_limit = kernel_workgroup_limit(runtime->kernel);
            const size_t deep_limit = kernel_workgroup_limit(runtime->deep_kernel);
            if (common_limit == 0 || deep_limit == 0) {
                common_limit = 0;
            } else {
                common_limit = std::min(common_limit, deep_limit);
                for (cl_kernel direct_kernel : runtime->direct_formula_kernels) {
                    if (!direct_kernel) continue;
                    const size_t limit = kernel_workgroup_limit(direct_kernel);
                    if (limit == 0) {
                        common_limit = 0;
                        break;
                    }
                    common_limit = std::min(common_limit, limit);
                }
            }
            runtime->workgroup_size = choose_workgroup(
                runtime->kernel, common_limit, 256);
            if (runtime->deep_scalar_kernel) {
                const size_t scalar_limit = kernel_workgroup_limit(
                    runtime->deep_scalar_kernel);
                runtime->deep_scalar_workgroup_size = choose_workgroup(
                    runtime->deep_scalar_kernel,
                    scalar_limit,
                    runtime->workgroup_size > 0 ? runtime->workgroup_size : 256);
            }
            if (runtime->deep_scalar_bla_kernel) {
                const size_t scalar_bla_limit = kernel_workgroup_limit(
                    runtime->deep_scalar_bla_kernel);
                runtime->deep_scalar_bla_workgroup_size = choose_workgroup(
                    runtime->deep_scalar_bla_kernel,
                    scalar_bla_limit,
                    runtime->workgroup_size > 0 ? runtime->workgroup_size : 256);
            }
            if (runtime->deep_mixed_kernel) {
                const size_t mixed_limit = kernel_workgroup_limit(
                    runtime->deep_mixed_kernel);
                runtime->deep_mixed_workgroup_size = choose_workgroup(
                    runtime->deep_mixed_kernel,
                    mixed_limit,
                    runtime->workgroup_size > 0 ? runtime->workgroup_size : 256);
                // The mixed recurrence has no plane writes and benefits from
                // twice the driver's conservative preferred multiple on the
                // tested discrete NVIDIA path. Keep the public override as
                // the authority, and only widen the automatic GPU choice
                // when the kernel advertises a compatible limit.
                if (!std::getenv("FRACTAL_OPENCL_WORKGROUP")
                    && runtime->device_is_gpu
                    && runtime->deep_mixed_workgroup_size > 0
                    && runtime->deep_mixed_workgroup_size
                        <= std::numeric_limits<size_t>::max() / 2) {
                    const size_t wider = runtime->deep_mixed_workgroup_size * 2;
                    if (wider <= mixed_limit) {
                        runtime->deep_mixed_workgroup_size = wider;
                    }
                }
            }
            if (runtime->deep_mixed_mandelbrot_kernel) {
                const size_t mixed_mandelbrot_limit = kernel_workgroup_limit(
                    runtime->deep_mixed_mandelbrot_kernel);
                runtime->deep_mixed_mandelbrot_workgroup_size = choose_workgroup(
                    runtime->deep_mixed_mandelbrot_kernel,
                    mixed_mandelbrot_limit,
                    runtime->deep_mixed_workgroup_size > 0
                        ? runtime->deep_mixed_workgroup_size
                        : (runtime->workgroup_size > 0
                            ? runtime->workgroup_size : 256));
                if (!std::getenv("FRACTAL_OPENCL_WORKGROUP")
                    && runtime->device_is_gpu
                    && runtime->deep_mixed_mandelbrot_workgroup_size > 0
                    && runtime->deep_mixed_mandelbrot_workgroup_size
                        <= std::numeric_limits<size_t>::max() / 2) {
                    const size_t wider =
                        runtime->deep_mixed_mandelbrot_workgroup_size * 2;
                    if (wider <= mixed_mandelbrot_limit) {
                        runtime->deep_mixed_mandelbrot_workgroup_size = wider;
                    }
                }
            }

            // Keep the device-side launch choice observable when tuning a
            // new driver.  This is opt-in so normal GUI/live-view runs stay
            // quiet, but it avoids guessing from an unavailable `clinfo`
            // installation.
            if (std::getenv("FRACTAL_OPENCL_DIAGNOSTICS") != nullptr) {
                const auto report_kernel = [selected, kernel_workgroup_limit](
                    const char* name, cl_kernel kernel, size_t chosen) {
                    size_t preferred = 0;
                    size_t private_bytes = 0;
                    size_t local_bytes = 0;
                    if (kernel) {
                        clGetKernelWorkGroupInfo(
                            kernel, selected,
                            CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE,
                            sizeof(preferred), &preferred, nullptr);
                        clGetKernelWorkGroupInfo(
                            kernel, selected, CL_KERNEL_PRIVATE_MEM_SIZE,
                            sizeof(private_bytes), &private_bytes, nullptr);
                        clGetKernelWorkGroupInfo(
                            kernel, selected, CL_KERNEL_LOCAL_MEM_SIZE,
                            sizeof(local_bytes), &local_bytes, nullptr);
                    }
                    std::fprintf(
                        stderr,
                        "OpenCL %s: limit=%zu preferred=%zu chosen=%zu private=%zu local=%zu\n",
                        name, kernel_workgroup_limit(kernel), preferred, chosen,
                        private_bytes, local_bytes);
                };
                report_kernel("generic-deep", runtime->deep_kernel,
                              runtime->workgroup_size);
                report_kernel("scalar-deep", runtime->deep_scalar_kernel,
                              runtime->deep_scalar_workgroup_size);
                report_kernel("scalar-bla-deep", runtime->deep_scalar_bla_kernel,
                              runtime->deep_scalar_bla_workgroup_size);
                report_kernel("mixed-deep", runtime->deep_mixed_kernel,
                              runtime->deep_mixed_workgroup_size);
                report_kernel(
                    "mixed-mandelbrot-deep",
                    runtime->deep_mixed_mandelbrot_kernel,
                    runtime->deep_mixed_mandelbrot_workgroup_size);
                size_t compute_units = 0;
                clGetDeviceInfo(
                    selected, CL_DEVICE_MAX_COMPUTE_UNITS,
                    sizeof(compute_units), &compute_units, nullptr);
                size_t device_limit = 0;
                clGetDeviceInfo(
                    selected, CL_DEVICE_MAX_WORK_GROUP_SIZE,
                    sizeof(device_limit), &device_limit, nullptr);
                std::fprintf(
                    stderr, "OpenCL device: compute_units=%zu max_workgroup=%zu gpu=%s\n",
                    compute_units, device_limit, runtime->device_is_gpu ? "yes" : "no");
            }
        }
        if (runtime->error.empty() && runtime->deep_kernel) {
            cl_int colour_status = CL_SUCCESS;
            runtime->colour_kernel = clCreateKernel(
                runtime->program, "aurora_colourise", &colour_status);
            // Colourisation is an optional second-stage acceleration. A
            // device compiler may reject it (for example because of a
            // restricted byte-addressing implementation) while accepting
            // both double-precision field kernels. Keep the field backend
            // usable in that case and let the caller retain its CPU mapper.
            if (colour_status != CL_SUCCESS || !runtime->colour_kernel) {
                runtime->colour_kernel = nullptr;
            } else {
                const size_t colour_limit = kernel_workgroup_limit(runtime->colour_kernel);
                runtime->colour_workgroup_size = choose_workgroup(
                    runtime->colour_kernel, colour_limit,
                    runtime->workgroup_size > 0 ? runtime->workgroup_size : 256);
            }
            cl_int kfp_colour_status = CL_SUCCESS;
            runtime->kfp_colour_kernel = clCreateKernel(
                runtime->program, "kfp_colourise", &kfp_colour_status);
            // KFP colourisation is an optional acceleration layer. Keep the
            // field backend usable when an older or restricted OpenCL
            // compiler accepts the orbit kernels but rejects this larger
            // source-faithful stencil kernel.
            if (kfp_colour_status != CL_SUCCESS || !runtime->kfp_colour_kernel) {
                runtime->kfp_colour_kernel = nullptr;
            } else {
                const size_t kfp_colour_limit = kernel_workgroup_limit(
                    runtime->kfp_colour_kernel);
                runtime->kfp_colour_workgroup_size = choose_workgroup(
                    runtime->kfp_colour_kernel,
                    kfp_colour_limit,
                    runtime->colour_workgroup_size > 0
                        ? runtime->colour_workgroup_size
                        : (runtime->workgroup_size > 0
                            ? runtime->workgroup_size : 256));
            }
            cl_int rgb_status = CL_SUCCESS;
            runtime->rgb_crop_kernel = clCreateKernel(
                runtime->program, "rgb_crop", &rgb_status);
            if (rgb_status != CL_SUCCESS || !runtime->rgb_crop_kernel) {
                runtime->rgb_crop_kernel = nullptr;
            }
            rgb_status = CL_SUCCESS;
            runtime->rgb_composite_kernel = clCreateKernel(
                runtime->program, "rgb_atlas_composite", &rgb_status);
            if (rgb_status != CL_SUCCESS || !runtime->rgb_composite_kernel) {
                runtime->rgb_composite_kernel = nullptr;
            }
            if (runtime->rgb_crop_kernel && runtime->rgb_composite_kernel) {
                const size_t crop_limit = kernel_workgroup_limit(
                    runtime->rgb_crop_kernel);
                const size_t composite_limit = kernel_workgroup_limit(
                    runtime->rgb_composite_kernel);
                const size_t common_rgb_limit = crop_limit > 0
                    && composite_limit > 0
                    ? std::min(crop_limit, composite_limit)
                    : 0;
                runtime->rgb_workgroup_size = choose_workgroup(
                    runtime->rgb_crop_kernel,
                    common_rgb_limit,
                    runtime->kfp_colour_workgroup_size > 0
                        ? runtime->kfp_colour_workgroup_size
                    : (runtime->workgroup_size > 0
                        ? runtime->workgroup_size : 256));
            }
            cl_int atlas_status = CL_SUCCESS;
            runtime->atlas_colour_kernel = clCreateKernel(
                runtime->program, "aurora_atlas_colourise", &atlas_status);
            // The fused atlas path is an optional optimization.  Keep the
            // ordinary OpenCL field and colour entry points usable on older
            // drivers whose compiler accepts the smaller kernels only.
            if (atlas_status != CL_SUCCESS || !runtime->atlas_colour_kernel) {
                runtime->atlas_colour_kernel = nullptr;
            } else {
                cl_int secondary_kernel_status = CL_SUCCESS;
                if (runtime->atlas_queue_secondary) {
                    runtime->atlas_colour_kernel_secondary = clCreateKernel(
                        runtime->program,
                        "aurora_atlas_colourise",
                        &secondary_kernel_status);
                    if (secondary_kernel_status != CL_SUCCESS
                        || !runtime->atlas_colour_kernel_secondary) {
                        runtime->atlas_colour_kernel_secondary = nullptr;
                    }
                }
                const size_t atlas_limit = kernel_workgroup_limit(
                    runtime->atlas_colour_kernel);
                runtime->atlas_colour_workgroup_limit = atlas_limit;
                runtime->atlas_colour_workgroup_size = choose_workgroup(
                    runtime->atlas_colour_kernel,
                    atlas_limit,
                    runtime->colour_workgroup_size > 0
                        ? runtime->colour_workgroup_size
                        : (runtime->workgroup_size > 0
                            ? runtime->workgroup_size : 256));
            }
            cl_int tiled_atlas_status = CL_SUCCESS;
            runtime->atlas_tiled_colour_kernel = clCreateKernel(
                runtime->program,
                "aurora_atlas_colourise_tiled",
                &tiled_atlas_status);
            if (tiled_atlas_status != CL_SUCCESS
                || !runtime->atlas_tiled_colour_kernel) {
                runtime->atlas_tiled_colour_kernel = nullptr;
            } else {
                cl_int tiled_secondary_status = CL_SUCCESS;
                if (runtime->atlas_queue_secondary) {
                    runtime->atlas_tiled_colour_kernel_secondary = clCreateKernel(
                        runtime->program,
                        "aurora_atlas_colourise_tiled",
                        &tiled_secondary_status);
                    if (tiled_secondary_status != CL_SUCCESS
                        || !runtime->atlas_tiled_colour_kernel_secondary) {
                        runtime->atlas_tiled_colour_kernel_secondary = nullptr;
                    }
                }
                runtime->atlas_tiled_colour_workgroup_limit =
                    kernel_workgroup_limit(runtime->atlas_tiled_colour_kernel);
            }
            if (runtime->atlas_images) {
                cl_int image_status = CL_SUCCESS;
                runtime->atlas_image_colour_kernel = clCreateKernel(
                    runtime->program,
                    "aurora_atlas_colourise_image",
                    &image_status);
                if (image_status != CL_SUCCESS
                    || !runtime->atlas_image_colour_kernel) {
                    runtime->atlas_image_colour_kernel = nullptr;
                    runtime->atlas_images = false;
                } else {
                    cl_int image_secondary_status = CL_SUCCESS;
                    if (runtime->atlas_queue_secondary) {
                        runtime->atlas_image_colour_kernel_secondary =
                            clCreateKernel(
                                runtime->program,
                                "aurora_atlas_colourise_image",
                                &image_secondary_status);
                        if (image_secondary_status != CL_SUCCESS
                            || !runtime->atlas_image_colour_kernel_secondary) {
                            runtime->atlas_image_colour_kernel_secondary = nullptr;
                        }
                    }
                    const size_t image_limit = kernel_workgroup_limit(
                        runtime->atlas_image_colour_kernel);
                    runtime->atlas_colour_workgroup_limit = image_limit;
                    runtime->atlas_colour_workgroup_size = choose_workgroup(
                        runtime->atlas_image_colour_kernel,
                        image_limit,
                        runtime->colour_workgroup_size > 0
                            ? runtime->colour_workgroup_size
                            : (runtime->workgroup_size > 0
                                ? runtime->workgroup_size : 256));
                }
            }
        }
        opencl_runtime = std::move(runtime);
    });
}

bool opencl_available() {
    initialise_opencl();
    return opencl_runtime != nullptr
        && opencl_runtime->kernel != nullptr
        && opencl_runtime->deep_kernel != nullptr
        && opencl_runtime->error.empty();
}

bool opencl_colour_available() {
    return opencl_available()
        && opencl_runtime != nullptr
        && opencl_runtime->colour_kernel != nullptr;
}

bool opencl_kfp_colour_available() {
    return opencl_available()
        && opencl_runtime != nullptr
        && opencl_runtime->kfp_colour_kernel != nullptr;
}

bool opencl_atlas_colour_available() {
    return opencl_available()
        && opencl_runtime != nullptr
        && (opencl_runtime->atlas_colour_kernel != nullptr
            || opencl_runtime->atlas_tiled_colour_kernel != nullptr
            || opencl_runtime->atlas_image_colour_kernel != nullptr);
}

void render_direct_opencl(
    float* output,
    int width,
    int height,
    double zoom,
    double x_center,
    double y_center,
    int max_iter,
    double output_bias,
    int formula,
    double julia_real,
    double julia_imag,
    int escape_radius_mode,
    int coordinate_mode
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
    const double height_span = viewport_height_factor(coordinate_mode) / zoom;
    const double width_span = height_span * static_cast<double>(width)
        / static_cast<double>(height);
    const double escape_squared = escape_radius_squared_double(escape_radius_mode);
    const size_t count = static_cast<size_t>(width) * static_cast<size_t>(height);
    cl_int status = CL_SUCCESS;
    const size_t output_bytes = count * sizeof(float);
    if (!runtime.direct_output || runtime.direct_output_capacity < output_bytes) {
        cl_mem replacement = clCreateBuffer(
            runtime.context, CL_MEM_WRITE_ONLY, output_bytes, nullptr, &status);
        if (status != CL_SUCCESS || !replacement) {
            throw std::runtime_error(opencl_error_text(status));
        }
        if (runtime.direct_output) clReleaseMemObject(runtime.direct_output);
        runtime.direct_output = replacement;
        runtime.direct_output_capacity = output_bytes;
    }
    cl_mem device_output = runtime.direct_output;
    if (!device_output) {
        throw std::runtime_error(opencl_error_text(status));
    }
    cl_kernel selected_kernel = runtime.kernel;
    if (valid_formula(formula)
        && runtime.direct_formula_kernels[static_cast<size_t>(formula)]) {
        selected_kernel = runtime.direct_formula_kernels[static_cast<size_t>(formula)];
    }
    status = clSetKernelArg(selected_kernel, 0, sizeof(device_output), &device_output);
    status |= clSetKernelArg(selected_kernel, 1, sizeof(width), &width);
    status |= clSetKernelArg(selected_kernel, 2, sizeof(height), &height);
    status |= clSetKernelArg(selected_kernel, 3, sizeof(x_center), &x_center);
    status |= clSetKernelArg(selected_kernel, 4, sizeof(y_center), &y_center);
    status |= clSetKernelArg(selected_kernel, 5, sizeof(width_span), &width_span);
    status |= clSetKernelArg(selected_kernel, 6, sizeof(height_span), &height_span);
    status |= clSetKernelArg(selected_kernel, 7, sizeof(max_iter), &max_iter);
    status |= clSetKernelArg(selected_kernel, 8, sizeof(output_bias), &output_bias);
    status |= clSetKernelArg(selected_kernel, 9, sizeof(formula), &formula);
    status |= clSetKernelArg(selected_kernel, 10, sizeof(julia_real), &julia_real);
    status |= clSetKernelArg(selected_kernel, 11, sizeof(julia_imag), &julia_imag);
    status |= clSetKernelArg(selected_kernel, 12, sizeof(escape_squared), &escape_squared);
    status |= clSetKernelArg(selected_kernel, 13, sizeof(coordinate_mode), &coordinate_mode);
    if (status != CL_SUCCESS) {
        throw std::runtime_error(opencl_error_text(status));
    }
    const size_t workgroup = runtime.workgroup_size;
    const size_t global_size = workgroup > 0
        ? ((count + workgroup - 1U) / workgroup) * workgroup
        : count;
    const size_t* local_work_size = workgroup > 0 ? &workgroup : nullptr;
    status = clEnqueueNDRangeKernel(
        runtime.queue, selected_kernel, 1, nullptr,
        &global_size, local_work_size, 0, nullptr, nullptr);
    if (status == CL_SUCCESS) {
        status = clEnqueueReadBuffer(
            runtime.queue, device_output, CL_TRUE, 0,
            count * sizeof(float), output, 0, nullptr, nullptr);
    }
    // CL_TRUE readback waits for every preceding command in this in-order
    // queue, so a separate clFinish would only add another driver round-trip.
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
    options.escape_radius_mode = ESCAPE_RADIUS_MODE_CLASSIC;
    options.coordinate_mode = COORDINATE_MODE_PROJECT;
    options.output_bias = 0.0;
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
    if (!valid_escape_radius_mode(options.escape_radius_mode)) {
        throw std::runtime_error("unknown escape-radius mode");
    }
    if (!valid_coordinate_mode(options.coordinate_mode)) {
        throw std::runtime_error("unknown coordinate mode");
    }
    options.series_min_terms = std::clamp(options.series_min_terms, 1, 32);
    options.series_max_terms = std::clamp(options.series_max_terms, options.series_min_terms, 32);
    options.max_bla_length = std::clamp(options.max_bla_length, 1, MAX_SAFE_BLA_LENGTH);
    options.max_linear_bla_length = std::clamp(
        options.max_linear_bla_length, 1, MAX_SAFE_LINEAR_BLA_LENGTH);
    if (options.time_budget_ms < 0) {
        throw std::runtime_error("render time budget cannot be negative");
    }
    if (!std::isfinite(options.output_bias) || options.output_bias < 0.0) {
        throw std::runtime_error("render output bias must be finite and non-negative");
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
    double pitch,
    const std::uint8_t* accents = nullptr
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
        std::array<double, 3> colour = original;
        if (accents != nullptr) {
            // Ordinary palettes are the same three Aurora waves, projected
            // through three RGB accents.  Do that projection while the
            // palette is being built so the frame needs one native pass,
            // rather than an Aurora pass followed by a full-image Python or
            // native recolour pass.
            const double red_weight = std::clamp(
                static_cast<double>(colour_byte(original[0])) / 140.0,
                0.0,
                1.0);
            const double green_weight = std::clamp(
                static_cast<double>(colour_byte(original[1])) / 140.0,
                0.0,
                1.0);
            const double blue_weight = std::clamp(
                static_cast<double>(colour_byte(original[2])) / 140.0,
                0.0,
                1.0);
            colour = {{
                (red_weight * static_cast<double>(accents[0])
                    + green_weight * static_cast<double>(accents[3])
                    + blue_weight * static_cast<double>(accents[6])) / 1.8,
                (red_weight * static_cast<double>(accents[1])
                    + green_weight * static_cast<double>(accents[4])
                    + blue_weight * static_cast<double>(accents[7])) / 1.8,
                (red_weight * static_cast<double>(accents[2])
                    + green_weight * static_cast<double>(accents[5])
                    + blue_weight * static_cast<double>(accents[8])) / 1.8,
            }};
        }
        if (rotate) {
            colour = rotate_hue_rgb(colour, hue_cos, hue_sin);
        }
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

// Fill a bilinear map for a window that extends beyond the logical
// destination rectangle.  The KFP atlas uses a one/two-pixel halo around the
// visible child so its 3x3 difference/slope stencil can read real neighbours
// instead of reflecting the moving child edge. ``destination_origin`` and
// ``destination_span`` are expressed in pixels of the logical output child.
// ``source_zoom`` applies the same centred crop as fill_bilinear_axis while
// retaining the extra destination pixels needed by the KFP stencil halo.
void fill_bilinear_axis_window(
    BilinearAxis& axis,
    int source_size,
    int destination_size,
    double destination_origin,
    double destination_span,
    double source_zoom = 1.0
) {
    if (source_size <= 0 || destination_size <= 0
        || !std::isfinite(destination_origin)
        || !std::isfinite(destination_span) || destination_span <= 0.0
        || !std::isfinite(source_zoom) || source_zoom <= 0.0) {
        throw std::runtime_error("invalid bilinear window dimensions");
    }
    source_zoom = std::max(source_zoom, 1.0);
    const double crop_size = static_cast<double>(source_size) / source_zoom;
    const double source_left = (
        static_cast<double>(source_size) - crop_size) * 0.5;
    axis.index0.resize(static_cast<size_t>(destination_size));
    axis.index1.resize(static_cast<size_t>(destination_size));
    axis.weight.resize(static_cast<size_t>(destination_size));
    for (int destination = 0; destination < destination_size; ++destination) {
        const double logical_pixel = destination_origin
            + static_cast<double>(destination) + 0.5;
        double source = source_left
            + logical_pixel * crop_size / destination_span - 0.5;
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
    std::vector<float> kfp_parent_field;
    std::vector<float> kfp_child_field;
    // Reprojected Kalles metadata for the fused plane-aware atlas path.
    // These vectors are resized only for planes supplied by the caller and
    // retain capacity between video frames to avoid allocator churn.
    std::vector<std::int64_t> kfp_atlas_orbit_iteration;
    std::vector<std::int64_t> kfp_atlas_iteration;
    std::vector<double> kfp_atlas_bailout;
    std::vector<double> kfp_atlas_transition;
    std::vector<double> kfp_atlas_phase;
    std::vector<double> kfp_atlas_de_x;
    std::vector<double> kfp_atlas_de_y;
    std::vector<double> kfp_atlas_test1;
    std::vector<double> kfp_atlas_test2;
};

thread_local BilinearWorkspace bilinear_workspace;

// Quality KFP calls opt out of the compact approximation kernel while keeping
// the fast entry points ABI-compatible for the live/upscaled renderer.  The
// flag is only consulted before an OpenMP loop is selected; the worker loop
// itself is already routed by that decision, so OpenMP thread-local state does
// not need to be inherited.
thread_local bool kfp_force_precise = false;

struct KfpPreciseGuard {
    bool previous = false;

    KfpPreciseGuard() noexcept : previous(kfp_force_precise) {
        kfp_force_precise = true;
    }

    ~KfpPreciseGuard() {
        kfp_force_precise = previous;
    }
};

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

inline double sample_bilinear_mapped_double(
    const double* source,
    int source_width,
    const BilinearAxis& x_axis,
    const BilinearAxis& y_axis,
    int x,
    int y,
    bool preserve_negative = false
) {
    const int x_index = x_axis.index0[static_cast<size_t>(x)];
    const int x_next = x_axis.index1[static_cast<size_t>(x)];
    const double x_weight = static_cast<double>(
        x_axis.weight[static_cast<size_t>(x)]);
    const int y_index = y_axis.index0[static_cast<size_t>(y)];
    const int y_next = y_axis.index1[static_cast<size_t>(y)];
    const double y_weight = static_cast<double>(
        y_axis.weight[static_cast<size_t>(y)]);
    const double* top = source + static_cast<size_t>(y_index) * source_width;
    const double* bottom = source + static_cast<size_t>(y_next) * source_width;
    const auto ordinary = [preserve_negative](double value) {
        return preserve_negative && value < 0.0 ? 0.0 : value;
    };
    const double top_value = ordinary(top[x_index]) * (1.0 - x_weight)
        + ordinary(top[x_next]) * x_weight;
    const double bottom_value = ordinary(bottom[x_index]) * (1.0 - x_weight)
        + ordinary(bottom[x_next]) * x_weight;
    return top_value * (1.0 - y_weight) + bottom_value * y_weight;
}

inline double sample_nearest_mapped_double(
    const double* source,
    int source_width,
    const BilinearAxis& x_axis,
    const BilinearAxis& y_axis,
    int x,
    int y
) {
    const int x_index = x_axis.weight[static_cast<size_t>(x)] < 0.5F
        ? x_axis.index0[static_cast<size_t>(x)]
        : x_axis.index1[static_cast<size_t>(x)];
    const int y_index = y_axis.weight[static_cast<size_t>(y)] < 0.5F
        ? y_axis.index0[static_cast<size_t>(y)]
        : y_axis.index1[static_cast<size_t>(y)];
    return source[static_cast<size_t>(y_index) * source_width
        + static_cast<size_t>(x_index)];
}

inline std::int64_t sample_nearest_mapped_int64(
    const std::int64_t* source,
    int source_width,
    const BilinearAxis& x_axis,
    const BilinearAxis& y_axis,
    int x,
    int y
) {
    const int x_index = x_axis.weight[static_cast<size_t>(x)] < 0.5F
        ? x_axis.index0[static_cast<size_t>(x)]
        : x_axis.index1[static_cast<size_t>(x)];
    const int y_index = y_axis.weight[static_cast<size_t>(y)] < 0.5F
        ? y_axis.index0[static_cast<size_t>(y)]
        : y_axis.index1[static_cast<size_t>(y)];
    return source[static_cast<size_t>(y_index) * source_width
        + static_cast<size_t>(x_index)];
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
    bool& inside,
    double source_field_bias = 0.0
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
        if (std::isfinite(value)
            && static_cast<double>(value)
                >= static_cast<double>(source_max_iter) - source_field_bias) {
            interior_weight += weight;
        } else if (std::isfinite(value)) {
            exterior_weight += weight;
            exterior_value += (
                static_cast<double>(value) + source_field_bias) * weight;
        }
    }
    if (interior_weight >= 0.5) {
        inside = true;
        return static_cast<float>(source_max_iter);
    }
    if (exterior_weight <= 1.0e-12) {
        // Ignore invalid neighbours rather than treating them as interior.
        // Public callers reject non-finite fields, but this keeps the native
        // ABI safe for diagnostic/test inputs and prevents black rectangles.
        inside = false;
        return 0.0F;
    }
    inside = false;
    return static_cast<float>(exterior_value / exterior_weight);
}

inline void write_colour_pixel(
    float smooth,
    int source_max_iter,
    const AuroraPalette& palette,
    float palette_index_scale,
    std::uint8_t* destination,
    const int* interior_color = nullptr
) {
    if (!std::isfinite(smooth)
        || smooth >= static_cast<float>(source_max_iter)) {
        destination[0] = interior_color ? static_cast<std::uint8_t>(interior_color[0]) : 0;
        destination[1] = interior_color ? static_cast<std::uint8_t>(interior_color[1]) : 0;
        destination[2] = interior_color ? static_cast<std::uint8_t>(interior_color[2]) : 0;
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

inline std::uint8_t rounded_colour_byte(double value) {
    return static_cast<std::uint8_t>(std::clamp(
        static_cast<int>(std::nearbyint(std::clamp(value, 0.0, 255.0))),
        0,
        255));
}

inline float kfp_colour_dither_mask(
    int x,
    int y,
    int channel
) noexcept {
    // Kalles' final srgb8 conversion uses the compact ordered mask from
    // colour.h.  The Burtle hash in fraktal_sft.cpp is used for positional
    // jitter, not for colour quantisation. Keep the arithmetic in uint32_t so
    // an unusually large but valid frame still has defined wraparound.
    const std::uint32_t coordinate =
        (static_cast<std::uint32_t>(x)
            + static_cast<std::uint32_t>(channel) * 67U
            + static_cast<std::uint32_t>(y) * 236U) * 119U;
    return static_cast<float>(coordinate & 255U) / 256.0F;
}

inline void sample_rgb_bilinear_mapped(
    const std::uint8_t* source,
    int source_width,
    const BilinearAxis& x_axis,
    const BilinearAxis& y_axis,
    int x,
    int y,
    std::uint8_t* destination
) {
    const int x_index = x_axis.index0[static_cast<size_t>(x)];
    const int x_next = x_axis.index1[static_cast<size_t>(x)];
    const float x_weight = x_axis.weight[static_cast<size_t>(x)];
    const int y_index = y_axis.index0[static_cast<size_t>(y)];
    const int y_next = y_axis.index1[static_cast<size_t>(y)];
    const float y_weight = y_axis.weight[static_cast<size_t>(y)];
    const std::uint8_t* top = source + (
        static_cast<size_t>(y_index) * static_cast<size_t>(source_width)
        + static_cast<size_t>(x_index)) * 3U;
    const std::uint8_t* top_next = source + (
        static_cast<size_t>(y_index) * static_cast<size_t>(source_width)
        + static_cast<size_t>(x_next)) * 3U;
    const std::uint8_t* bottom = source + (
        static_cast<size_t>(y_next) * static_cast<size_t>(source_width)
        + static_cast<size_t>(x_index)) * 3U;
    const std::uint8_t* bottom_next = source + (
        static_cast<size_t>(y_next) * static_cast<size_t>(source_width)
        + static_cast<size_t>(x_next)) * 3U;
    const float inverse_x = 1.0F - x_weight;
    const float inverse_y = 1.0F - y_weight;
    for (int channel = 0; channel < 3; ++channel) {
        const float top_value = static_cast<float>(top[channel]) * inverse_x
            + static_cast<float>(top_next[channel]) * x_weight;
        const float bottom_value = static_cast<float>(bottom[channel]) * inverse_x
            + static_cast<float>(bottom_next[channel]) * x_weight;
        destination[channel] = rounded_colour_byte(
            static_cast<double>(top_value * inverse_y + bottom_value * y_weight));
    }
}

inline std::uint8_t kfp_dithered_colour_byte(
    double value,
    int x,
    int y,
    int channel
) {
    // Kalles adds a deterministic 8-bit ordered dither before truncating the
    // final sRGB colour. This is intentionally separate from the Burtle hash
    // used by Kalles for positional jitter.
    if (!std::isfinite(value)) value = 0.0;
    const double mask = static_cast<double>(kfp_colour_dither_mask(x, y, channel));
    return static_cast<std::uint8_t>(std::clamp(
        static_cast<int>(std::floor(std::clamp(value, 0.0, 255.0) + mask)),
        0,
        255));
}

inline std::uint8_t kfp_dithered_srgb_byte(
    float value,
    int x,
    int y,
    int channel
) {
    // Kalles' colour.h keeps the colour in float sRGB [0, 1] until this
    // exact operation.  Keeping this as a separate helper is important: the
    // older fast default kernel stores its palette in byte units, while the
    // general KFP SetColor path must not quantise to 0..255 before dither.
    if (!std::isfinite(value)) value = 0.0F;
    const float mask = static_cast<float>(
        kfp_colour_dither_mask(x, y, channel));
    const float quantized = 255.0F * value + mask;
    return static_cast<std::uint8_t>(std::clamp(
        static_cast<int>(std::floor(quantized)),
        0,
        255));
}

inline double kfp_safe_sample(
    const float* field,
    int width,
    int height,
    int x,
    int y,
    int max_iter,
    double field_bias = 0.0
) {
    x = std::clamp(x, 0, width - 1);
    y = std::clamp(y, 0, height - 1);
    const float value = field[static_cast<size_t>(y) * static_cast<size_t>(width)
        + static_cast<size_t>(x)];
    // Kalles stores an interior pixel as (nIter=max_iter, offs=0).  Its
    // reconstructed neighbour value is therefore max_iter + 1, even though
    // the centre pixel itself is painted with the configured interior colour.
    // Preserving that extra unit is important: it keeps the distance/slope
    // signal sharp at the set boundary instead of making a flat square around
    // every interior tile.
    if (!std::isfinite(value)) {
        // Invalid scalar samples are not proof of membership.  Treat them as
        // a finite escaped fallback so one numerical glitch cannot paint a
        // rectangular black interior through the atlas.
        return 0.0;
    }
    const double interior_limit = static_cast<double>(max_iter) - field_bias;
    if (static_cast<double>(value) >= interior_limit) {
        return static_cast<double>(max_iter) + 1.0;
    }
    return std::clamp(
        static_cast<double>(value) + field_bias,
        0.0,
        static_cast<double>(max_iter));
}

inline bool kfp_inside_value(
    float value,
    int max_iter,
    double field_bias = 0.0
) noexcept {
    return std::isfinite(value)
        && static_cast<double>(value)
            >= static_cast<double>(max_iter) - field_bias;
}

// The renderer stores the compact smooth-iteration value produced by its
// own scalar path. Kalles reconstructs its colour value from nIter and
// transition as
//
//   nIter + 1 - log(log(|z|) / log(10000)) / log(power)
//
// while the scalar renderer already contains the `nIter + 1` part and the
// `-log(log(|z|))/log(power)` part. Add only the selected bailout's
// `log(log(radius))/log(power)` term inside the KFP transfer; changing the
// shared scalar field would move ordinary Aurora palettes as well. The
// default Kalles palette uses only the distance transfer, so its branch-free
// hot path keeps the unshifted scalar differences where the constant cancels
// exactly.
inline double kfp_smooth_offset(const FractalKfpOptions& options) noexcept {
    if (options.smooth_method != 0
        || !std::isfinite(options.power) || options.power <= 0.0
        || std::abs(options.power - 1.0) < 1.0e-12) {
        return 0.0;
    }
    double radius = 10000.0;
    switch (options.bailout_radius_preset) {
        case 0:
            radius = 10000.0;
            break;
        case 1:
            radius = 2.0;
            break;
        case 2:
            radius = std::pow(2.0, 1.0 / (options.power - 1.0));
            break;
        case 3:
            radius = options.bailout_radius_custom;
            break;
        default:
            return 0.0;
    }
    if (!std::isfinite(radius) || radius <= 1.0) return 0.0;
    return std::log(std::log(radius)) / std::log(options.power);
}

// The bundled/default profile always uses Power=2. Keep its edge-only
// correction in float so the AVX and scalar fast paths take the same route.
constexpr float KFP_DEFAULT_SMOOTH_OFFSET = 3.2032545F;

inline double kfp_colour_sample(
    const float* field,
    int width,
    int height,
    int x,
    int y,
    int max_iter,
    double smooth_offset,
    double field_bias = 0.0
) {
    x = std::clamp(x, 0, width - 1);
    y = std::clamp(y, 0, height - 1);
    const float value = field[static_cast<size_t>(y) * static_cast<size_t>(width)
        + static_cast<size_t>(x)];
    // Match the scalar colouriser's escaped fallback: invalid samples are
    // treated as a zero iteration value, then receive the same fixed smooth
    // offset as every other escaped sample.  Returning a bare zero here made
    // the precise native stencil disagree with the portable Kalles path
    // beside a numerical fault.
    if (!std::isfinite(value)) return std::max(0.0, smooth_offset);
    if (static_cast<double>(value)
        >= static_cast<double>(max_iter) - field_bias) {
        return static_cast<double>(max_iter) + 1.0;
    }
    return std::max(
        0.0,
        std::clamp(
            static_cast<double>(value) + field_bias,
            0.0,
            static_cast<double>(max_iter))
            + smooth_offset);
}

inline double kfp_reflected_sample(
    const float* field,
    int width,
    int height,
    int x,
    int y,
    int offset_x,
    int offset_y,
    int max_iter,
    double centre,
    double smooth_offset,
    double field_bias = 0.0
) {
    const int sample_x = x + offset_x;
    const int sample_y = y + offset_y;
    if (sample_x >= 0 && sample_x < width && sample_y >= 0 && sample_y < height) {
        return kfp_colour_sample(
            field,
            width,
            height,
            sample_x,
            sample_y,
            max_iter,
            smooth_offset,
            field_bias);
    }
    // Match Kalles' reflected boundary stencil. A one-pixel dimension has no
    // opposite sample, so use the centre value instead of propagating NaNs.
    const int opposite_x = x - offset_x;
    const int opposite_y = y - offset_y;
    if (opposite_x >= 0 && opposite_x < width
        && opposite_y >= 0 && opposite_y < height) {
        return 2.0 * centre - kfp_colour_sample(
            field,
            width,
            height,
            opposite_x,
            opposite_y,
            max_iter,
            smooth_offset,
            field_bias);
    }
    return centre;
}

inline double kfp_difference_magnitude(
    int differences,
    double centre,
    double left,
    double right,
    double up,
    double down,
    double top_left,
    double top_right,
    double bottom_left,
    double bottom_right
) {
    // Kalles' gradient.cpp calls this hypot1, which is deliberately just
    // sqrt(x*x + y*y), rather than the range-stable std::hypot.  Keeping the
    // same operation matters at palette boundaries and avoids a libm helper
    // call in the two least-squares operators.
    const auto kfp_hypot1 = [](double x, double y) noexcept {
        return std::sqrt(x * x + y * y);
    };
    constexpr double inverse_sqrt_two = 0.7071067811865475244008443621048490;
    constexpr double diagonal_distance_squared = 2.0;
    if (differences == 0) {
        // CFraktalSFT::SetColor uses the historical literal 1.414 here,
        // rather than recomputing sqrt(2). Keep that last bit of arithmetic
        // identical for KFP profiles using the traditional stencil.
        constexpr double axis_scale = 1.414;
        return std::abs(left - centre) * axis_scale
            + std::abs(up - centre) * axis_scale
            // Diagonal neighbours are sqrt(2) pixels away, cancelling the
            // historical sqrt(2) factor used before distance normalization.
            + std::abs(top_left - centre)
            + std::abs(bottom_left - centre);
    }
    if (differences == 1) {
        const double squared =
            (left - centre) * (left - centre)
            + (right - centre) * (right - centre)
            + (up - centre) * (up - centre)
            + (down - centre) * (down - centre)
            + ((top_left - centre) * (top_left - centre)
                + (bottom_right - centre) * (bottom_right - centre))
                * inverse_sqrt_two * inverse_sqrt_two
            + ((bottom_left - centre) * (bottom_left - centre)
                + (top_right - centre) * (top_right - centre))
                * inverse_sqrt_two * inverse_sqrt_two;
        return std::sqrt(std::max(0.0, squared * 0.25)) * 2.8284271247461903;
    }
    if (differences == 2) {
        const double squared =
            (right - left) * (right - left) * 0.25
            + (down - up) * (down - up) * 0.25
            + (bottom_right - top_left) * (bottom_right - top_left) * 0.125
            + (top_right - bottom_left) * (top_right - bottom_left) * 0.125;
        return std::sqrt(std::max(0.0, squared * 0.5)) * 2.8284271247461903;
    }
    if (differences == 3) {
        const double squared =
            (top_left - centre) * (top_left - centre) / diagonal_distance_squared
            + (left - up) * (left - up) / diagonal_distance_squared;
        return std::sqrt(std::max(0.0, squared)) * 2.8284271247461903;
    }
    if (differences == 4) {
        const double dx = ((up - top_left) + (centre - left)) * 0.5;
        const double dy = ((left - top_left) + (centre - up)) * 0.5;
        return kfp_hypot1(dx, dy) * 2.8284271247461903;
    }
    if (differences == 5) {
        const double dx = (right + top_right + bottom_right
            - left - top_left - bottom_left) / 6.0;
        const double dy = (down + bottom_left + bottom_right
            - up - top_left - top_right) / 6.0;
        return kfp_hypot1(dx, dy) * 2.8284271247461903;
    }
    if (differences == 6) {
        const double laplacian = top_left + 4.0 * up + top_right
            + 4.0 * left - 20.0 * centre + 4.0 * right
            + bottom_left + 4.0 * down + bottom_right;
        return std::sqrt(std::abs(laplacian / 6.0 * 1.4426950408889634))
            * 2.8284271247461903;
    }
    // Differences_Analytic is the only remaining enum value. Kalles gets
    // this transfer from its optional DE planes; the scalar ABI has no such
    // planes, so its own SetColor leaves the distance value at zero.
    return 0.0;
}

inline void kfp_hsv_to_rgb(
    float hue,
    float saturation,
    float value,
    float& red,
    float& green,
    float& blue
) {
    // This is Kalles' colour.h::hsv2rgb.  The source deliberately uses float
    // here, even though the wave accumulator in SetColor is double.  That
    // narrowing is visible at palette boundaries and must be retained for
    // imported .kfp files that use MultiColor.
    const float scaled_hue = hue * 6.0F;
    const int sector = static_cast<int>(std::floor(scaled_hue));
    float fraction = scaled_hue - static_cast<float>(sector);
    if ((sector & 1) == 0) fraction = 1.0F - fraction;
    const float minimum = value * (1.0F - saturation);
    const float transition = value * (1.0F - saturation * fraction);
    red = 0.0F;
    green = 0.0F;
    blue = 0.0F;
    switch (sector) {
        case 0:
        case 6:
            red = minimum;
            green = transition;
            blue = value;
            break;
        case 1:
            red = minimum;
            green = value;
            blue = transition;
            break;
        case 2:
            red = transition;
            green = value;
            blue = minimum;
            break;
        case 3:
            red = value;
            green = transition;
            blue = minimum;
            break;
        case 4:
            red = value;
            green = minimum;
            blue = transition;
            break;
        case 5:
            red = transition;
            green = minimum;
            blue = value;
            break;
    }
}

struct KfpSlopeDirection {
    double cosine = 1.0;
    double sine = 0.0;
};

// Values that are invariant for every pixel in one KFP colour pass. Keeping
// them beside the pixel writer makes the exact Kalles path cheaper without
// moving any arithmetic out of the reference-defined per-pixel stages.
struct KfpPixelContext {
    double smooth_offset = 0.0;
    double smooth_power_log = 0.0;
    double smooth_bailout_radius = 0.0;
    double smooth_bailout_log = 0.0;
    double smooth_norm_exponent = 0.0;
    int smooth_method = 0;
    bool has_iteration_planes = false;
    bool has_constant_bailout = false;
    bool has_smooth_power_log = false;
    bool has_smooth_bailout_log = false;
    bool has_smooth_norm_exponent = false;
    bool texture_active = false;
    bool needs_difference = false;
    bool needs_slopes = false;
};

inline double kfp_selected_bailout_radius(
    const FractalKfpOptions& options
) noexcept {
    double radius = 10000.0;
    switch (options.bailout_radius_preset) {
        case 0:
            radius = 10000.0;
            break;
        case 1:
            radius = 2.0;
            break;
        case 2:
            if (options.power > 1.0 && std::isfinite(options.power)) {
                radius = std::pow(2.0, 1.0 / (options.power - 1.0));
            } else {
                return 0.0;
            }
            break;
        case 3:
            radius = options.bailout_radius_custom;
            break;
        default:
            return 0.0;
    }
    return std::isfinite(radius) && radius > 1.0 ? radius : 0.0;
}

inline double kfp_selected_bailout_norm(
    const FractalKfpOptions& options
) noexcept {
    switch (options.bailout_norm_preset) {
        case 0:
            return 1.0;
        case 1:
            return 2.0;
        case 2:
            return std::numeric_limits<double>::infinity();
        case 3:
            return std::isfinite(options.bailout_norm_custom)
                && options.bailout_norm_custom > 0.0
                ? options.bailout_norm_custom : 0.0;
        default:
            return 0.0;
    }
}

inline bool kfp_planes_have_iteration(const FractalKfpPlanes* planes) noexcept {
    return planes != nullptr
        && (planes->orbit_iteration != nullptr || planes->iteration != nullptr);
}

inline KfpPixelContext kfp_pixel_context(
    const FractalKfpOptions& options,
    double smooth_offset,
    const FractalKfpPlanes* planes
) noexcept {
    KfpPixelContext context;
    context.smooth_offset = smooth_offset;
    context.smooth_method = options.smooth_method;
    context.has_iteration_planes = kfp_planes_have_iteration(planes);
    context.has_constant_bailout = planes != nullptr
        && planes->bailout == nullptr;
    context.smooth_bailout_radius = kfp_selected_bailout_radius(options);
    if (std::isfinite(options.power) && options.power > 1.0) {
        context.smooth_power_log = std::log(options.power);
        context.has_smooth_power_log = std::isfinite(context.smooth_power_log)
            && context.smooth_power_log != 0.0;
    }
    if (context.has_constant_bailout
        && context.smooth_bailout_radius > 1.0
        && std::isfinite(context.smooth_bailout_radius)) {
        context.smooth_bailout_log = std::log(context.smooth_bailout_radius);
        context.has_smooth_bailout_log = std::isfinite(context.smooth_bailout_log)
            && context.smooth_bailout_log != 0.0;
    }
    const double norm = kfp_selected_bailout_norm(options);
    if (norm > 0.0 && std::isfinite(norm)) {
        context.smooth_norm_exponent = 1.0 / norm;
        context.has_smooth_norm_exponent = true;
    } else if (std::isinf(norm) && norm > 0.0) {
        context.smooth_norm_exponent = 1.0;
        context.has_smooth_norm_exponent = true;
    }
    context.texture_active = planes != nullptr
        && planes->texture_rgb != nullptr
        && options.texture_enabled != 0;
    context.needs_difference = options.color_method >= 5
        && options.color_method <= 8;
    context.needs_slopes = options.slopes && options.slope_power > 0.0
        && options.slope_ratio > 0.0;
    return context;
}

inline double kfp_plane_bailout(
    const FractalKfpOptions& options,
    const FractalKfpPlanes* planes,
    size_t index
) noexcept {
    if (planes != nullptr && planes->bailout != nullptr) {
        const double value = planes->bailout[index];
        if (std::isfinite(value) && value > 1.0) return value;
    }
    return kfp_selected_bailout_radius(options);
}

inline double kfp_plane_smooth_part(
    const FractalKfpOptions& options,
    const FractalKfpPlanes* planes,
    size_t index,
    const KfpPixelContext* pixel_context = nullptr
) noexcept {
    if (planes == nullptr || planes->test1 == nullptr) return 0.0;
    const double test1 = planes->test1[index];
    if (!std::isfinite(test1) || test1 <= 0.0) return 0.0;
    double smooth = 0.0;
    const int smooth_method = pixel_context != nullptr
        ? pixel_context->smooth_method : options.smooth_method;
    if (smooth_method == 0) {
        const double radius = pixel_context != nullptr
            && pixel_context->has_constant_bailout
            ? pixel_context->smooth_bailout_radius
            : kfp_plane_bailout(options, planes, index);
        const double power = options.power;
        if (radius > 1.0 && std::isfinite(power) && power > 1.0) {
            const double numerator = std::log(std::sqrt(test1));
            const double denominator = pixel_context != nullptr
                && pixel_context->has_smooth_bailout_log
                ? pixel_context->smooth_bailout_log
                : std::log(radius);
            const double power_log = pixel_context != nullptr
                && pixel_context->has_smooth_power_log
                ? pixel_context->smooth_power_log
                : std::log(power);
            if (std::isfinite(numerator) && numerator > 0.0
                && std::isfinite(denominator) && denominator != 0.0
                && std::isfinite(power_log) && power_log != 0.0) {
                smooth = 1.0 - std::log(numerator / denominator) / power_log;
            }
        }
    } else if (smooth_method == 1 && planes->test2 != nullptr) {
        const double test2 = planes->test2[index];
        const double fallback_norm = pixel_context == nullptr
            ? kfp_selected_bailout_norm(options) : 0.0;
        const bool has_norm = pixel_context != nullptr
            ? pixel_context->has_smooth_norm_exponent
            : fallback_norm > 0.0;
        if (std::isfinite(test2) && test2 >= 0.0 && has_norm) {
            const double exponent = pixel_context != nullptr
                && pixel_context->has_smooth_norm_exponent
                ? pixel_context->smooth_norm_exponent
                : (std::isinf(fallback_norm) ? 1.0 : 1.0 / fallback_norm);
            const double root1 = std::pow(test1, exponent);
            const double root2 = std::pow(test2, exponent);
            const double denominator = root1 - root2;
            const double radius = pixel_context != nullptr
                && pixel_context->has_constant_bailout
                ? pixel_context->smooth_bailout_radius
                : kfp_plane_bailout(options, planes, index);
            if (std::isfinite(root1) && std::isfinite(root2)
                && std::isfinite(radius) && denominator != 0.0) {
                smooth = 1.0 - (root1 - radius) / denominator;
            }
        }
    }
    return std::isfinite(smooth) ? smooth : 0.0;
}

struct KfpOrbitSample {
    double colour = 0.0;
    bool inside = false;
};

inline KfpOrbitSample kfp_plane_orbit_sample(
    int max_iter,
    const FractalKfpOptions& options,
    const FractalKfpPlanes* planes,
    size_t index,
    const KfpPixelContext* pixel_context = nullptr
) noexcept {
    if (planes == nullptr || planes->orbit_iteration == nullptr) return {};
    const std::int64_t raw = std::max<std::int64_t>(
        0, planes->orbit_iteration[index]);
    if (raw >= max_iter) {
        return {static_cast<double>(max_iter) + 1.0, true};
    }
    const double combined = static_cast<double>(raw)
        + kfp_plane_smooth_part(options, planes, index, pixel_context);
    const bool inside = combined >= static_cast<double>(max_iter);
    return {
        inside ? static_cast<double>(max_iter) + 1.0 : combined,
        inside,
    };
}

inline double kfp_plane_colour_sample(
    const float* field,
    int width,
    int height,
    int x,
    int y,
    int max_iter,
    const FractalKfpOptions& options,
    const FractalKfpPlanes* planes,
    double smooth_offset = std::numeric_limits<double>::quiet_NaN(),
    const KfpPixelContext* pixel_context = nullptr
) noexcept {
    x = std::clamp(x, 0, width - 1);
    y = std::clamp(y, 0, height - 1);
    const size_t index = static_cast<size_t>(y) * static_cast<size_t>(width)
        + static_cast<size_t>(x);
    if (planes != nullptr && planes->orbit_iteration != nullptr) {
        const KfpOrbitSample sample = kfp_plane_orbit_sample(
            max_iter, options, planes, index, pixel_context);
        // OutputIterationData stores nTrans = 1 - fraction.  SetColor then
        // reconstructs the continuous value as nPixels + 1 - nTrans, which
        // is the original raw counter plus its smoothing fraction.  Do not
        // subtract the fraction twice here: that would mirror every Kalles
        // transition around the next integer.
        return sample.colour;
    }
    if (planes != nullptr && planes->iteration != nullptr) {
        const std::int64_t stored = planes->iteration[index];
        if (stored >= max_iter) return static_cast<double>(max_iter) + 1.0;
        const double transition = planes->transition != nullptr
            ? planes->transition[index] : 1.0;
        return static_cast<double>(std::max<std::int64_t>(0, stored))
            + 1.0 - (std::isfinite(transition) ? transition : 1.0);
    }
    if (!std::isfinite(smooth_offset)) {
        smooth_offset = kfp_smooth_offset(options);
    }
    return kfp_colour_sample(field, width, height, x, y, max_iter,
        smooth_offset, options.field_bias);
}

inline bool kfp_plane_inside(
    int width,
    int height,
    int x,
    int y,
    int max_iter,
    const FractalKfpOptions& options,
    const FractalKfpPlanes* planes,
    float scalar_value,
    const KfpPixelContext* pixel_context = nullptr
) noexcept {
    (void)height;
    if (planes != nullptr && planes->orbit_iteration != nullptr) {
        const size_t index = static_cast<size_t>(y) * static_cast<size_t>(width)
            + static_cast<size_t>(x);
        return kfp_plane_orbit_sample(
            max_iter, options, planes, index, pixel_context).inside;
    }
    if (planes != nullptr && planes->iteration != nullptr) {
        const size_t index = static_cast<size_t>(y) * static_cast<size_t>(width)
            + static_cast<size_t>(x);
        return planes->iteration[index] >= max_iter;
    }
    return kfp_inside_value(scalar_value, max_iter, options.field_bias);
}

inline double kfp_plane_phase(
    const FractalKfpPlanes* planes,
    size_t index
) noexcept {
    if (planes == nullptr || planes->phase == nullptr) return 0.0;
    const double value = planes->phase[index];
    return std::isfinite(value) ? value : 0.0;
}

inline double kfp_plane_value(
    const double* values,
    size_t index
) noexcept {
    if (values == nullptr) return 0.0;
    const double value = values[index];
    return std::isfinite(value) ? value : 0.0;
}

inline double kfp_plane_stored_iteration(
    const float* field,
    int width,
    int height,
    int x,
    int y,
    int max_iter,
    const FractalKfpOptions& options,
    const FractalKfpPlanes* planes,
    const KfpPixelContext* pixel_context = nullptr
) noexcept {
    const double value = kfp_plane_colour_sample(
        field, width, height, x, y, max_iter, options, planes,
        std::numeric_limits<double>::quiet_NaN(), pixel_context);
    if (value > static_cast<double>(max_iter)) {
        return static_cast<double>(max_iter);
    }
    return std::floor(std::max(0.0, value));
}

inline int kfp_texture_index(double value, int limit) noexcept {
    if (limit <= 0 || !std::isfinite(value)) return 0;
    if (value <= 0.0) return 0;
    if (value >= static_cast<double>(limit - 1)) return limit - 1;
    return static_cast<int>(value);
}

// This predicate is retained for profiling the old approximation, but it is
// deliberately not used by production KFP rendering. That kernel replaces
// libm log/atan and reconstructs only the bundled profile; it cannot be
// pixel-identical to Kalles' complete SetColor path. The generic writer is
// now the authoritative implementation for direct, crop, and atlas output.
inline bool kfp_is_default_fast_options(
    const FractalKfpOptions& options
) noexcept {
    return options.iter_div == 0.01
        && options.color_offset == 0.0
        && options.ratio == 360.0
        && options.color_method == 7
        && options.smooth_method == 0
        && options.smooth != 0
        && options.flat == 0
        && options.inverse_transition == 0
        && options.phase_color_strength == 0.0
        && options.multi_color == 0
        && options.blend_multi_color == 0
        && options.multi_color_count == 0
        && options.power == 2.0
        && options.slopes != 0
        && options.slope_power == 50.0
        && options.slope_ratio == 20.0
        && options.slope_angle == 45.0
        && options.differences == 3
        && options.interior_color[0] == 0
        && options.interior_color[1] == 0
        && options.interior_color[2] == 0;
}

inline float kfp_fast_log1p(float value) noexcept {
    if (!(value > 0.0F)) return 0.0F;
    const float shifted = 1.0F + value;
    if (!std::isfinite(shifted)) return static_cast<float>(std::log1p(value));
    int exponent = 0;
    float mantissa = std::frexp(shifted, &exponent) * 2.0F;
    --exponent;
    const float z = (mantissa - 1.0F) / (mantissa + 1.0F);
    const float z_squared = z * z;
    // The atanh series converges rapidly because the reduced mantissa is in
    // [1, 2). Keep a few more odd terms than the original approximation: KFP
    // divides this result by IterDiv (0.01 in the default palette), so a
    // seemingly tiny log error can otherwise move a colour several 8-bit
    // levels at a cyclic LUT boundary.
    const float series = 1.0F + z_squared * (
        1.0F / 3.0F + z_squared * (
            1.0F / 5.0F + z_squared * (
                1.0F / 7.0F + z_squared * (
                    1.0F / 9.0F + z_squared * (
                        1.0F / 11.0F + z_squared * (
                            1.0F / 13.0F + z_squared * (
                                1.0F / 15.0F + z_squared * (
                                    1.0F / 17.0F + z_squared / 19.0F))))))));
    return 2.0F * z * series
        + static_cast<float>(exponent) * 0.6931471805599453F;
}

inline float kfp_fast_atan_nonnegative(float value) noexcept {
    if (!std::isfinite(value)) return 1.5707963267948966F;
    const bool invert = value > 1.0F;
    const float x = invert ? 1.0F / value : value;
    const float x_squared = x * x;
    // Minimax-like odd polynomial on [0, 1]. Its maximum error is small
    // compared with the slope blend's final 8-bit quantisation, while it is
    // substantially cheaper and easier for the compiler to vectorise than
    // a scalar libm atan call.
    const float result = x * (0.9998660F + x_squared * (
        -0.3302995F + x_squared * (
            0.1801410F + x_squared * (
                -0.0851330F + x_squared * 0.0208351F))));
    return invert ? 1.5707963267948966F - result : result;
}

inline float kfp_fast_sample_inner(
    const float* field,
    size_t index,
    int max_iter,
    double field_bias = 0.0
) noexcept {
    const float value = field[index];
    if (!std::isfinite(value)) return 0.0F;
    const float bias = static_cast<float>(field_bias);
    const float limit = static_cast<float>(max_iter) - bias;
    if (value >= limit) {
        return static_cast<float>(max_iter) + 1.0F - bias;
    }
    return std::clamp(value, -bias, limit);
}

#if defined(__AVX2__)
inline __m256 kfp_fast_inside_avx(__m256 raw, __m256 limit) {
    const __m256 absolute = _mm256_andnot_ps(
        _mm256_set1_ps(-0.0F), raw);
    const __m256 finite = _mm256_cmp_ps(
        absolute,
        _mm256_set1_ps(std::numeric_limits<float>::infinity()),
        _CMP_LT_OQ);
    return _mm256_and_ps(
        finite,
        _mm256_cmp_ps(raw, limit, _CMP_GE_OQ));
}

inline __m256 kfp_fast_sample_inner_avx(
    __m256 raw,
    int max_iter,
    double field_bias = 0.0
) {
    const __m256 absolute = _mm256_andnot_ps(
        _mm256_set1_ps(-0.0F), raw);
    const __m256 finite = _mm256_cmp_ps(
        absolute,
        _mm256_set1_ps(std::numeric_limits<float>::infinity()),
        _CMP_LT_OQ);
    const float bias = static_cast<float>(field_bias);
    const __m256 zero = _mm256_setzero_ps();
    const __m256 lower = _mm256_set1_ps(-bias);
    const __m256 limit = _mm256_set1_ps(static_cast<float>(max_iter) - bias);
    __m256 value = _mm256_max_ps(raw, lower);
    value = _mm256_min_ps(value, limit);
    value = _mm256_blendv_ps(zero, value, finite);
    const __m256 inside = kfp_fast_inside_avx(raw, limit);
    return _mm256_blendv_ps(
        value,
        _mm256_set1_ps(static_cast<float>(max_iter) + 1.0F - bias),
        inside);
}

inline __m256 kfp_fast_log1p_avx(__m256 value) {
    const __m256 shifted = _mm256_add_ps(value, _mm256_set1_ps(1.0F));
    const __m256i bits = _mm256_castps_si256(shifted);
    const __m256i exponent_bits = _mm256_and_si256(
        _mm256_srli_epi32(bits, 23),
        _mm256_set1_epi32(0xff));
    const __m256i exponent = _mm256_sub_epi32(
        exponent_bits,
        _mm256_set1_epi32(127));
    const __m256i mantissa_bits = _mm256_or_si256(
        _mm256_and_si256(bits, _mm256_set1_epi32(0x007fffff)),
        _mm256_set1_epi32(0x3f800000));
    const __m256 mantissa = _mm256_castsi256_ps(mantissa_bits);
    const __m256 z = _mm256_div_ps(
        _mm256_sub_ps(mantissa, _mm256_set1_ps(1.0F)),
        _mm256_add_ps(mantissa, _mm256_set1_ps(1.0F)));
    const __m256 z_squared = _mm256_mul_ps(z, z);
    __m256 series = _mm256_set1_ps(1.0F / 19.0F);
    series = _mm256_add_ps(
        _mm256_set1_ps(1.0F / 17.0F), _mm256_mul_ps(z_squared, series));
    series = _mm256_add_ps(
        _mm256_set1_ps(1.0F / 15.0F), _mm256_mul_ps(z_squared, series));
    series = _mm256_add_ps(
        _mm256_set1_ps(1.0F / 13.0F), _mm256_mul_ps(z_squared, series));
    series = _mm256_add_ps(
        _mm256_set1_ps(1.0F / 11.0F), _mm256_mul_ps(z_squared, series));
    series = _mm256_add_ps(
        _mm256_set1_ps(1.0F / 9.0F), _mm256_mul_ps(z_squared, series));
    series = _mm256_add_ps(
        _mm256_set1_ps(1.0F / 7.0F), _mm256_mul_ps(z_squared, series));
    series = _mm256_add_ps(
        _mm256_set1_ps(1.0F / 5.0F), _mm256_mul_ps(z_squared, series));
    series = _mm256_add_ps(
        _mm256_set1_ps(1.0F / 3.0F), _mm256_mul_ps(z_squared, series));
    series = _mm256_add_ps(
        _mm256_set1_ps(1.0F), _mm256_mul_ps(z_squared, series));
    return _mm256_add_ps(
        _mm256_mul_ps(_mm256_set1_ps(2.0F), _mm256_mul_ps(z, series)),
        _mm256_mul_ps(
            _mm256_cvtepi32_ps(exponent),
            _mm256_set1_ps(0.6931471805599453F)));
}

inline __m256 kfp_fast_atan_nonnegative_avx(__m256 value) {
    const __m256 one = _mm256_set1_ps(1.0F);
    const __m256 invert = _mm256_cmp_ps(value, one, _CMP_GT_OQ);
    const __m256 x = _mm256_blendv_ps(
        value,
        _mm256_div_ps(one, value),
        invert);
    const __m256 x_squared = _mm256_mul_ps(x, x);
    __m256 polynomial = _mm256_set1_ps(0.0208351F);
    polynomial = _mm256_add_ps(
        _mm256_set1_ps(-0.0851330F), _mm256_mul_ps(x_squared, polynomial));
    polynomial = _mm256_add_ps(
        _mm256_set1_ps(0.1801410F), _mm256_mul_ps(x_squared, polynomial));
    polynomial = _mm256_add_ps(
        _mm256_set1_ps(-0.3302995F), _mm256_mul_ps(x_squared, polynomial));
    polynomial = _mm256_add_ps(
        _mm256_set1_ps(0.9998660F), _mm256_mul_ps(x_squared, polynomial));
    const __m256 result = _mm256_mul_ps(x, polynomial);
    return _mm256_blendv_ps(
        result,
        _mm256_sub_ps(_mm256_set1_ps(1.5707963267948966F), result),
        invert);
}

inline void write_kfp_default_fast_block8(
    const float* field,
    int width,
    int height,
    int source_x,
    int source_y,
    int dither_x,
    int dither_y,
    int spatial_width,
    int max_iter,
    const std::uint8_t* lut,
    int lut_size,
    std::uint8_t* destination,
    double field_bias = 0.0
) {
    (void) height;
    const size_t centre_index = static_cast<size_t>(source_y)
        * static_cast<size_t>(width) + static_cast<size_t>(source_x);
    const __m256 raw = _mm256_loadu_ps(field + centre_index);
    const __m256 absolute = _mm256_andnot_ps(
        _mm256_set1_ps(-0.0F), raw);
    const __m256 finite = _mm256_cmp_ps(
        absolute,
        _mm256_set1_ps(std::numeric_limits<float>::infinity()),
        _CMP_LT_OQ);
    const float bias = static_cast<float>(field_bias);
    const __m256 lower_bound = _mm256_set1_ps(-bias);
    const __m256 limit = _mm256_set1_ps(static_cast<float>(max_iter) - bias);
    const __m256 inside = kfp_fast_inside_avx(raw, limit);
    const __m256 zero = _mm256_setzero_ps();
    __m256 centre = _mm256_max_ps(raw, lower_bound);
    centre = _mm256_min_ps(centre, limit);
    centre = _mm256_blendv_ps(zero, centre, finite);

    const size_t row_width = static_cast<size_t>(width);
    const __m256 left_raw = _mm256_loadu_ps(field + centre_index - 1U);
    const __m256 up_raw = _mm256_loadu_ps(field + centre_index - row_width);
    const __m256 top_left_raw = _mm256_loadu_ps(
        field + centre_index - row_width - 1U);
    const __m256 left = kfp_fast_sample_inner_avx(
        left_raw, max_iter, field_bias);
    const __m256 up = kfp_fast_sample_inner_avx(
        up_raw, max_iter, field_bias);
    const __m256 top_left = kfp_fast_sample_inner_avx(
        top_left_raw, max_iter, field_bias);
    const __m256 smooth_offset = _mm256_set1_ps(KFP_DEFAULT_SMOOTH_OFFSET);
    const __m256 difference_x = _mm256_sub_ps(
        _mm256_sub_ps(top_left, centre),
        _mm256_and_ps(
            kfp_fast_inside_avx(top_left_raw, limit),
            smooth_offset));
    const __m256 difference_y = _mm256_add_ps(
        _mm256_sub_ps(left, up),
        _mm256_sub_ps(
            _mm256_and_ps(kfp_fast_inside_avx(up_raw, limit), smooth_offset),
            _mm256_and_ps(kfp_fast_inside_avx(left_raw, limit), smooth_offset)));
    const __m256 gradient = _mm256_mul_ps(
        _mm256_sqrt_ps(_mm256_max_ps(
            zero,
            _mm256_mul_ps(
                _mm256_add_ps(
                    _mm256_mul_ps(difference_x, difference_x),
                    _mm256_mul_ps(difference_y, difference_y)),
                _mm256_set1_ps(0.5F)))),
        _mm256_set1_ps(2.8284271247461903F));
    const __m256 distance = _mm256_min_ps(
        _mm256_mul_ps(
            gradient,
            _mm256_set1_ps(static_cast<float>(spatial_width) / 640.0F)),
        _mm256_set1_ps(1.0e12F));
    const __m256 transfer = kfp_fast_log1p_avx(distance);
    const __m256 cycle = _mm256_set1_ps(static_cast<float>(lut_size));
    const __m256 raw_position = _mm256_mul_ps(transfer, _mm256_set1_ps(100.0F));
    const __m256 position = _mm256_sub_ps(
        raw_position,
        _mm256_mul_ps(_mm256_floor_ps(_mm256_div_ps(raw_position, cycle)), cycle));
    __m256i lower_index = _mm256_cvttps_epi32(position);
    lower_index = _mm256_min_epi32(lower_index, _mm256_set1_epi32(lut_size - 1));
    __m256i upper_index = _mm256_add_epi32(lower_index, _mm256_set1_epi32(1));
    const __m256i upper_wrap = _mm256_cmpgt_epi32(
        upper_index, _mm256_set1_epi32(lut_size - 1));
    upper_index = _mm256_blendv_epi8(
        upper_index, _mm256_setzero_si256(), upper_wrap);
    const __m256 fraction = _mm256_sub_ps(
        position,
        _mm256_cvtepi32_ps(lower_index));
    const __m256 inverse_fraction = _mm256_sub_ps(
        _mm256_set1_ps(1.0F), fraction);
    const __m256i three = _mm256_set1_epi32(3);
    // A packed RGB lookup reads four bytes at a time. The last three-byte
    // entry has no fourth byte inside the caller-owned LUT, so clamp gather
    // addresses to the preceding entry and patch last-entry lanes below.
    // Without this guard the AVX2 fast path performed an out-of-bounds read at
    // a perfectly valid palette cycle boundary, which could surface as noisy
    // pixels or a sporadic crash.
    const __m256i last_index = _mm256_set1_epi32(lut_size - 1);
    const __m256i safe_index = _mm256_set1_epi32(lut_size - 2);
    const __m256i lower_last = _mm256_cmpeq_epi32(lower_index, last_index);
    const __m256i upper_last = _mm256_cmpeq_epi32(upper_index, last_index);
    const __m256i lower_offsets = _mm256_mullo_epi32(
        _mm256_min_epi32(lower_index, safe_index), three);
    const __m256i upper_offsets = _mm256_mullo_epi32(
        _mm256_min_epi32(upper_index, safe_index), three);
    const __m256i lower_packed = _mm256_i32gather_epi32(
        reinterpret_cast<const int*>(lut), lower_offsets, 1);
    const __m256i upper_packed = _mm256_i32gather_epi32(
        reinterpret_cast<const int*>(lut), upper_offsets, 1);
    __m256 lower_red = _mm256_cvtepi32_ps(_mm256_and_si256(
        lower_packed, _mm256_set1_epi32(0xff)));
    __m256 lower_green = _mm256_cvtepi32_ps(_mm256_and_si256(
        _mm256_srli_epi32(lower_packed, 8), _mm256_set1_epi32(0xff)));
    __m256 lower_blue = _mm256_cvtepi32_ps(_mm256_and_si256(
        _mm256_srli_epi32(lower_packed, 16), _mm256_set1_epi32(0xff)));
    __m256 upper_red = _mm256_cvtepi32_ps(_mm256_and_si256(
        upper_packed, _mm256_set1_epi32(0xff)));
    __m256 upper_green = _mm256_cvtepi32_ps(_mm256_and_si256(
        _mm256_srli_epi32(upper_packed, 8), _mm256_set1_epi32(0xff)));
    __m256 upper_blue = _mm256_cvtepi32_ps(_mm256_and_si256(
        _mm256_srli_epi32(upper_packed, 16), _mm256_set1_epi32(0xff)));
    const __m256 lower_last_float = _mm256_castsi256_ps(lower_last);
    const __m256 upper_last_float = _mm256_castsi256_ps(upper_last);
    lower_red = _mm256_blendv_ps(
        lower_red,
        _mm256_set1_ps(static_cast<float>(lut[(lut_size - 1) * 3])),
        lower_last_float);
    lower_green = _mm256_blendv_ps(
        lower_green,
        _mm256_set1_ps(static_cast<float>(lut[(lut_size - 1) * 3 + 1])),
        lower_last_float);
    lower_blue = _mm256_blendv_ps(
        lower_blue,
        _mm256_set1_ps(static_cast<float>(lut[(lut_size - 1) * 3 + 2])),
        lower_last_float);
    upper_red = _mm256_blendv_ps(
        upper_red,
        _mm256_set1_ps(static_cast<float>(lut[(lut_size - 1) * 3])),
        upper_last_float);
    upper_green = _mm256_blendv_ps(
        upper_green,
        _mm256_set1_ps(static_cast<float>(lut[(lut_size - 1) * 3 + 1])),
        upper_last_float);
    upper_blue = _mm256_blendv_ps(
        upper_blue,
        _mm256_set1_ps(static_cast<float>(lut[(lut_size - 1) * 3 + 2])),
        upper_last_float);
    __m256 red = _mm256_add_ps(
        _mm256_mul_ps(lower_red, inverse_fraction),
        _mm256_mul_ps(upper_red, fraction));
    __m256 green = _mm256_add_ps(
        _mm256_mul_ps(lower_green, inverse_fraction),
        _mm256_mul_ps(upper_green, fraction));
    __m256 blue = _mm256_add_ps(
        _mm256_mul_ps(lower_blue, inverse_fraction),
        _mm256_mul_ps(upper_blue, fraction));

    const __m256 horizontal = _mm256_sub_ps(
        _mm256_sub_ps(left, centre),
        _mm256_and_ps(
            kfp_fast_inside_avx(left_raw, limit),
            smooth_offset));
    const __m256 vertical = _mm256_sub_ps(
        _mm256_sub_ps(up, centre),
        _mm256_and_ps(
            kfp_fast_inside_avx(up_raw, limit),
            smooth_offset));
    const __m256 projected = _mm256_mul_ps(
        _mm256_add_ps(horizontal, vertical),
        _mm256_set1_ps(
            0.7071067811865475F * 50.0F
                * static_cast<float>(spatial_width) / 640.0F));
    const __m256 absolute_projected = _mm256_andnot_ps(
        _mm256_set1_ps(-0.0F), projected);
    const __m256 strength = _mm256_mul_ps(
        kfp_fast_atan_nonnegative_avx(absolute_projected),
        _mm256_set1_ps(0.2F / 1.5707963267948966F));
    const __m256 factor = _mm256_sub_ps(_mm256_set1_ps(1.0F), strength);
    const __m256 light_red = _mm256_add_ps(
        _mm256_mul_ps(red, factor), _mm256_mul_ps(_mm256_set1_ps(255.0F), strength));
    const __m256 light_green = _mm256_add_ps(
        _mm256_mul_ps(green, factor), _mm256_mul_ps(_mm256_set1_ps(255.0F), strength));
    const __m256 light_blue = _mm256_add_ps(
        _mm256_mul_ps(blue, factor), _mm256_mul_ps(_mm256_set1_ps(255.0F), strength));
    const __m256 dark = _mm256_cmp_ps(projected, zero, _CMP_GE_OQ);
    red = _mm256_blendv_ps(light_red, _mm256_mul_ps(red, factor), dark);
    green = _mm256_blendv_ps(light_green, _mm256_mul_ps(green, factor), dark);
    blue = _mm256_blendv_ps(light_blue, _mm256_mul_ps(blue, factor), dark);

    alignas(32) float red_values[8];
    alignas(32) float green_values[8];
    alignas(32) float blue_values[8];
    _mm256_store_ps(red_values, red);
    _mm256_store_ps(green_values, green);
    _mm256_store_ps(blue_values, blue);
    const int inside_bits = _mm256_movemask_ps(inside);
    for (int lane = 0; lane < 8; ++lane) {
        std::uint8_t* pixel = destination
            + static_cast<size_t>(dither_x + lane) * 3U;
        if ((inside_bits & (1 << lane)) != 0) {
            pixel[0] = 0;
            pixel[1] = 0;
            pixel[2] = 0;
        } else {
            pixel[0] = kfp_dithered_colour_byte(
                red_values[lane], dither_x + lane, dither_y, 0);
            pixel[1] = kfp_dithered_colour_byte(
                green_values[lane], dither_x + lane, dither_y, 1);
            pixel[2] = kfp_dithered_colour_byte(
                blue_values[lane], dither_x + lane, dither_y, 2);
        }
    }
}
#endif

inline void write_kfp_default_fast_pixel(
    const float* field,
    int width,
    int height,
    int x,
    int y,
    int dither_x,
    int dither_y,
    int spatial_width,
    int max_iter,
    const std::uint8_t* lut,
    int lut_size,
    std::uint8_t* destination,
    double field_bias = 0.0
) {
    const size_t centre_index = static_cast<size_t>(y)
        * static_cast<size_t>(width) + static_cast<size_t>(x);
    const float raw_value = field[centre_index];
    const float bias = static_cast<float>(field_bias);
    const float limit = static_cast<float>(max_iter) - bias;
    if (std::isfinite(raw_value) && raw_value >= limit) {
        destination[0] = 0;
        destination[1] = 0;
        destination[2] = 0;
        return;
    }
    const float centre = std::isfinite(raw_value)
        ? std::clamp(raw_value, -bias, limit)
        : 0.0F;

    float left;
    float up;
    float top_left;
    float right = centre;
    float down = centre;
    float stencil_centre = centre;
    bool left_inside = false;
    bool up_inside = false;
    bool top_left_inside = false;
    if (x > 0 && y > 0 && x + 1 < width && y + 1 < height) {
        left_inside = kfp_inside_value(
            field[centre_index - 1U], max_iter, field_bias);
        up_inside = kfp_inside_value(
            field[centre_index - static_cast<size_t>(width)], max_iter, field_bias);
        top_left_inside = kfp_inside_value(
            field[centre_index - static_cast<size_t>(width) - 1U], max_iter, field_bias);
        left = kfp_fast_sample_inner(
            field,
            centre_index - 1U,
            max_iter,
            field_bias);
        up = kfp_fast_sample_inner(
            field,
            centre_index - static_cast<size_t>(width),
            max_iter,
            field_bias);
        top_left = kfp_fast_sample_inner(
            field,
            centre_index - static_cast<size_t>(width) - 1U,
            max_iter,
            field_bias);
    } else {
        // Edge samples returned by kfp_reflected_sample are decoded back to
        // the absolute smooth-iteration domain. Bring the centred field's
        // centre into that same domain before taking the reflected stencil;
        // otherwise every edge pixel compares an absolute neighbour with a
        // bias-relative centre and gets a visibly different slope.
        stencil_centre = centre + bias + KFP_DEFAULT_SMOOTH_OFFSET;
        left = static_cast<float>(kfp_reflected_sample(
            field,
            width,
            height,
            x,
            y,
            -1,
            0,
            max_iter,
            stencil_centre,
            KFP_DEFAULT_SMOOTH_OFFSET,
            field_bias));
        up = static_cast<float>(kfp_reflected_sample(
            field,
            width,
            height,
            x,
            y,
            0,
            -1,
            max_iter,
            stencil_centre,
            KFP_DEFAULT_SMOOTH_OFFSET,
            field_bias));
        top_left = static_cast<float>(kfp_reflected_sample(
            field,
            width,
            height,
            x,
            y,
            -1,
            -1,
            max_iter,
            stencil_centre,
            KFP_DEFAULT_SMOOTH_OFFSET,
            field_bias));
        if (x == 0) {
            right = static_cast<float>(kfp_reflected_sample(
                field,
                width,
                height,
                x,
                y,
                1,
                0,
                max_iter,
                stencil_centre,
                KFP_DEFAULT_SMOOTH_OFFSET,
                field_bias));
        }
        if (y == 0) {
            down = static_cast<float>(kfp_reflected_sample(
                field,
                width,
                height,
                x,
                y,
                0,
                1,
                max_iter,
                stencil_centre,
                KFP_DEFAULT_SMOOTH_OFFSET,
                field_bias));
        }
    }

    const float difference_x = top_left - stencil_centre
        - (top_left_inside ? KFP_DEFAULT_SMOOTH_OFFSET : 0.0F);
    const float difference_y = left - up
        + (up_inside ? KFP_DEFAULT_SMOOTH_OFFSET : 0.0F)
        - (left_inside ? KFP_DEFAULT_SMOOTH_OFFSET : 0.0F);
    const float gradient = std::sqrt(std::max(
        0.0F,
        (difference_x * difference_x + difference_y * difference_y) * 0.5F))
        * 2.8284271247461903F;
    const float distance = std::clamp(
        gradient * static_cast<float>(spatial_width) / 640.0F,
        0.0F,
        1.0e12F);
    const float transfer = kfp_fast_log1p(distance);
    const float raw_position = transfer / 0.01F;
    const float cycle = static_cast<float>(lut_size);
    const float position = raw_position
        - std::floor(raw_position / cycle) * cycle;
    const int lower = std::clamp(static_cast<int>(position), 0, lut_size - 1);
    const float fraction = position - static_cast<float>(lower);
    const int upper = (lower + 1) % lut_size;
    const float inverse_fraction = 1.0F - fraction;
    float red = static_cast<float>(lut[lower * 3]) * inverse_fraction
        + static_cast<float>(lut[upper * 3]) * fraction;
    float green = static_cast<float>(lut[lower * 3 + 1]) * inverse_fraction
        + static_cast<float>(lut[upper * 3 + 1]) * fraction;
    float blue = static_cast<float>(lut[lower * 3 + 2]) * inverse_fraction
        + static_cast<float>(lut[upper * 3 + 2]) * fraction;

    const float horizontal = x > 0
        ? left - stencil_centre
            - (left_inside ? KFP_DEFAULT_SMOOTH_OFFSET : 0.0F)
        : stencil_centre - right;
    const float vertical = y == 0
        ? stencil_centre - down
        : up - stencil_centre
            - (up_inside ? KFP_DEFAULT_SMOOTH_OFFSET : 0.0F);
    const float projected = (horizontal + vertical) * 0.7071067811865475F
        * 50.0F * static_cast<float>(spatial_width) / 640.0F;
    const float strength = std::clamp(
        kfp_fast_atan_nonnegative(std::abs(projected))
            / 1.5707963267948966F * 0.2F,
        0.0F,
        1.0F);
    if (projected >= 0.0F) {
        red *= 1.0F - strength;
        green *= 1.0F - strength;
        blue *= 1.0F - strength;
    } else {
        red = red * (1.0F - strength) + 255.0F * strength;
        green = green * (1.0F - strength) + 255.0F * strength;
        blue = blue * (1.0F - strength) + 255.0F * strength;
    }
    destination[0] = kfp_dithered_colour_byte(red, dither_x, dither_y, 0);
    destination[1] = kfp_dithered_colour_byte(green, dither_x, dither_y, 1);
    destination[2] = kfp_dithered_colour_byte(blue, dither_x, dither_y, 2);
}

#if !defined(__AVX2__)
// The atlas compositor uses the same eight-pixel helper on both the AVX2 and
// scalar builds.  Keep a scalar implementation for portable release builds;
// otherwise the non-AVX2 compiler never sees the AVX2 definition above and a
// portable Linux/MinGW build fails even though the scalar pixel path exists.
inline void write_kfp_default_fast_block8(
    const float* field,
    int width,
    int height,
    int source_x,
    int source_y,
    int dither_x,
    int dither_y,
    int spatial_width,
    int max_iter,
    const std::uint8_t* lut,
    int lut_size,
    std::uint8_t* destination,
    double field_bias = 0.0
) {
    for (int lane = 0; lane < 8; ++lane) {
        write_kfp_default_fast_pixel(
            field,
            width,
            height,
            source_x + lane,
            source_y,
            dither_x + lane,
            dither_y,
            spatial_width,
            max_iter,
            lut,
            lut_size,
            destination + static_cast<size_t>(dither_x + lane) * 3U,
            field_bias);
    }
}
#endif

inline void colourise_kfp_default_fast_field(
    const float* field,
    int width,
    int height,
    int max_iter,
    const std::uint8_t* lut,
    int lut_size,
    std::uint8_t* output,
    int threads,
    double field_bias = 0.0
) {
#ifdef _OPENMP
    if (threads > 0) {
        omp_set_dynamic(0);
        omp_set_num_threads(threads);
    }
#pragma omp parallel for schedule(static)
#endif
    for (int y = 0; y < height; ++y) {
        int x = 0;
        write_kfp_default_fast_pixel(
            field,
            width,
            height,
            x,
            y,
            x,
            y,
            width,
            max_iter,
            lut,
            lut_size,
            output + static_cast<size_t>(y) * static_cast<size_t>(width) * 3U,
            field_bias);
        x = 1;
#if defined(__AVX2__)
        if (y > 0 && y + 1 < height && width >= 10) {
            const size_t row_offset = static_cast<size_t>(y)
                * static_cast<size_t>(width) * 3U;
            for (; x + 7 < width - 1; x += 8) {
                write_kfp_default_fast_block8(
                    field,
                    width,
                    height,
                    x,
                    y,
                    x,
                    y,
                    width,
                    max_iter,
                    lut,
                    lut_size,
                    output + row_offset,
                    field_bias);
            }
        }
#endif
        for (; x < width; ++x) {
            write_kfp_default_fast_pixel(
                field,
                width,
                height,
                x,
                y,
                x,
                y,
                width,
                max_iter,
                lut,
                lut_size,
                output + (static_cast<size_t>(y) * static_cast<size_t>(width)
                    + static_cast<size_t>(x)) * 3U,
                field_bias);
        }
    }
}

inline bool kfp_show_glitches(const FractalKfpOptions& options) noexcept;

inline void write_kfp_pixel(
    const float* field,
    int width,
    int height,
    int x,
    int y,
    int dither_x,
    int dither_y,
    int spatial_width,
    int max_iter,
    const FractalKfpOptions& options,
    const KfpPixelContext& pixel_context,
    const std::uint8_t* lut,
    int lut_size,
    double transfer_minimum,
    double transfer_maximum,
    const KfpSlopeDirection& slope_direction,
    std::uint8_t* destination,
    const FractalKfpPlanes* planes = nullptr,
    const double* orbit_colour_samples = nullptr,
    const double* scalar_colour_samples = nullptr
) {
    const bool orbit_planes = planes != nullptr
        && planes->orbit_iteration != nullptr;
    const bool has_iteration_planes = pixel_context.has_iteration_planes;
    const bool materialized_scalar = !has_iteration_planes
        && scalar_colour_samples != nullptr;
    // A complete Kalles metadata pass sources the centre sample from the
    // orbit/test planes. Avoid touching the scalar compatibility field in
    // that case; it is otherwise an unnecessary bandwidth/cache read for
    // every pixel and is not part of the colour decision.
    const float raw_value = orbit_planes
        ? 0.0F
        : field[static_cast<size_t>(y) * static_cast<size_t>(width)
            + static_cast<size_t>(x)];
    const size_t plane_index = static_cast<size_t>(y) * static_cast<size_t>(width)
        + static_cast<size_t>(x);
    if (planes != nullptr && planes->transition != nullptr
        && planes->transition[plane_index] < 0.0
        && !kfp_show_glitches(options)) {
        // SetColor returns without touching a hidden glitch pixel. Native
        // callers provide a fresh output buffer, so use deterministic black
        // rather than exposing stale or uninitialized memory.
        destination[0] = 0;
        destination[1] = 0;
        destination[2] = 0;
        return;
    }
    bool inside = false;
    double smooth_iter = 0.0;
    if (orbit_planes) {
        if (orbit_colour_samples != nullptr) {
            // The complete plane pass may have materialized this exact
            // Kalles sample once for the whole frame. Reusing it here also
            // covers the centre pixel; the stencil below reuses the same
            // array for neighbouring pixels.
            smooth_iter = orbit_colour_samples[plane_index];
            inside = smooth_iter > static_cast<double>(max_iter);
        } else {
            // The orbit path needs the same smoothed value for both the
            // interior test and SetColor. Share that sample so high-bailout
            // KFP frames do not pay for the log/pow smoothing stage twice per
            // pixel.
            const KfpOrbitSample sample = kfp_plane_orbit_sample(
                max_iter, options, planes, plane_index, &pixel_context);
            inside = sample.inside;
            smooth_iter = sample.colour;
        }
    } else if (materialized_scalar) {
        smooth_iter = scalar_colour_samples[plane_index];
        // The cached continuous value can cross max_iter because its Kalles
        // smoothing offset is added after the raw field was produced. That
        // does not make an escaped sample interior; membership is defined by
        // the original scalar cap, just as it is in kfp_plane_inside().
        inside = kfp_inside_value(
            raw_value, max_iter, options.field_bias);
    } else {
        inside = kfp_plane_inside(
            width, height, x, y, max_iter, options, planes, raw_value,
            &pixel_context);
        smooth_iter = has_iteration_planes
            ? kfp_plane_colour_sample(
                field, width, height, x, y, max_iter, options, planes,
                pixel_context.smooth_offset, &pixel_context)
            : [&]() {
                const double safe = std::isfinite(raw_value)
                    ? std::clamp(
                        static_cast<double>(raw_value) + options.field_bias,
                        0.0,
                        static_cast<double>(max_iter))
                    : 0.0;
                return std::max(0.0, safe + pixel_context.smooth_offset);
            }();
    }
    const bool texture_active = pixel_context.texture_active;
    if (inside && !texture_active) {
        destination[0] = static_cast<std::uint8_t>(options.interior_color[0]);
        destination[1] = static_cast<std::uint8_t>(options.interior_color[1]);
        destination[2] = static_cast<std::uint8_t>(options.interior_color[2]);
        return;
    }

    const double smooth_offset = pixel_context.smooth_offset;
    const double colour_iter = options.flat
        ? std::floor(smooth_iter)
        : smooth_iter;
    const double centre = smooth_iter;
    const auto reflected = [&](int offset_x, int offset_y) {
        if (has_iteration_planes) {
            const int sample_x = x + offset_x;
            const int sample_y = y + offset_y;
            if (sample_x >= 0 && sample_x < width
                && sample_y >= 0 && sample_y < height) {
                const size_t sample_index = static_cast<size_t>(sample_y)
                    * static_cast<size_t>(width)
                    + static_cast<size_t>(sample_x);
                if (orbit_planes && orbit_colour_samples != nullptr) {
                    return orbit_colour_samples[sample_index];
                }
                return kfp_plane_colour_sample(
                    field,
                    width,
                    height,
                    sample_x,
                    sample_y,
                    max_iter,
                    options,
                    planes,
                    smooth_offset,
                    &pixel_context);
            }
            const int opposite_x = x - offset_x;
            const int opposite_y = y - offset_y;
            if (opposite_x >= 0 && opposite_x < width
                && opposite_y >= 0 && opposite_y < height) {
                const size_t opposite_index = static_cast<size_t>(opposite_y)
                    * static_cast<size_t>(width)
                    + static_cast<size_t>(opposite_x);
                if (orbit_planes && orbit_colour_samples != nullptr) {
                    return 2.0 * centre - orbit_colour_samples[opposite_index];
                }
                return 2.0 * centre - kfp_plane_colour_sample(
                    field,
                    width,
                    height,
                    opposite_x,
                    opposite_y,
                    max_iter,
                    options,
                    planes,
                    smooth_offset,
                    &pixel_context);
            }
            return centre;
        }
        if (materialized_scalar) {
            const int sample_x = x + offset_x;
            const int sample_y = y + offset_y;
            if (sample_x >= 0 && sample_x < width
                && sample_y >= 0 && sample_y < height) {
                return scalar_colour_samples[
                    static_cast<size_t>(sample_y) * static_cast<size_t>(width)
                    + static_cast<size_t>(sample_x)];
            }
            const int opposite_x = x - offset_x;
            const int opposite_y = y - offset_y;
            if (opposite_x >= 0 && opposite_x < width
                && opposite_y >= 0 && opposite_y < height) {
                return 2.0 * centre - scalar_colour_samples[
                    static_cast<size_t>(opposite_y) * static_cast<size_t>(width)
                    + static_cast<size_t>(opposite_x)];
            }
            return centre;
        }
        return kfp_reflected_sample(
            field,
            width,
            height,
            x,
            y,
            offset_x,
            offset_y,
            max_iter,
            centre,
            smooth_offset,
            options.field_bias);
    };
    const bool needs_difference = pixel_context.needs_difference;
    const bool needs_texture = texture_active;
    const bool needs_slopes = pixel_context.needs_slopes;
    double left = centre;
    double right = centre;
    double up = centre;
    double down = centre;
    double top_left = centre;
    double top_right = centre;
    double bottom_left = centre;
    double bottom_right = centre;
    if (needs_difference || needs_slopes || needs_texture) {
        left = reflected(-1, 0);
        up = reflected(0, -1);
        if (needs_difference) {
            // Keep the stencil narrow for the common Kalles default
            // (Differences=3). Its operator uses only the upper-left,
            // left, and upper samples; loading all eight neighbours here
            // made every pixel pay for values that were immediately ignored.
            switch (options.differences) {
                case 0:
                    top_left = reflected(-1, -1);
                    bottom_left = reflected(-1, 1);
                    break;
                case 1:
                case 2:
                case 5:
                case 6:
                    right = reflected(1, 0);
                    down = reflected(0, 1);
                    top_left = reflected(-1, -1);
                    top_right = reflected(1, -1);
                    bottom_left = reflected(-1, 1);
                    bottom_right = reflected(1, 1);
                    break;
                case 3:
                    top_left = reflected(-1, -1);
                    break;
                case 4:
                    top_left = reflected(-1, -1);
                    break;
                default:
                    right = reflected(1, 0);
                    down = reflected(0, 1);
                    break;
            }
        }
        // The slope pass is one-sided: it uses the previous sample except at
        // the left/top edge, where Kalles falls forward.  Keep the narrow
        // difference stencil above, but load those two forward samples when
        // an edge slope actually needs them.
        if (needs_slopes || needs_texture) {
            if (x == 0) {
                right = reflected(1, 0);
            }
            if (y == 0) {
                down = reflected(0, 1);
            }
        }
    }

    double analytic_slope_x = 0.0;
    double analytic_slope_y = 0.0;
    double gradient = 0.0;
    if ((needs_difference || needs_slopes) && options.differences == 7
        && planes != nullptr
        && planes->de_x != nullptr && planes->de_y != nullptr) {
        // Kalles stores the analytic derivatives as float and its SetColor
        // path performs the reciprocal/norm calculation in float as well.
        // Keep that narrowing here so native output follows the reference
        // renderer instead of using a silently different double path.
        const float de_x = static_cast<float>(
            kfp_plane_value(planes->de_x, plane_index));
        const float de_y = static_cast<float>(
            kfp_plane_value(planes->de_y, plane_index));
        const float denominator = de_x * de_x + de_y * de_y;
        if (denominator > 0.0F && std::isfinite(denominator)) {
            const float gradient_float = 1.0F / std::sqrt(denominator);
            gradient = static_cast<double>(gradient_float);
            analytic_slope_x = static_cast<double>(de_x / denominator);
            analytic_slope_y = static_cast<double>(-de_y / denominator);
        }
    } else if (needs_difference) {
        gradient = kfp_difference_magnitude(
            options.differences,
            centre,
            left,
            right,
            up,
            down,
            top_left,
            top_right,
            bottom_left,
            bottom_right);
    }
    // Kalles clamps the *post-transfer* distance value to 1024.  Capping the
    // raw DE here changes ColorMethod 7 (DistanceLog), because its logarithm
    // is still sensitive to values such as 1e13 before that final clamp.
    // Preserve finite large distances and let each Kalles transfer branch do
    // its own final limiting.  A NaN is still treated as the zero-distance
    // fallback used by the rest of this ABI.
    double distance = std::isnan(gradient)
        ? 0.0
        : gradient * static_cast<double>(spatial_width) / 640.0;
    if (distance < 0.0) distance = 0.0;

    double transfer = colour_iter;
    switch (options.color_method) {
        case 1:
            transfer = std::sqrt(std::max(0.0, colour_iter));
            break;
        case 2:
            // Kalles uses pow(fmax(0, iter), 1/3), rather than cbrt().
            // They agree mathematically for positive values but can round
            // differently, which matters at a palette boundary.
            transfer = std::pow(std::max(0.0, colour_iter), 1.0 / 3.0);
            break;
        case 3:
            transfer = std::log(std::max(1.0, colour_iter));
            break;
        case 4:
            transfer = 1024.0 * (colour_iter - transfer_minimum)
                / std::max(transfer_maximum - transfer_minimum, 1.0e-12);
            break;
        case 5:
            transfer = std::min(distance, 1024.0);
            break;
        case 6: {
            // Kalles applies the distance-square-root transfer before the
            // DE-plus-standard threshold check.  Omitting the root makes
            // imported ColorMethod 6 profiles switch to iteration colours
            // far too early, especially on deep fields.
            const double distance_transfer = std::min(
                std::sqrt(std::max(0.0, distance)), 1024.0);
            transfer = distance_transfer > options.iter_div
                // Kalles restores nIter + 1 - offs here, even if Flat was
                // selected for the initial colour-method input.
                ? smooth_iter : distance_transfer;
            break;
        }
        case 7:
            transfer = std::log(std::max(1.0, distance + 1.0));
            break;
        case 8:
            transfer = std::sqrt(std::max(0.0, distance));
            break;
        case 9:
            // Keep the source expression instead of log1p(log1p()). Kalles'
            // CPU renderer evaluates log(1 + log(1 + iter)).
            transfer = std::log(
                1.0 + std::log(1.0 + std::max(0.0, colour_iter)));
            break;
        case 10:
            transfer = std::atan(colour_iter);
            break;
        case 11:
            // This is deliberately written as two square roots, matching
            // CFraktalSFT::SetColor rather than a generic pow(x, .25).
            transfer = std::sqrt(std::sqrt(std::max(0.0, colour_iter)));
            break;
        default:
            break;
    }
    // Keep positive infinity until the method-specific clamp below.  Kalles'
    // distance branches turn it into 1024; collapsing it to zero first would
    // produce a different palette position.  Negative/NaN values remain the
    // safe zero fallback.
    if (std::isnan(transfer) || transfer < 0.0) transfer = 0.0;
    if (options.color_method == 5 || options.color_method == 7
        || options.color_method == 8) {
        transfer = std::clamp(transfer, 0.0, 1024.0);
    }

    const double phase_shift = planes != nullptr && planes->phase != nullptr
        ? options.phase_color_strength / 100.0 * 1024.0
            * kfp_plane_phase(planes, plane_index)
        : 0.0;
    // CFraktalSFT::SetColor applies the serialized IterDiv once, after the
    // colour-method transfer and before the palette offset.  Keep this as a
    // separate stage: it is not part of the distance transfer itself.
    double palette_value = transfer;
    if (options.iter_div != 1.0) {
        palette_value /= options.iter_div;
    }
    // Kalles' srgb struct is float-valued in the complete SetColor path.
    // Keep palette, multi-colour, texture, and slope stages in that same
    // normalized representation; doing the whole stage in double and only
    // narrowing at the final byte conversion produces visible one-level
    // differences in steep KFP gradients.
    float red;
    float green;
    float blue;
    const auto palette_rgb = [&](double palette_input,
                                 float& output_red,
                                 float& output_green,
                                 float& output_blue) {
        double position = std::fmod(
            palette_input, static_cast<double>(lut_size));
        if (position < 0.0) position += static_cast<double>(lut_size);
        const double lower_position = std::floor(position);
        const int lower = std::clamp(
            static_cast<int>(lower_position), 0, lut_size - 1);
        if (options.smooth) {
            double fraction = position - lower_position;
            if (options.inverse_transition) fraction = 1.0 - fraction;
            const int upper = (lower + 1) % lut_size;
            // This preserves the source order: the byte interpolation is
            // formed in double because `offs` is double, then assigned to
            // srgb after division by the float literal 255.0f.
            const double inverse_fraction = 1.0 - fraction;
            output_red = static_cast<float>(
                (static_cast<double>(lut[lower * 3]) * inverse_fraction
                    + static_cast<double>(lut[upper * 3]) * fraction)
                / 255.0F);
            output_green = static_cast<float>(
                (static_cast<double>(lut[lower * 3 + 1]) * inverse_fraction
                    + static_cast<double>(lut[upper * 3 + 1]) * fraction)
                / 255.0F);
            output_blue = static_cast<float>(
                (static_cast<double>(lut[lower * 3 + 2]) * inverse_fraction
                    + static_cast<double>(lut[upper * 3 + 2]) * fraction)
                / 255.0F);
        } else {
            output_red = static_cast<float>(lut[lower * 3]) / 255.0F;
            output_green = static_cast<float>(lut[lower * 3 + 1]) / 255.0F;
            output_blue = static_cast<float>(lut[lower * 3 + 2]) / 255.0F;
        }
    };

    const double base_palette_input = palette_value + options.color_offset;
    if (options.multi_color) {
        const double wave_input = options.smooth
            ? base_palette_input : std::floor(base_palette_input);
        double hue_sum = 0.0;
        double saturation_sum = 0.0;
        double value_sum = 0.0;
        int hue_count = 0;
        int saturation_count = 0;
        int value_count = 0;
        for (std::uint32_t index = 0; index < options.multi_color_count; ++index) {
            const double period = options.multi_color_period[index];
            const double wave = period < 0.0
                ? -period / 100.0
                : 0.5 + 0.5 * std::sin(
                    3.14159265358979323846 * wave_input / period);
            switch (options.multi_color_type[index]) {
                case 0:
                    hue_sum += wave;
                    ++hue_count;
                    break;
                case 1:
                    saturation_sum += wave;
                    ++saturation_count;
                    break;
                default:
                    value_sum += wave;
                    ++value_count;
                    break;
            }
        }
        const float hue = static_cast<float>(
            hue_count > 0 ? hue_sum / hue_count : 0.0);
        // Kalles initializes nS and nB to zero.  A palette containing only a
        // hue wave therefore produces black until it also supplies a
        // saturation/brightness wave; defaulting either component to one is
        // a tempting HSV convenience, but is not Kalles' renderer.
        const float saturation = static_cast<float>(
            saturation_count > 0 ? saturation_sum / saturation_count : 0.0);
        const float value = static_cast<float>(
            value_count > 0 ? value_sum / value_count : 0.0);
        float multi_red;
        float multi_green;
        float multi_blue;
        kfp_hsv_to_rgb(
            hue, saturation, value, multi_red, multi_green, multi_blue);
        if (options.blend_multi_color) {
            // In Kalles' SetColor the phase shift belongs to the ordinary
            // palette half of a blended MultiColor result.  It does not
            // alter the wave input, and a non-blended MultiColor result is
            // completely phase-independent.
            palette_rgb(
                base_palette_input + phase_shift,
                red,
                green,
                blue);
            red = (red + multi_red) * 0.5F;
            green = (green + multi_green) * 0.5F;
            blue = (blue + multi_blue) * 0.5F;
        } else {
            red = multi_red;
            green = multi_green;
            blue = multi_blue;
        }
    }
    else {
        palette_rgb(base_palette_input + phase_shift, red, green, blue);
    }

    if (texture_active && inside) {
        // Kalles seeds textured interior pixels with the configured interior
        // colour, then lets the image merge replace it.  Without this reset,
        // an interior pixel would inherit an arbitrary palette sample before
        // SetTexture runs.
        red = static_cast<float>(options.interior_color[0]) / 255.0F;
        green = static_cast<float>(options.interior_color[1]) / 255.0F;
        blue = static_cast<float>(options.interior_color[2]) / 255.0F;
    }

    if (texture_active) {
        // This is CFraktalSFT::SetTexture's integer CPU path.  The image is
        // supplied top-down by the caller, while Kalles stores its DIB
        // bottom-up, so the final row lookup is inverted below.
        const double texture_power = options.texture_power;
        const double texture_ratio = options.texture_ratio;
        const double texture_dx = x > 0
            ? left - centre
            : (x + 1 < width ? centre - right : 0.0);
        const double texture_dy = y > 0
            ? up - centre
            : (y + 1 < height ? centre - down : 0.0);
        const auto texture_offset = [&](double difference) {
            const double adjusted = 1.0 + difference;
            const double powered = std::pow(adjusted, texture_power);
            const bool forward = powered > 1.0;
            const double selected = forward ? powered : 1.0 / powered;
            // Kalles deliberately lets +/-infinity reach atan(); atan(inf)
            // is the finite limiting warp produced by its SetTexture code.
            // Only NaN is invalid here. Rejecting infinity made high-power
            // textures silently sample the unwarped centre instead.
            const double mapped = !std::isnan(selected)
                ? (std::atan(selected) - 3.14159265358979323846 / 4.0)
                    / (3.14159265358979323846 / 4.0)
                    * (texture_ratio / 100.0)
                : 0.0;
            return std::isfinite(mapped)
                ? (forward ? texture_power * mapped : -texture_power * mapped)
                : 0.0;
        };
        const int image_offset = std::isfinite(texture_power)
            ? static_cast<int>(std::clamp(
                texture_power / 64.0,
                static_cast<double>(std::numeric_limits<int>::min()),
                static_cast<double>(std::numeric_limits<int>::max())))
            : 0;
        const int sample_x = kfp_texture_index(
            static_cast<double>(x + image_offset) + texture_offset(texture_dx),
            planes->texture_width);
        const int sample_y = kfp_texture_index(
            static_cast<double>(y + image_offset) + texture_offset(texture_dy),
            planes->texture_height);
        const int row = planes->texture_height - 1 - sample_y;
        const std::uint8_t* texture = planes->texture_rgb
            + static_cast<size_t>(row) * static_cast<size_t>(planes->texture_stride)
            + static_cast<size_t>(sample_x) * 3U;
        const double merge = options.texture_merge;
        red = static_cast<float>(
            static_cast<double>(red) * (1.0 - merge)
            + merge * static_cast<double>(texture[0]) / 255.0);
        green = static_cast<float>(
            static_cast<double>(green) * (1.0 - merge)
            + merge * static_cast<double>(texture[1]) / 255.0);
        blue = static_cast<float>(
            static_cast<double>(blue) * (1.0 - merge)
            + merge * static_cast<double>(texture[2]) / 255.0);
    }

    if (needs_slopes) {
        // Kalles' default CPU colour path prefers the previous horizontal
        // neighbour and falls forward only at the left edge. Keep this
        // orientation in the native path so imported .kfp relief is aligned
        // with the reference renderer users normally see.
        const double horizontal = options.differences == 7
            && planes != nullptr && planes->de_x != nullptr
            && planes->de_y != nullptr
            ? analytic_slope_x
            : (x > 0 ? left - centre : centre - right);
        const double vertical = options.differences == 7
            && planes != nullptr && planes->de_x != nullptr
            && planes->de_y != nullptr
            ? analytic_slope_y
            : (y == 0 ? centre - down : up - centre);
        const double projected = (
            (horizontal * slope_direction.cosine)
            + (vertical * slope_direction.sine))
                * options.slope_power * static_cast<double>(spatial_width) / 640.0;
        // Kalles does not clamp the post-atan SlopeRatio multiplier.  The UI
        // normally keeps it in the visible range, but imported profiles are
        // allowed to use stronger values and the source renderer preserves
        // those values.
        const double strength = std::atan(std::abs(projected))
            / (3.14159265358979323846 / 2.0)
            * options.slope_ratio / 100.0;
        if (projected >= 0.0) {
            red = static_cast<float>(static_cast<double>(red) * (1.0 - strength));
            green = static_cast<float>(static_cast<double>(green) * (1.0 - strength));
            blue = static_cast<float>(static_cast<double>(blue) * (1.0 - strength));
        } else {
            red = static_cast<float>(static_cast<double>(red) * (1.0 - strength) + strength);
            green = static_cast<float>(static_cast<double>(green) * (1.0 - strength) + strength);
            blue = static_cast<float>(static_cast<double>(blue) * (1.0 - strength) + strength);
        }
    }

    destination[0] = kfp_dithered_srgb_byte(red, dither_x, dither_y, 0);
    destination[1] = kfp_dithered_srgb_byte(green, dither_x, dither_y, 1);
    destination[2] = kfp_dithered_srgb_byte(blue, dither_x, dither_y, 2);
}

struct KfpTransferBounds {
    double minimum = 0.0;
    double maximum = 1.0;
};

KfpTransferBounds kfp_transfer_bounds(
    const float* field,
    int width,
    int height,
    int max_iter,
    double field_bias = 0.0,
    double smooth_offset = 0.0
) {
    double minimum = std::numeric_limits<double>::infinity();
    double maximum = -std::numeric_limits<double>::infinity();
#ifdef _OPENMP
#pragma omp parallel for reduction(min:minimum) reduction(max:maximum) schedule(static)
#endif
    for (int pixel = 0; pixel < width * height; ++pixel) {
        const float raw_value = field[pixel];
        if (kfp_inside_value(raw_value, max_iter, field_bias)) continue;
        const double safe = std::isfinite(raw_value)
            ? std::clamp(
                static_cast<double>(raw_value) + field_bias,
                0.0,
                static_cast<double>(max_iter))
            : 0.0;
        // The compatibility field stores the already-smoothed scalar value,
        // so mirror the Python fallback's materialized nPixels range here.
        // A complete Kalles plane render uses kfp_transfer_bounds_planes and
        // gets the true pre-smoothing nIter0 range instead. Kalles also
        // excludes the final escaped band (nPixels >= max_iter - 1) from
        // GetIterations' stretched range.
        const double value = std::floor(safe + smooth_offset);
        if (value >= static_cast<double>(max_iter - 1)) continue;
        minimum = std::min(minimum, value);
        maximum = std::max(maximum, value);
    }
    if (!std::isfinite(minimum) || !std::isfinite(maximum)) {
        return {};
    }
    return {minimum, maximum};
}

KfpTransferBounds kfp_transfer_bounds_planes(
    const float* field,
    int width,
    int height,
    int max_iter,
    const FractalKfpOptions& options,
    const FractalKfpPlanes* planes,
    const double* orbit_colour_samples = nullptr
) {
    const KfpPixelContext pixel_context = kfp_pixel_context(
        options, kfp_smooth_offset(options), planes);
    double minimum = std::numeric_limits<double>::infinity();
    double maximum = -std::numeric_limits<double>::infinity();
#ifdef _OPENMP
#pragma omp parallel for reduction(min:minimum) reduction(max:maximum) schedule(static)
#endif
    for (int pixel = 0; pixel < width * height; ++pixel) {
        const int y = pixel / width;
        const int x = pixel - y * width;
        const float scalar_value = field[pixel];
        if (planes != nullptr && planes->orbit_iteration != nullptr) {
            const size_t plane_index = static_cast<size_t>(pixel);
            const double sample_colour = orbit_colour_samples != nullptr
                ? orbit_colour_samples[plane_index]
                : kfp_plane_orbit_sample(
                    max_iter, options, planes, plane_index,
                    &pixel_context).colour;
            // Materialized orbit samples encode an interior value as
            // max_iter + 1, exactly as KfpOrbitSample does.
            if (sample_colour > static_cast<double>(max_iter)) continue;
            // For an escaped orbit, the stored integer used by Kalles'
            // GetIterations is the floor of the same continuous sample that
            // SetColor consumes. Reuse it rather than running smoothing a
            // second time through kfp_plane_stored_iteration.
            const double value = std::floor(std::max(0.0, sample_colour));
            if (value >= static_cast<double>(max_iter - 1)) continue;
            minimum = std::min(minimum, value);
            maximum = std::max(maximum, value);
            continue;
        }
        if (kfp_plane_inside(
                width, height, x, y, max_iter, options, planes, scalar_value,
                &pixel_context)) {
            continue;
        }
        const double value = kfp_plane_stored_iteration(
            field, width, height, x, y, max_iter, options, planes,
            &pixel_context);
        // CFraktalSFT::GetIterations skips nPixels >= maxIter - 1 when it
        // computes the bounds for ColorMethod 4.  Keep this test on the
        // post-OutputIterationData integer value, not on the smoothed field.
        if (value >= static_cast<double>(max_iter - 1)) continue;
        minimum = std::min(minimum, value);
        maximum = std::max(maximum, value);
    }
    if (!std::isfinite(minimum) || !std::isfinite(maximum)) {
        return {};
    }
    return {minimum, maximum};
}

bool valid_kfp_options(const FractalKfpOptions* options, int lut_size) noexcept {
    const std::uint32_t legacy_size = static_cast<std::uint32_t>(
        offsetof(FractalKfpOptions, show_glitches));
    if (!options || options->struct_size < legacy_size
        || options->version != FRACTAL_KFP_OPTIONS_VERSION
        || lut_size < 2 || lut_size > 65536
        || !std::isfinite(options->iter_div) || options->iter_div <= 0.0
        || !std::isfinite(options->color_offset)
        || !std::isfinite(options->ratio)
        || options->color_method < 0 || options->color_method > 11
        || options->smooth_method < 0 || options->smooth_method > 2
        || options->smooth < 0 || options->smooth > 1
        || options->flat < 0 || options->flat > 1
        || options->inverse_transition < 0 || options->inverse_transition > 1
        || !std::isfinite(options->phase_color_strength)
        || options->multi_color < 0 || options->multi_color > 1
        || options->blend_multi_color < 0 || options->blend_multi_color > 1
        || options->multi_color_count > FRACTAL_KFP_MAX_MULTI_COLORS
        || !std::isfinite(options->power) || options->power <= 0.0
        || options->slopes < 0 || options->slopes > 1
        || !std::isfinite(options->slope_power) || options->slope_power < 0.0
        || !std::isfinite(options->slope_ratio) || options->slope_ratio < 0.0
        || !std::isfinite(options->slope_angle)
        || options->differences < 0 || options->differences > 7
        || !std::isfinite(options->field_bias)
        || options->field_bias < 0.0
        || options->bailout_radius_preset < 0
        || options->bailout_radius_preset > 3
        || !std::isfinite(options->bailout_radius_custom)
        || options->bailout_radius_custom <= 0.0
        || options->bailout_norm_preset < 0
        || options->bailout_norm_preset > 3
        || !std::isfinite(options->bailout_norm_custom)
        || options->bailout_norm_custom <= 0.0
        || options->texture_enabled < 0 || options->texture_enabled > 1
        || !std::isfinite(options->texture_merge)
        || !std::isfinite(options->texture_power)
        || !std::isfinite(options->texture_ratio)
        || options->texture_resize < 0 || options->texture_resize > 1
        || options->use_opengl < 0 || options->use_opengl > 1
        || options->use_srgb < 0 || options->use_srgb > 1) {
        return false;
    }
    if (options->struct_size >= legacy_size + sizeof(options->show_glitches)
        && (options->show_glitches < 0 || options->show_glitches > 1)) {
        return false;
    }
    for (std::uint32_t index = 0; index < options->multi_color_count; ++index) {
        if (!std::isfinite(options->multi_color_period[index])
            || options->multi_color_period[index] == 0.0
            || options->multi_color_type[index] < 0
            || options->multi_color_type[index] > 2) {
            return false;
        }
    }
    return options->interior_color[0] >= 0 && options->interior_color[0] <= 255
        && options->interior_color[1] >= 0 && options->interior_color[1] <= 255
        && options->interior_color[2] >= 0 && options->interior_color[2] <= 255;
}

inline bool kfp_show_glitches(const FractalKfpOptions& options) noexcept {
    const std::uint32_t field_offset = static_cast<std::uint32_t>(
        offsetof(FractalKfpOptions, show_glitches));
    if (options.struct_size < field_offset + sizeof(options.show_glitches)) {
        // The field was appended without changing the version so older
        // callers remain valid. Kalles' default is to show glitch pixels.
        return true;
    }
    return options.show_glitches != 0;
}

bool valid_kfp_planes(
    const FractalKfpPlanes* planes,
    int width,
    int height
) noexcept {
    if (planes == nullptr) return true;
    if (planes->struct_size < sizeof(FractalKfpPlanes)
        || planes->version != FRACTAL_KFP_PLANES_VERSION) {
        return false;
    }
    if (planes->texture_rgb == nullptr) {
        return planes->texture_width == 0
            && planes->texture_height == 0
            && planes->texture_stride == 0;
    }
    if (planes->texture_width <= 0 || planes->texture_height <= 0
        || static_cast<std::int64_t>(planes->texture_stride)
            < static_cast<std::int64_t>(planes->texture_width) * 3) {
        return false;
    }
    const std::uint64_t texture_bytes = static_cast<std::uint64_t>(
        planes->texture_stride) * static_cast<std::uint64_t>(planes->texture_height);
    return texture_bytes <= static_cast<std::uint64_t>(MAX_NATIVE_PIXELS) * 3U
        && valid_pixel_dimensions(width, height);
}

bool valid_render_planes(
    const FractalRenderPlanes* planes,
    int width,
    int height
) noexcept {
    if (!planes || !valid_pixel_dimensions(width, height)) return false;
    return planes->struct_size >= sizeof(FractalRenderPlanes)
        && planes->version == FRACTAL_RENDER_PLANES_VERSION
        && planes->orbit_iteration != nullptr
        && planes->phase != nullptr
        && planes->de_x != nullptr
        && planes->de_y != nullptr
        && planes->test1 != nullptr
        && planes->test2 != nullptr;
}

KfpSlopeDirection kfp_slope_direction(const FractalKfpOptions& options) noexcept {
    if (!options.slopes) return {};
    const double angle = options.slope_angle
        * 3.14159265358979323846 / 180.0;
    return {std::cos(angle), std::sin(angle)};
}

long double parse_zoom(const char* text) {
    if (!valid_c_string(text)) throw std::runtime_error("native zoom text is too long or null");
    long double zoom = 0.0L;
    if (!parse_classic_long_double(text, zoom)
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

inline const FloatExp& escape_radius_squared_float_exp(int mode) {
    static const FloatExp classic = FloatExp::from_parts(4.0, 0);
    static const FloatExp kalles_high = FloatExp::from_parts(1.0e8, 0);
    return mode == ESCAPE_RADIUS_MODE_KALLES_HIGH ? kalles_high : classic;
}

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
        if (magnitude >= 2.0) {
            // A BLA polynomial can legitimately produce a value larger than
            // two binary units in one component.  The old one-bit fast path
            // left that value unnormalised, which made a later aligned sum
            // compare exponent fields instead of actual magnitudes and could
            // discard a neighbouring-pixel perturbation at extreme zoom.
            int shift = 0;
            (void)std::frexp(magnitude, &shift);
            real = std::ldexp(real, -shift);
            imag = std::ldexp(imag, -shift);
            exponent = saturating_exponent(
                static_cast<long long>(exponent) + static_cast<long long>(shift));
            return;
        }
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

inline ScaledComplex sc_conjugate(const ScaledComplex& value) {
    return {value.real, -value.imag, value.exponent};
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

inline bool sc_outside_escape(
    const ScaledNorm& norm,
    int escape_radius_mode = ESCAPE_RADIUS_MODE_CLASSIC
) {
    const FloatExp& threshold = escape_radius_squared_float_exp(escape_radius_mode);
    return sc_compare_norm(
        norm,
        ScaledNorm{threshold.mantissa, threshold.exponent}) > 0;
}

inline double sc_to_double(const ScaledComplex& value) {
    return std::ldexp(value.real, value.exponent);
}

inline double sc_imag_to_double(const ScaledComplex& value) {
    return std::ldexp(value.imag, value.exponent);
}

inline double scaled_norm_to_double(const ScaledNorm& norm) {
    if (norm.mantissa == 0.0) return 0.0;
    if (!std::isfinite(norm.mantissa)) {
        return std::numeric_limits<double>::max();
    }
    const double value = std::ldexp(norm.mantissa, norm.exponent);
    return std::isfinite(value)
        ? std::max(0.0, value)
        : std::numeric_limits<double>::max();
}

inline ScaledNorm scaled_norm_from_double(double value) {
    if (!(value > 0.0) || !std::isfinite(value)) {
        return value > 0.0
            ? ScaledNorm{std::numeric_limits<double>::max(), 0}
            : ScaledNorm{};
    }
    int exponent = 0;
    const double mantissa = std::frexp(value, &exponent);
    return {mantissa, exponent};
}

/*
 * Convert the scaled Mandelbrot derivative into the complex DE value that
 * Kalles stores in m_nDEx/m_nDEy.  Kalles computes
 *
 *   de = |z| log|z| / (normalize(z) * transpose(J * s))
 *
 * for the identity transform, where J is the real 2x2 Jacobian and s is the
 * pixel spacing.  For z², Kalles stores that Jacobian as
 *
 *   [ dr  -di ]
 *   [ di   dr ]
 *
 * and evaluates ``normalise(z) * transpose(J)`` as the row-vector form
 * ``z * conjugate(dz)``.  The conjugate is easy to lose when this special
 * case is reduced to complex arithmetic; doing so leaves the ordinary
 * palette unchanged but rotates the analytic slope field into the spoke and
 * ring artefacts seen in relief renders.  Keep the exact Kalles orientation
 * here while retaining scaled arithmetic for deep orbits.
 */
inline bool render_plane_de(
    const ScaledComplex& total,
    const ScaledComplex& derivative,
    const FloatExp& pixel_spacing,
    double& output_real,
    double& output_imag
) noexcept {
    const double component_magnitude = std::hypot(total.real, total.imag);
    if (!(component_magnitude > 0.0) || !std::isfinite(component_magnitude)) {
        return false;
    }
    const double log_magnitude =
        std::log(component_magnitude)
        + static_cast<double>(total.exponent) * LOG_TWO;
    if (!(log_magnitude > 0.0) || !std::isfinite(log_magnitude)) {
        return false;
    }
    const double unit_real = total.real / component_magnitude;
    const double unit_imag = total.imag / component_magnitude;
    const ScaledComplex unit = ScaledComplex::from_float_exp(
        FloatExp::from_parts(unit_real, 0),
        FloatExp::from_parts(unit_imag, 0));
    const ScaledComplex spacing = ScaledComplex::from_float_exp(
        pixel_spacing,
        FloatExp{});
    const ScaledComplex denominator = sc_mul(
        unit,
        sc_mul(sc_conjugate(derivative), spacing));
    const double denominator_squared =
        denominator.real * denominator.real
        + denominator.imag * denominator.imag;
    if (!(denominator_squared > 0.0)
        || !std::isfinite(denominator_squared)) {
        return false;
    }
    // numerator = component_magnitude * 2^exponent * log(|z|).
    // Division by the normalized denominator is represented with one shared
    // exponent, then converted to the ABI's double pair at the end.
    ScaledComplex result{
        component_magnitude * log_magnitude * denominator.real
            / denominator_squared,
        -component_magnitude * log_magnitude * denominator.imag
            / denominator_squared,
        saturating_exponent(
            static_cast<long long>(total.exponent)
            - static_cast<long long>(denominator.exponent)),
    };
    result.normalize();
    output_real = sc_to_double(result);
    output_imag = sc_imag_to_double(result);
    return std::isfinite(output_real) && std::isfinite(output_imag);
}

// The scaled-arithmetic helpers are defined below the legacy DE helper. Keep
// declarations here so the Kalles-compatible matrix code can sit beside the
// existing complex implementation without changing the hot-loop helper
// ordering.
inline FloatExp fe_neg(const FloatExp& value);
inline FloatExp fe_abs(const FloatExp& value);
inline FloatExp fe_add(const FloatExp& a, const FloatExp& b);
inline FloatExp fe_mul(const FloatExp& a, const FloatExp& b);
inline FloatExp fe_mul(const FloatExp& a, double b);
inline FloatExp fe_div(const FloatExp& a, const FloatExp& b);
inline FloatExp fe_sqr(const FloatExp& value);

/*
 * Kalles stores the full real 2x2 Jacobian for non-holomorphic formulas.
 * Keeping only a complex derivative is sufficient for z^2/Julia, but it
 * silently turns Tricorn and Burning Ship analytic distance colouring into a
 * Mandelbrot-shaped approximation.  This form mirrors compute_de() in
 * Kalles' fraktal_sft.h: J is the screen-space Jacobian and the palette gets
 * |z| log|z| / (normalise(z) * transpose(J * pixel_spacing)).
 */
struct AlternateJacobian {
    // Row-major: [dRe/dx, dRe/dy, dIm/dx, dIm/dy].
    FloatExp xa{};
    FloatExp xb{};
    FloatExp ya{};
    FloatExp yb{};
};

inline AlternateJacobian alternate_identity_jacobian() {
    return {
        FloatExp::from_parts(1.0, 0),
        FloatExp{},
        FloatExp{},
        FloatExp::from_parts(1.0, 0),
    };
}

inline AlternateJacobian alternate_holomorphic_jacobian(
    const ScaledComplex& derivative
) {
    const FloatExp real = FloatExp::from_parts(
        derivative.real,
        derivative.exponent);
    const FloatExp imag = FloatExp::from_parts(
        derivative.imag,
        derivative.exponent);
    return {
        real,
        fe_neg(imag),
        imag,
        real,
    };
}

inline bool render_plane_de_matrix(
    const ScaledComplex& total,
    const AlternateJacobian& jacobian,
    const FloatExp& pixel_spacing,
    double& output_real,
    double& output_imag
) noexcept {
    const double component_magnitude = std::hypot(total.real, total.imag);
    if (!(component_magnitude > 0.0) || !std::isfinite(component_magnitude)) {
        return false;
    }
    const double log_magnitude =
        std::log(component_magnitude)
        + static_cast<double>(total.exponent) * LOG_TWO;
    if (!(log_magnitude > 0.0) || !std::isfinite(log_magnitude)) {
        return false;
    }
    const double unit_real = total.real / component_magnitude;
    const double unit_imag = total.imag / component_magnitude;
    const FloatExp scaled_xa = fe_mul(jacobian.xa, pixel_spacing);
    const FloatExp scaled_xb = fe_mul(jacobian.xb, pixel_spacing);
    const FloatExp scaled_ya = fe_mul(jacobian.ya, pixel_spacing);
    const FloatExp scaled_yb = fe_mul(jacobian.yb, pixel_spacing);
    // transpose(J) * normalise(z), written with Kalles' row-major derivative
    // names.  For a holomorphic derivative this reduces to zhat * dz/dc,
    // exactly the complex implementation above.
    const FloatExp denominator_real = fe_add(
        fe_mul(scaled_xa, unit_real),
        fe_mul(scaled_ya, unit_imag));
    const FloatExp denominator_imag = fe_add(
        fe_mul(scaled_xb, unit_real),
        fe_mul(scaled_yb, unit_imag));
    const FloatExp denominator_squared = fe_add(
        fe_sqr(denominator_real),
        fe_sqr(denominator_imag));
    if (denominator_squared.zero() || !denominator_squared.finite()) {
        return false;
    }
    const FloatExp numerator = FloatExp::from_parts(
        component_magnitude * log_magnitude,
        total.exponent);
    const FloatExp result_real = fe_div(
        fe_mul(numerator, denominator_real),
        denominator_squared);
    const FloatExp result_imag = fe_div(
        fe_neg(fe_mul(numerator, denominator_imag)),
        denominator_squared);
    output_real = std::ldexp(result_real.mantissa, result_real.exponent);
    output_imag = std::ldexp(result_imag.mantissa, result_imag.exponent);
    return std::isfinite(output_real) && std::isfinite(output_imag);
}

inline AlternateJacobian alternate_jacobian_step(
    int formula,
    const ScaledComplex& total,
    const AlternateJacobian& previous,
    bool parameter_plane
) {
    const FloatExp real = FloatExp::from_parts(total.real, total.exponent);
    const FloatExp imag = FloatExp::from_parts(total.imag, total.exponent);
    FloatExp m00{};
    FloatExp m01{};
    FloatExp m10{};
    FloatExp m11{};
    if (formula == FRACTAL_FORMULA_TRICORN) {
        m00 = fe_mul(real, 2.0);
        m01 = fe_mul(imag, -2.0);
        m10 = fe_mul(imag, -2.0);
        m11 = fe_mul(real, -2.0);
    } else if (formula == FRACTAL_FORMULA_BURNING_SHIP) {
        const FloatExp absolute_real = fe_abs(real);
        const FloatExp absolute_imag = fe_abs(imag);
        const double real_sign = real.mantissa < 0.0 ? -1.0 : 1.0;
        const double imag_sign = imag.mantissa < 0.0 ? -1.0 : 1.0;
        m00 = fe_mul(absolute_real, 2.0 * real_sign);
        m01 = fe_mul(absolute_imag, -2.0 * imag_sign);
        m10 = fe_mul(absolute_imag, 2.0 * real_sign);
        m11 = fe_mul(absolute_real, 2.0 * imag_sign);
    } else {
        m00 = fe_mul(real, 2.0);
        m01 = fe_mul(imag, -2.0);
        m10 = fe_mul(imag, 2.0);
        m11 = fe_mul(real, 2.0);
    }
    AlternateJacobian result{
        fe_add(
            fe_add(fe_mul(m00, previous.xa), fe_mul(m01, previous.ya)),
            parameter_plane ? FloatExp::from_parts(1.0, 0) : FloatExp{}),
        fe_add(
            fe_add(fe_mul(m00, previous.xb), fe_mul(m01, previous.yb)),
            FloatExp{}),
        fe_add(
            fe_add(fe_mul(m10, previous.xa), fe_mul(m11, previous.ya)),
            FloatExp{}),
        fe_add(
            fe_add(fe_mul(m10, previous.xb), fe_mul(m11, previous.yb)),
            parameter_plane ? FloatExp::from_parts(1.0, 0) : FloatExp{}),
    };
    return result;
}

inline AlternateJacobian alternate_linear_bla_jacobian(
    const std::array<FloatExp, 8>& coefficients,
    const AlternateJacobian& previous,
    bool parameter_plane
) {
    const auto& c = coefficients;
    return {
        fe_add(
            fe_add(fe_mul(c[0], previous.xa), fe_mul(c[1], previous.ya)),
            parameter_plane ? c[4] : FloatExp{}),
        fe_add(
            fe_add(fe_mul(c[0], previous.xb), fe_mul(c[1], previous.yb)),
            parameter_plane ? c[5] : FloatExp{}),
        fe_add(
            fe_add(fe_mul(c[2], previous.xa), fe_mul(c[3], previous.ya)),
            parameter_plane ? c[6] : FloatExp{}),
        fe_add(
            fe_add(fe_mul(c[2], previous.xb), fe_mul(c[3], previous.yb)),
            parameter_plane ? c[7] : FloatExp{}),
    };
}

inline double render_plane_phase(const ScaledComplex& value) {
    const double real = sc_to_double(value);
    const double imag = sc_imag_to_double(value);
    if (!std::isfinite(real) || !std::isfinite(imag)) return 0.0;
    double phase = std::atan2(imag, real) / TWO_PI;
    phase -= std::floor(phase);
    return std::isfinite(phase) ? phase : 0.0;
}

inline void clear_render_planes_pixel(
    FractalRenderPlanes* planes,
    size_t index,
    int max_iter
) noexcept {
    if (planes == nullptr) return;
    planes->orbit_iteration[index] = max_iter;
    planes->phase[index] = 0.0;
    planes->de_x[index] = 0.0;
    planes->de_y[index] = 0.0;
    planes->test1[index] = 0.0;
    planes->test2[index] = 0.0;
}

// The deep Mandelbrot loops keep the current orbit sample in `iteration`
// form: z_1 is iteration 1, z_2 is iteration 2, and so on.  Kalles' `antal`
// is the number of samples completed before the one being tested, so the
// metadata for z_n is n - 1.  Keep the conversion at the call sites instead
// of changing the generic plane writer, which is also used by the zero-based
// direct renderer and the Julia alternate path.
inline int kalles_mandelbrot_raw_iteration(int iteration) noexcept {
    return std::max(0, iteration - 1);
}

inline void store_render_planes_escape(
    FractalRenderPlanes* planes,
    size_t index,
    int iteration,
    const ScaledComplex& total,
    const ScaledNorm& test1,
    const ScaledNorm& test2,
    const ScaledComplex* derivative = nullptr,
    const FloatExp* pixel_spacing = nullptr
) noexcept {
    if (planes == nullptr) return;
    planes->orbit_iteration[index] = iteration;
    planes->phase[index] = render_plane_phase(total);
    planes->de_x[index] = 0.0;
    planes->de_y[index] = 0.0;
    if (derivative != nullptr && pixel_spacing != nullptr) {
        double de_x = 0.0;
        double de_y = 0.0;
        if (render_plane_de(
                total,
                *derivative,
                *pixel_spacing,
                de_x,
                de_y)) {
            planes->de_x[index] = de_x;
            planes->de_y[index] = de_y;
        }
    }
    planes->test1[index] = scaled_norm_to_double(test1);
    planes->test2[index] = scaled_norm_to_double(test2);
}

inline void store_render_planes_escape_jacobian(
    FractalRenderPlanes* planes,
    size_t index,
    int iteration,
    const ScaledComplex& total,
    const ScaledNorm& test1,
    const ScaledNorm& test2,
    const AlternateJacobian& jacobian,
    const FloatExp& pixel_spacing
) noexcept {
    if (planes == nullptr) return;
    planes->orbit_iteration[index] = iteration;
    planes->phase[index] = render_plane_phase(total);
    planes->de_x[index] = 0.0;
    planes->de_y[index] = 0.0;
    double de_x = 0.0;
    double de_y = 0.0;
    if (render_plane_de_matrix(
            total,
            jacobian,
            pixel_spacing,
            de_x,
            de_y)) {
        planes->de_x[index] = de_x;
        planes->de_y[index] = de_y;
    }
    planes->test1[index] = scaled_norm_to_double(test1);
    planes->test2[index] = scaled_norm_to_double(test2);
}

inline long double smooth_escape_value(int iteration, const ScaledNorm& norm) {
    if (norm.mantissa == 0.0) return static_cast<long double>(iteration);
    const long double log_magnitude = 0.5L * (
        std::log(static_cast<long double>(norm.mantissa))
        + static_cast<long double>(norm.exponent) * LOG_TWO);
    if (!(log_magnitude > 0.0L) || !std::isfinite(log_magnitude)) {
        return static_cast<long double>(iteration);
    }
    return static_cast<long double>(iteration)
        - std::log(log_magnitude) / LOG_TWO;
}

inline float smooth_escape_scaled(
    int iteration,
    const ScaledNorm& norm,
    double output_bias = 0.0
) {
    return encode_render_value(smooth_escape_value(iteration, norm), output_bias);
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
    const FloatExpComplex& delta,
    int escape_radius_mode = ESCAPE_RADIUS_MODE_CLASSIC
) {
    const FloatExp cross = fe_mul(
        fe_add(
            fe_mul(reference.real, delta.real),
            fe_mul(reference.imag, delta.imag)),
        2.0);
    const FloatExp reference_margin = fe_sub(
        fec_norm_squared(reference),
        escape_radius_squared_float_exp(escape_radius_mode));
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
    const ScaledComplex& delta,
    int escape_radius_mode = ESCAPE_RADIUS_MODE_CLASSIC
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
    return fec_escape_margin_with_delta(
        reference_parts, delta_parts, escape_radius_mode);
}

inline FloatExp sc_escape_margin_with_reference_margin(
    const FloatExp& reference_margin,
    const ScaledComplex& reference,
    const ScaledComplex& delta
) {
    const FloatExp cross = fe_mul(
        fe_add(
            fe_mul(sc_component_as_float_exp(reference, false),
                   sc_component_as_float_exp(delta, false)),
            fe_mul(sc_component_as_float_exp(reference, true),
                   sc_component_as_float_exp(delta, true))),
        2.0);
    const ScaledNorm delta_norm = sc_norm_squared(delta);
    return fe_add(
        fe_add(reference_margin, cross),
        FloatExp{delta_norm.mantissa, delta_norm.exponent});
}

inline ScaledNorm sc_norm_squared_with_delta(
    const ScaledComplex& reference,
    const ScaledComplex& delta,
    int escape_radius_mode = ESCAPE_RADIUS_MODE_CLASSIC
) {
    const FloatExp norm = fe_add(
        escape_radius_squared_float_exp(escape_radius_mode),
        sc_escape_margin_with_delta(reference, delta, escape_radius_mode));
    return {norm.mantissa, norm.exponent};
}

inline bool sc_outside_escape_with_delta(
    const ScaledComplex& reference,
    const ScaledComplex& delta,
    int escape_radius_mode = ESCAPE_RADIUS_MODE_CLASSIC
) {
    const FloatExp margin = sc_escape_margin_with_delta(
        reference, delta, escape_radius_mode);
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

// Alternate formulas are still quadratic maps, but Tricorn is
// anti-holomorphic and Burning Ship is piecewise.  A complex A/B BLA cannot
// represent either derivative without silently turning one of those maps
// into Mandelbrot.  Keep the same compact block idea as the Mandelbrot path,
// but store the derivative and the constant-parameter response as two real
// 2x2 matrices:
//
//     d' = M d + P dc
//
// The nonlinear d^2 term is bounded by radius_squared and is replayed exactly
// whenever a block approaches the escape boundary.  This keeps the hot path
// native and SIMD-friendly at deep zooms while retaining formula-specific
// perturbation arithmetic at seams and absolute-value axes.
struct AlternateLinearBlaStep {
    // M row-major followed by P row-major.
    std::array<FloatExp, 8> coefficients{};
    FloatExp radius_squared;
    int length = 1;
};

struct AlternateLinearBlaLevels {
    std::vector<std::vector<AlternateLinearBlaStep>> levels;
    FloatExp input_radius;
    FloatExp input_radius_squared;
    int start_index = 0;

    static int highest_level_for_length(int max_length) noexcept {
        return max_length > 0
            ? 31 - __builtin_clz(static_cast<unsigned int>(max_length))
            : 0;
    }

    const AlternateLinearBlaStep* lookup(
        int start,
        const FloatExp& delta_norm_squared,
        const FloatExp& parameter_norm_squared,
        int max_length
    ) const noexcept {
        if (levels.empty() || start < start_index || max_length <= 0
            || !delta_norm_squared.finite() || !parameter_norm_squared.finite()) {
            return nullptr;
        }
        const int offset = start - start_index;
        const int base_count = static_cast<int>(levels[0].size());
        if (offset < 0 || offset >= base_count) return nullptr;
        int highest_level = highest_level_for_length(max_length);
        highest_level = std::min(
            highest_level,
            static_cast<int>(levels.size()) - 1);
        for (int level = highest_level; level >= 0; --level) {
            const int span_mask = (1 << level) - 1;
            if ((offset & span_mask) != 0) continue;
            const int index = offset >> level;
            if (index >= static_cast<int>(levels[level].size())) continue;
            const AlternateLinearBlaStep& candidate =
                levels[level][static_cast<size_t>(index)];
            if (candidate.length > 1
                && candidate.radius_squared.finite()
                && fe_compare(delta_norm_squared, candidate.radius_squared) < 0
                && fe_compare(parameter_norm_squared, input_radius_squared) <= 0) {
                return &candidate;
            }
        }
        return nullptr;
    }
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

/*
 * The image series is a polynomial in dc with no constant term.  Keep its
 * derivative alongside the value so the Kalles analytic-DE plane can start at
 * the same iteration as the scalar perturbation path.  Evaluating only the
 * scalar series and restarting DE at one is visibly wrong on profiles using
 * Differences=Analytic.
 */
inline ScaledComplex evaluate_image_series_with_derivative(
    const ImageSeries& series,
    const ScaledComplex& dc,
    ScaledComplex& derivative
) {
    if (!series.enabled || series.coefficients.size() <= 1) {
        derivative = {};
        return {};
    }
    ScaledComplex value = series.coefficients.back();
    ScaledComplex polynomial_derivative{};
    for (size_t index = series.coefficients.size() - 1; index > 1; --index) {
        polynomial_derivative = sc_add(
            sc_mul(polynomial_derivative, dc),
            value);
        value = sc_add(
            sc_mul(value, dc),
            series.coefficients[index - 1]);
    }
    // value is q(dc), while the series result is q(dc)*dc.
    derivative = sc_add(
        sc_mul(polynomial_derivative, dc),
        value);
    return sc_mul(value, dc);
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

inline ScaledComplex apply_bla_series_derivative(
    const FastBlaStep& step,
    const ScaledComplex& delta,
    const ScaledComplex& parameter,
    const ScaledComplex& derivative,
    int order
) {
    const auto& coefficient = step.coefficients;
    if (order <= 1) {
        // P(d,c) = A*d + B*c.
        return sc_add(
            sc_mul(coefficient[0], derivative),
            coefficient[1]);
    }

    // Differentiate the bivariate BLA polynomial with respect to the pixel
    // parameter c, using dd/dc=derivative.  Keeping the partials separate is
    // important: using only A*derivative loses the direct parameter term and
    // produces the wrong Kalles distance estimator after every BLA jump.
    ScaledComplex partial_delta = coefficient[0];
    ScaledComplex partial_parameter = coefficient[1];
    if (order >= 2) {
        partial_delta = sc_add(
            partial_delta,
            sc_add(
                sc_double(sc_mul(coefficient[2], delta)),
                sc_mul(coefficient[3], parameter)));
        partial_parameter = sc_add(
            partial_parameter,
            sc_add(
                sc_mul(coefficient[3], delta),
                sc_double(sc_mul(coefficient[4], parameter))));
    }
    if (order >= 3) {
        const ScaledComplex delta_squared = sc_mul(delta, delta);
        const ScaledComplex parameter_squared = sc_mul(parameter, parameter);
        partial_delta = sc_add(
            partial_delta,
            sc_add(
                sc_mul(
                    sc_add(
                        sc_double(sc_mul(coefficient[5], delta)),
                        sc_mul(coefficient[5], delta)),
                    delta),
                sc_add(
                    sc_double(sc_mul(coefficient[6], sc_mul(delta, parameter))),
                    sc_mul(coefficient[7], parameter_squared))));
        partial_parameter = sc_add(
            partial_parameter,
            sc_add(
                sc_mul(coefficient[6], delta_squared),
                sc_add(
                    sc_double(sc_mul(coefficient[7], sc_mul(delta, parameter))),
                    sc_mul(
                        sc_add(
                            sc_double(sc_mul(coefficient[8], parameter)),
                            sc_mul(coefficient[8], parameter)),
                        parameter))));
    }
    return sc_add(
        sc_mul(partial_delta, derivative),
        partial_parameter);
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
    // Keep the MPFR-computed signed escape margin alongside the compact orbit.
    // Reconstructing |Z + delta|^2 from a rounded reference can lose the sign
    // of a perturbation sitting on |Z| = 2, which used to create rectangular
    // fills in deep alternate-formula tiles.
    std::vector<FloatExp> escape_margin;
};

struct ReferenceContext {
    // Assigned when the immutable context enters the opaque-handle registry.
    // Device-side caches use this generation rather than a host pointer so a
    // later allocation can never accidentally reuse an old GPU orbit.
    std::uint64_t generation = 0;
    std::vector<FloatExpComplex> fast_orbit;
    std::shared_ptr<const ReferenceOrbitData> orbit;
    BlaLevels bla;
    AlternateLinearBlaLevels alternate_bla;
    ImageSeries image_series;
    int requested_max_iter = 0;
    int requested_series_order = 8;
    std::uint64_t reference_build_ns = 0;
    std::uint64_t series_build_ns = 0;
    std::uint64_t bla_build_ns = 0;
    long double x_center = 0.0L;
    long double y_center = 0.0L;
    // Keep the original decimal centres for the rare MPFR pixel rebase used
    // after a compact perturbation glitch.  The long-double copies above are
    // intentionally sufficient for ordinary metadata, but are not precise
    // enough to reconstruct a 10^100-scale pixel coordinate.
    std::string x_center_text;
    std::string y_center_text;
    int formula = FRACTAL_FORMULA_MANDELBROT;
    double julia_real = 0.0;
    double julia_imag = 0.0;
    ScaledComplex parameter;
    int escape_radius_mode = ESCAPE_RADIUS_MODE_CLASSIC;
    int coordinate_mode = COORDINATE_MODE_PROJECT;
#ifdef FRACTAL_HAVE_MPFR
    mpfr_prec_t precision_bits = 0;
#endif
};

#ifdef FRACTAL_HAVE_MPFR
FloatExp parse_zoom_float_exp(const char* text, mpfr_prec_t precision_bits);
#endif

#ifdef FRACTAL_HAVE_OPENCL
// Do not pass the C++ ScaledComplex object directly across the OpenCL ABI:
// its tail padding is implementation-defined.  The explicit four-field
// transfer record matches the kernel's ``sc`` exactly on 64-bit hosts.
struct OpenClScaledComplex {
    double real;
    double imag;
    std::int32_t exponent;
    std::int32_t padding;
};
static_assert(sizeof(OpenClScaledComplex) == 24,
              "OpenCL scaled-complex transfer layout changed");

// OpenCL's ``fe`` is a double followed by a 32-bit exponent and has 16-byte
// struct size/alignment on the supported devices.  Keep an explicit padded
// host record so the reference-norm cache does not depend on C++ ABI padding.
struct OpenClFloatExp {
    double mantissa;
    std::int32_t exponent;
    std::int32_t padding;
};
static_assert(sizeof(OpenClFloatExp) == 16,
              "OpenCL float-exponent transfer layout changed");

struct OpenClLinearBlaStep {
    OpenClScaledComplex a;
    OpenClScaledComplex b;
    double radius_mantissa;
    std::int32_t radius_exponent;
    std::int32_t length;
};
static_assert(sizeof(OpenClLinearBlaStep) == 64,
              "OpenCL linear-BLA transfer layout changed");

// Large deep ordinary fields are the one workload where an fp64 consumer GPU
// can spend most of its time on arithmetic that does not need 53 mantissa
// bits.  Keep the binary exponent in full precision, but use a float
// mantissa for the device-side recurrence.  This is deliberately a separate
// ABI from the strict path: KFP orbit planes and small correctness probes
// continue to use the double-mantissa records above.
struct OpenClMixedScaledComplex {
    float real;
    float imag;
    std::int32_t exponent;
    std::int32_t padding;
};
static_assert(sizeof(OpenClMixedScaledComplex) == 16,
              "OpenCL mixed scaled-complex transfer layout changed");

struct OpenClMixedFloatExp {
    float mantissa;
    std::int32_t exponent;
};
static_assert(sizeof(OpenClMixedFloatExp) == 8,
              "OpenCL mixed float-exponent transfer layout changed");

struct OpenClMixedLinearBlaStep {
    OpenClMixedScaledComplex a;
    OpenClMixedScaledComplex b;
    float radius_mantissa;
    std::int32_t radius_exponent;
    std::int32_t length;
};
static_assert(sizeof(OpenClMixedLinearBlaStep) == 44,
              "OpenCL mixed linear-BLA transfer layout changed");

void render_deep_perturbation_opencl(
    float* output,
    int width,
    int height,
    const char* zoom_text,
    const ReferenceContext& context,
    int max_iter,
    const FractalRenderOptions& options,
    FractalRenderPlanes* planes = nullptr,
    const std::vector<ScaledComplex>* point_offsets = nullptr
) {
    if (!context.orbit || context.orbit->scaled.size() < 2) {
        throw std::runtime_error("OpenCL deep renderer has no complete reference orbit");
    }
    initialise_opencl();
    if (!opencl_available()) {
        throw std::runtime_error(
            opencl_runtime && !opencl_runtime->error.empty()
                ? opencl_runtime->error
                : "OpenCL deep renderer is unavailable");
    }
    const FloatExp zoom = parse_zoom_float_exp(zoom_text, context.precision_bits);
    const FloatExp view_height = fe_mul(
        fe_div(FloatExp::from_parts(1.0, 0), zoom),
        viewport_height_factor(options.coordinate_mode));
    const FloatExp view_width = fe_mul(
        view_height, static_cast<double>(width) / static_cast<double>(height));
    const FloatExp pixel_spacing = fe_div(
        view_height, FloatExp::from_parts(static_cast<double>(height), 0));
    const FloatExp& bailout = escape_radius_squared_float_exp(options.escape_radius_mode);
    if (!view_width.finite() || !view_height.finite() || !bailout.finite()) {
        throw std::runtime_error("OpenCL deep renderer received a non-finite viewport");
    }

    const size_t reference_count_size = context.orbit->scaled.size();
    if (reference_count_size > static_cast<size_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error("OpenCL deep reference is too large");
    }
    // The flattened hierarchy is immutable for a prepared reference.  Work
    // out only its shape up front; materialize and upload it only when the
    // device cache does not already hold this reference generation.
    const std::vector<std::vector<LinearBlaStep>>* linear_bla_levels = nullptr;
    size_t linear_bla_level_count = 0;
    size_t linear_bla_step_count = 0;
    if (planes == nullptr && options.disable_bla == 0) {
        // Both hierarchies are built with the same conservative 2^-38
        // linearisation bound.  The regular table is valid for the normal
        // viewport radius and removes the e12--e80 exact-iteration cliff;
        // the tighter table is needed once the reference itself is reused at
        // ultra-deep zooms.  The device still checks each candidate's radius
        // and alignment before applying it.
        const auto& levels = view_height.exponent < -260
            ? context.bla.deep_linear_levels
            : context.bla.linear_levels;
        linear_bla_level_count = std::min<size_t>(levels.size(), 30);
        for (size_t level = 0; level < linear_bla_level_count; ++level) {
            linear_bla_step_count += levels[level].size();
        }
        if (linear_bla_step_count > 0) linear_bla_levels = &levels;
    }
    const size_t count = static_cast<size_t>(width) * static_cast<size_t>(height);
    if (point_offsets != nullptr && point_offsets->size() != count) {
        throw std::runtime_error("OpenCL point offsets do not match the output size");
    }
    if (point_offsets != nullptr && planes != nullptr) {
        throw std::runtime_error("OpenCL point rendering cannot emit orbit planes");
    }
    OpenClRuntime& runtime = *opencl_runtime;
    std::lock_guard<std::mutex> lock(runtime.mutex);
    cl_int status = CL_SUCCESS;
    const size_t output_bytes = count * sizeof(float);
    if (!runtime.deep_output || runtime.deep_output_capacity < output_bytes) {
        cl_mem replacement = clCreateBuffer(
            runtime.context, CL_MEM_WRITE_ONLY, output_bytes, nullptr, &status);
        if (status != CL_SUCCESS || !replacement) {
            throw std::runtime_error(opencl_error_text(status));
        }
        if (runtime.deep_output) clReleaseMemObject(runtime.deep_output);
        runtime.deep_output = replacement;
        runtime.deep_output_capacity = output_bytes;
    }
    cl_mem device_output = runtime.deep_output;
    if (!device_output) throw std::runtime_error(opencl_error_text(status));
    const size_t reference_bytes = reference_count_size * sizeof(OpenClScaledComplex);
    const bool cached_reference_matches = runtime.deep_reference != nullptr
        && runtime.deep_reference_generation == context.generation
        && runtime.deep_reference_capacity >= reference_bytes;
    if (!cached_reference_matches) {
        if (!runtime.deep_reference || runtime.deep_reference_capacity < reference_bytes) {
            cl_mem replacement = clCreateBuffer(
                runtime.context, CL_MEM_READ_ONLY, reference_bytes, nullptr, &status);
            if (status != CL_SUCCESS || !replacement) {
                throw std::runtime_error(opencl_error_text(status));
            }
            if (runtime.deep_reference) clReleaseMemObject(runtime.deep_reference);
            runtime.deep_reference = replacement;
            runtime.deep_reference_capacity = reference_bytes;
        }
        std::vector<OpenClScaledComplex> reference(reference_count_size);
        for (size_t index = 0; index < reference.size(); ++index) {
            const ScaledComplex& value = context.orbit->scaled[index];
            reference[index] = {value.real, value.imag, value.exponent, 0};
        }
        status = clEnqueueWriteBuffer(runtime.queue, runtime.deep_reference, CL_TRUE, 0,
            reference_bytes, reference.data(), 0, nullptr, nullptr);
        if (status != CL_SUCCESS) throw std::runtime_error(opencl_error_text(status));
        runtime.deep_reference_generation = context.generation;
    }
    cl_mem device_reference = runtime.deep_reference;
    const size_t reference_norm_bytes = reference_count_size * sizeof(OpenClFloatExp);
    const bool cached_reference_norms_match = runtime.deep_reference_norms != nullptr
        && runtime.deep_reference_norms_generation == context.generation
        && runtime.deep_reference_norms_capacity >= reference_norm_bytes;
    if (!cached_reference_norms_match) {
        if (!runtime.deep_reference_norms
            || runtime.deep_reference_norms_capacity < reference_norm_bytes) {
            cl_mem replacement = clCreateBuffer(
                runtime.context, CL_MEM_READ_ONLY, reference_norm_bytes, nullptr, &status);
            if (status != CL_SUCCESS || !replacement) {
                throw std::runtime_error(opencl_error_text(status));
            }
            if (runtime.deep_reference_norms) {
                clReleaseMemObject(runtime.deep_reference_norms);
            }
            runtime.deep_reference_norms = replacement;
            runtime.deep_reference_norms_capacity = reference_norm_bytes;
        }
        std::vector<OpenClFloatExp> reference_norms(reference_count_size);
        for (size_t index = 0; index < reference_norms.size(); ++index) {
            const ScaledNorm norm = sc_norm_squared(context.orbit->scaled[index]);
            reference_norms[index] = {
                norm.mantissa, static_cast<std::int32_t>(norm.exponent), 0};
        }
        status = clEnqueueWriteBuffer(
            runtime.queue, runtime.deep_reference_norms, CL_TRUE, 0,
            reference_norm_bytes, reference_norms.data(), 0, nullptr, nullptr);
        if (status != CL_SUCCESS) throw std::runtime_error(opencl_error_text(status));
        runtime.deep_reference_norms_generation = context.generation;
    }
    cl_mem device_reference_norms = runtime.deep_reference_norms;
    cl_mem device_point_offsets = device_reference;
    if (point_offsets != nullptr) {
        const size_t point_bytes = count * sizeof(OpenClScaledComplex);
        if (!runtime.deep_point_offsets
            || runtime.deep_point_offsets_capacity < point_bytes) {
            cl_mem replacement = clCreateBuffer(
                runtime.context, CL_MEM_READ_ONLY, point_bytes, nullptr, &status);
            if (status != CL_SUCCESS || !replacement) {
                throw std::runtime_error(opencl_error_text(status));
            }
            if (runtime.deep_point_offsets) clReleaseMemObject(runtime.deep_point_offsets);
            runtime.deep_point_offsets = replacement;
            runtime.deep_point_offsets_capacity = point_bytes;
        }
        std::vector<OpenClScaledComplex> packed_points(count);
        for (size_t index = 0; index < count; ++index) {
            const ScaledComplex& value = (*point_offsets)[index];
            packed_points[index] = {value.real, value.imag, value.exponent, 0};
        }
        status = clEnqueueWriteBuffer(
            runtime.queue, runtime.deep_point_offsets, CL_TRUE, 0,
            point_bytes, packed_points.data(), 0, nullptr, nullptr);
        if (status != CL_SUCCESS) throw std::runtime_error(opencl_error_text(status));
        device_point_offsets = runtime.deep_point_offsets;
    }
    const auto cleanup = [&] {
    };
    cl_mem device_bla = device_reference;
    cl_mem device_bla_offsets = device_reference;
    cl_mem device_bla_counts = device_reference;
    const int use_linear_bla = linear_bla_levels ? 1 : 0;
    if (use_linear_bla) {
        const size_t bla_bytes = linear_bla_step_count * sizeof(OpenClLinearBlaStep);
        const size_t level_bytes = linear_bla_level_count * sizeof(std::int32_t);
        const bool cached_bla_matches = runtime.deep_bla != nullptr
            && runtime.deep_bla_generation == context.generation
            && runtime.deep_bla_capacity >= bla_bytes
            && runtime.deep_bla_levels_capacity >= level_bytes
            && runtime.deep_bla_level_count == static_cast<int>(linear_bla_level_count);
        if (cached_bla_matches) {
            device_bla = runtime.deep_bla;
            device_bla_offsets = runtime.deep_bla_offsets;
            device_bla_counts = runtime.deep_bla_counts;
        } else {
            std::vector<OpenClLinearBlaStep> linear_bla;
            std::vector<std::int32_t> linear_bla_offsets;
            std::vector<std::int32_t> linear_bla_counts;
            linear_bla.reserve(linear_bla_step_count);
            linear_bla_offsets.reserve(linear_bla_level_count);
            linear_bla_counts.reserve(linear_bla_level_count);
            for (size_t level = 0; level < linear_bla_level_count; ++level) {
                const auto& source_level = (*linear_bla_levels)[level];
                linear_bla_offsets.push_back(static_cast<std::int32_t>(linear_bla.size()));
                linear_bla_counts.push_back(static_cast<std::int32_t>(source_level.size()));
                for (const LinearBlaStep& step : source_level) {
                    linear_bla.push_back({
                        {step.A.real, step.A.imag, step.A.exponent, 0},
                        {step.B.real, step.B.imag, step.B.exponent, 0},
                        step.radius_squared.mantissa, step.radius_squared.exponent,
                        step.length,
                    });
                }
            }
            const bool reuse_bla_buffers = runtime.deep_bla != nullptr
                && runtime.deep_bla_offsets != nullptr
                && runtime.deep_bla_counts != nullptr
                && runtime.deep_bla_capacity >= bla_bytes
                && runtime.deep_bla_levels_capacity >= level_bytes;
            cl_mem new_bla = reuse_bla_buffers ? runtime.deep_bla : clCreateBuffer(
                runtime.context, CL_MEM_READ_ONLY, bla_bytes, nullptr, &status);
            cl_mem new_offsets = reuse_bla_buffers ? runtime.deep_bla_offsets : nullptr;
            cl_mem new_counts = reuse_bla_buffers ? runtime.deep_bla_counts : nullptr;
            if (!reuse_bla_buffers && status == CL_SUCCESS && new_bla) {
                new_offsets = clCreateBuffer(runtime.context, CL_MEM_READ_ONLY,
                    level_bytes, nullptr, &status);
            }
            if (!reuse_bla_buffers && status == CL_SUCCESS && new_offsets) {
                new_counts = clCreateBuffer(runtime.context, CL_MEM_READ_ONLY,
                    level_bytes, nullptr, &status);
            }
            if (status == CL_SUCCESS && new_counts) {
                status = clEnqueueWriteBuffer(runtime.queue, new_bla, CL_TRUE, 0,
                    bla_bytes, linear_bla.data(), 0, nullptr, nullptr);
            }
            if (status == CL_SUCCESS && new_counts) {
                status = clEnqueueWriteBuffer(runtime.queue, new_offsets, CL_TRUE, 0,
                    level_bytes, linear_bla_offsets.data(), 0, nullptr, nullptr);
            }
            if (status == CL_SUCCESS && new_counts) {
                status = clEnqueueWriteBuffer(runtime.queue, new_counts, CL_TRUE, 0,
                    level_bytes, linear_bla_counts.data(), 0, nullptr, nullptr);
            }
            if (status != CL_SUCCESS || !new_bla || !new_offsets || !new_counts) {
                if (!reuse_bla_buffers) {
                    if (new_counts) clReleaseMemObject(new_counts);
                    if (new_offsets) clReleaseMemObject(new_offsets);
                    if (new_bla) clReleaseMemObject(new_bla);
                }
                throw std::runtime_error(opencl_error_text(status));
            }
            if (!reuse_bla_buffers) {
                if (runtime.deep_bla_counts) clReleaseMemObject(runtime.deep_bla_counts);
                if (runtime.deep_bla_offsets) clReleaseMemObject(runtime.deep_bla_offsets);
                if (runtime.deep_bla) clReleaseMemObject(runtime.deep_bla);
            }
            runtime.deep_bla = new_bla;
            runtime.deep_bla_offsets = new_offsets;
            runtime.deep_bla_counts = new_counts;
            runtime.deep_bla_capacity = bla_bytes;
            runtime.deep_bla_levels_capacity = level_bytes;
            runtime.deep_bla_generation = context.generation;
            runtime.deep_bla_level_count = static_cast<int>(linear_bla_level_count);
            device_bla = new_bla;
            device_bla_offsets = new_offsets;
            device_bla_counts = new_counts;
        }
    }
    // Use the mixed path automatically for genuinely large scalar fields on
    // a physical GPU.  Small ABI/correctness probes and orbit-plane renders
    // retain the strict double contract.  FRACTAL_OPENCL_MIXED_PRECISION=0
    // is an escape hatch for visual A/B comparisons; setting it to 1 forces
    // the same production choice explicitly.
    const char* mixed_precision_text = std::getenv(
        "FRACTAL_OPENCL_MIXED_PRECISION");
    const char* strict_precision_text = std::getenv(
        "FRACTAL_OPENCL_STRICT_DOUBLE");
    const bool mixed_precision_allowed = strict_precision_text == nullptr
        || std::strcmp(strict_precision_text, "0") == 0;
    const bool mixed_precision_forced = mixed_precision_text != nullptr
        && std::strcmp(mixed_precision_text, "0") != 0;
    const bool use_mixed_precision = runtime.deep_mixed_kernel != nullptr
        && runtime.device_is_gpu
        && mixed_precision_allowed
        && (mixed_precision_forced || mixed_precision_text == nullptr)
        && planes == nullptr
        && point_offsets == nullptr
        && count >= static_cast<size_t>(1920) * static_cast<size_t>(1080);
    cl_mem mixed_device_reference = device_reference;
    cl_mem mixed_device_reference_norms = device_reference_norms;
    cl_mem mixed_device_bla = device_output;
    cl_mem mixed_device_bla_offsets = device_output;
    cl_mem mixed_device_bla_counts = device_output;
    if (use_mixed_precision) {
        const size_t mixed_reference_bytes = reference_count_size
            * sizeof(OpenClMixedScaledComplex);
        if (!runtime.mixed_reference
            || runtime.mixed_reference_capacity < mixed_reference_bytes) {
            cl_mem replacement = clCreateBuffer(
                runtime.context, CL_MEM_READ_ONLY, mixed_reference_bytes,
                nullptr, &status);
            if (status != CL_SUCCESS || !replacement) {
                throw std::runtime_error(opencl_error_text(status));
            }
            if (runtime.mixed_reference) clReleaseMemObject(runtime.mixed_reference);
            runtime.mixed_reference = replacement;
            runtime.mixed_reference_capacity = mixed_reference_bytes;
        }
        if (runtime.mixed_reference_generation != context.generation) {
            std::vector<OpenClMixedScaledComplex> reference(reference_count_size);
            for (size_t index = 0; index < reference.size(); ++index) {
                const ScaledComplex& value = context.orbit->scaled[index];
                reference[index] = {
                    static_cast<float>(value.real),
                    static_cast<float>(value.imag),
                    value.exponent, 0};
            }
            status = clEnqueueWriteBuffer(
                runtime.queue, runtime.mixed_reference, CL_TRUE, 0,
                mixed_reference_bytes, reference.data(), 0, nullptr, nullptr);
            if (status != CL_SUCCESS) {
                throw std::runtime_error(opencl_error_text(status));
            }
            runtime.mixed_reference_generation = context.generation;
        }
        mixed_device_reference = runtime.mixed_reference;

        const size_t mixed_norm_bytes = reference_count_size
            * sizeof(OpenClMixedFloatExp);
        if (!runtime.mixed_reference_norms
            || runtime.mixed_reference_norms_capacity < mixed_norm_bytes) {
            cl_mem replacement = clCreateBuffer(
                runtime.context, CL_MEM_READ_ONLY, mixed_norm_bytes,
                nullptr, &status);
            if (status != CL_SUCCESS || !replacement) {
                throw std::runtime_error(opencl_error_text(status));
            }
            if (runtime.mixed_reference_norms) {
                clReleaseMemObject(runtime.mixed_reference_norms);
            }
            runtime.mixed_reference_norms = replacement;
            runtime.mixed_reference_norms_capacity = mixed_norm_bytes;
        }
        if (runtime.mixed_reference_norms_generation != context.generation) {
            // The mixed kernels use this table only for the Kalles glitch
            // threshold. Upload that threshold directly instead of making
            // every device iteration rescale the immutable reference norm.
            std::vector<OpenClMixedFloatExp> reference_thresholds(
                reference_count_size);
            for (size_t index = 0; index < reference_thresholds.size(); ++index) {
                const ScaledNorm norm = sc_norm_squared(context.orbit->scaled[index]);
                float mantissa = static_cast<float>(norm.mantissa)
                    * 0.8388608f;
                std::int32_t exponent = static_cast<std::int32_t>(norm.exponent);
                if (mantissa != 0.0f) {
                    if (mantissa < 0.5f) {
                        mantissa *= 2.0f;
                        exponent -= 24;
                    } else {
                        exponent -= 23;
                    }
                } else {
                    exponent = 0;
                }
                reference_thresholds[index] = {mantissa, exponent};
            }
            status = clEnqueueWriteBuffer(
                runtime.queue, runtime.mixed_reference_norms, CL_TRUE, 0,
                mixed_norm_bytes, reference_thresholds.data(),
                0, nullptr, nullptr);
            if (status != CL_SUCCESS) {
                throw std::runtime_error(opencl_error_text(status));
            }
            runtime.mixed_reference_norms_generation = context.generation;
        }
        mixed_device_reference_norms = runtime.mixed_reference_norms;

        if (linear_bla_levels != nullptr) {
            const size_t mixed_bla_bytes = linear_bla_step_count
                * sizeof(OpenClMixedLinearBlaStep);
            const size_t mixed_level_bytes = linear_bla_level_count
                * sizeof(std::int32_t);
            const bool cached_mixed_bla = runtime.mixed_bla != nullptr
                && runtime.mixed_bla_generation == context.generation
                && runtime.mixed_bla_capacity >= mixed_bla_bytes
                && runtime.mixed_bla_levels_capacity >= mixed_level_bytes
                && runtime.mixed_bla_level_count
                    == static_cast<int>(linear_bla_level_count);
            if (cached_mixed_bla) {
                mixed_device_bla = runtime.mixed_bla;
                mixed_device_bla_offsets = runtime.mixed_bla_offsets;
                mixed_device_bla_counts = runtime.mixed_bla_counts;
            } else {
                std::vector<OpenClMixedLinearBlaStep> mixed_bla;
                std::vector<std::int32_t> mixed_offsets;
                std::vector<std::int32_t> mixed_counts;
                mixed_bla.reserve(linear_bla_step_count);
                mixed_offsets.reserve(linear_bla_level_count);
                mixed_counts.reserve(linear_bla_level_count);
                for (size_t level = 0; level < linear_bla_level_count; ++level) {
                    const auto& source_level = (*linear_bla_levels)[level];
                    mixed_offsets.push_back(static_cast<std::int32_t>(mixed_bla.size()));
                    mixed_counts.push_back(static_cast<std::int32_t>(source_level.size()));
                    for (const LinearBlaStep& step : source_level) {
                        mixed_bla.push_back({
                            {static_cast<float>(step.A.real),
                             static_cast<float>(step.A.imag), step.A.exponent, 0},
                            {static_cast<float>(step.B.real),
                             static_cast<float>(step.B.imag), step.B.exponent, 0},
                            static_cast<float>(step.radius_squared.mantissa),
                            step.radius_squared.exponent,
                            step.length});
                    }
                }
                const bool reuse_mixed_bla = runtime.mixed_bla != nullptr
                    && runtime.mixed_bla_offsets != nullptr
                    && runtime.mixed_bla_counts != nullptr
                    && runtime.mixed_bla_capacity >= mixed_bla_bytes
                    && runtime.mixed_bla_levels_capacity >= mixed_level_bytes;
                cl_mem new_bla = reuse_mixed_bla ? runtime.mixed_bla
                    : clCreateBuffer(runtime.context, CL_MEM_READ_ONLY,
                                     mixed_bla_bytes, nullptr, &status);
                cl_mem new_offsets = reuse_mixed_bla
                    ? runtime.mixed_bla_offsets : nullptr;
                cl_mem new_counts = reuse_mixed_bla
                    ? runtime.mixed_bla_counts : nullptr;
                if (!reuse_mixed_bla && status == CL_SUCCESS && new_bla) {
                    new_offsets = clCreateBuffer(
                        runtime.context, CL_MEM_READ_ONLY, mixed_level_bytes,
                        nullptr, &status);
                }
                if (!reuse_mixed_bla && status == CL_SUCCESS && new_offsets) {
                    new_counts = clCreateBuffer(
                        runtime.context, CL_MEM_READ_ONLY, mixed_level_bytes,
                        nullptr, &status);
                }
                if (status == CL_SUCCESS && new_counts) {
                    status = clEnqueueWriteBuffer(
                        runtime.queue, new_bla, CL_TRUE, 0, mixed_bla_bytes,
                        mixed_bla.data(), 0, nullptr, nullptr);
                }
                if (status == CL_SUCCESS && new_counts) {
                    status = clEnqueueWriteBuffer(
                        runtime.queue, new_offsets, CL_TRUE, 0, mixed_level_bytes,
                        mixed_offsets.data(), 0, nullptr, nullptr);
                }
                if (status == CL_SUCCESS && new_counts) {
                    status = clEnqueueWriteBuffer(
                        runtime.queue, new_counts, CL_TRUE, 0, mixed_level_bytes,
                        mixed_counts.data(), 0, nullptr, nullptr);
                }
                if (status != CL_SUCCESS || !new_bla || !new_offsets || !new_counts) {
                    if (!reuse_mixed_bla) {
                        if (new_counts) clReleaseMemObject(new_counts);
                        if (new_offsets) clReleaseMemObject(new_offsets);
                        if (new_bla) clReleaseMemObject(new_bla);
                    }
                    throw std::runtime_error(opencl_error_text(status));
                }
                if (!reuse_mixed_bla) {
                    if (runtime.mixed_bla_counts) clReleaseMemObject(runtime.mixed_bla_counts);
                    if (runtime.mixed_bla_offsets) clReleaseMemObject(runtime.mixed_bla_offsets);
                    if (runtime.mixed_bla) clReleaseMemObject(runtime.mixed_bla);
                }
                runtime.mixed_bla = new_bla;
                runtime.mixed_bla_offsets = new_offsets;
                runtime.mixed_bla_counts = new_counts;
                runtime.mixed_bla_capacity = mixed_bla_bytes;
                runtime.mixed_bla_levels_capacity = mixed_level_bytes;
                runtime.mixed_bla_generation = context.generation;
                runtime.mixed_bla_level_count = static_cast<int>(linear_bla_level_count);
                mixed_device_bla = new_bla;
                mixed_device_bla_offsets = new_offsets;
                mixed_device_bla_counts = new_counts;
            }
        }
    }
    std::array<cl_mem, 6> device_planes{};
    const std::array<size_t, 6> plane_sizes{
        sizeof(std::int64_t), sizeof(double), sizeof(double), sizeof(double),
        sizeof(double), sizeof(double)};
    const std::array<void*, 6> plane_outputs{
        planes ? static_cast<void*>(planes->orbit_iteration) : nullptr,
        planes ? static_cast<void*>(planes->phase) : nullptr,
        planes ? static_cast<void*>(planes->de_x) : nullptr,
        planes ? static_cast<void*>(planes->de_y) : nullptr,
        planes ? static_cast<void*>(planes->test1) : nullptr,
        planes ? static_cast<void*>(planes->test2) : nullptr,
    };
    if (planes != nullptr) {
        for (size_t index = 0; index < device_planes.size(); ++index) {
            device_planes[index] = clCreateBuffer(
                runtime.context, CL_MEM_WRITE_ONLY, count * plane_sizes[index],
                nullptr, &status);
            if (status != CL_SUCCESS || !device_planes[index]) {
                for (cl_mem buffer : device_planes) if (buffer) clReleaseMemObject(buffer);
                cleanup();
                throw std::runtime_error(opencl_error_text(status));
            }
        }
    } else {
        // These arguments are never dereferenced when write_planes is zero.
        // Supplying a valid allocation keeps ICDs that reject null cl_mem
        // arguments happy without allocating six unused full-frame buffers.
        device_planes.fill(device_output);
    }
    const auto cleanup_planes = [&] {
        if (planes != nullptr) {
            for (cl_mem buffer : device_planes) if (buffer) clReleaseMemObject(buffer);
        }
    };
    status = CL_SUCCESS;
    const int reference_count = static_cast<int>(reference_count_size);
    const int bailout_exponent = bailout.exponent;
    const double bailout_mantissa = bailout.mantissa;
    const float mixed_bailout_mantissa = static_cast<float>(bailout.mantissa);
    // The mixed kernel keeps the viewport exponent separate, so its
    // mantissa/width ratio is safe to precompute once on the host.  Doing the
    // division in the kernel used to reintroduce two fp64 operations for
    // every pixel even though the recurrence itself was fp32.
    const float mixed_view_width_scale = static_cast<float>(
        view_width.mantissa / static_cast<double>(width));
    const float mixed_view_height_scale = static_cast<float>(
        view_height.mantissa / static_cast<double>(height));
    const float mixed_output_bias = static_cast<float>(options.output_bias);
    const int write_planes = planes != nullptr ? 1 : 0;
    const int bla_level_count = static_cast<int>(linear_bla_level_count);
    const int formula = context.formula;
    const double parameter_real = context.parameter.real;
    const double parameter_imag = context.parameter.imag;
    const int parameter_exponent = context.parameter.exponent;
    const float mixed_parameter_real = static_cast<float>(parameter_real);
    const float mixed_parameter_imag = static_cast<float>(parameter_imag);
    const int point_mode = point_offsets != nullptr ? 1 : 0;
    const bool use_scalar_bla_kernel = runtime.deep_scalar_bla_kernel != nullptr
        && !use_mixed_precision
        && planes == nullptr
        && point_offsets == nullptr
        && formula == FRACTAL_FORMULA_MANDELBROT
        && use_linear_bla;
    const bool use_scalar_kernel = runtime.deep_scalar_kernel != nullptr
        && !use_mixed_precision
        && planes == nullptr
        && point_offsets == nullptr
        && formula == FRACTAL_FORMULA_MANDELBROT
        && !use_linear_bla;
    const bool use_simple_scalar_kernel = use_scalar_bla_kernel || use_scalar_kernel;
    const bool use_mixed_mandelbrot_kernel = use_mixed_precision
        && formula == FRACTAL_FORMULA_MANDELBROT
        && runtime.deep_mixed_mandelbrot_kernel != nullptr;
    cl_kernel active_kernel = use_mixed_precision
        ? (use_mixed_mandelbrot_kernel
            ? runtime.deep_mixed_mandelbrot_kernel
            : runtime.deep_mixed_kernel)
        : (use_scalar_bla_kernel
            ? runtime.deep_scalar_bla_kernel
            : (use_scalar_kernel ? runtime.deep_scalar_kernel : runtime.deep_kernel));
    if (use_mixed_precision) {
        if (status == CL_SUCCESS) status = clSetKernelArg(
            active_kernel, 0, sizeof(device_output), &device_output);
        if (status == CL_SUCCESS) status = clSetKernelArg(
            active_kernel, 1, sizeof(mixed_device_reference), &mixed_device_reference);
        if (status == CL_SUCCESS) status = clSetKernelArg(
            active_kernel, 2, sizeof(reference_count), &reference_count);
        if (status == CL_SUCCESS) status = clSetKernelArg(
            active_kernel, 3, sizeof(width), &width);
        if (status == CL_SUCCESS) status = clSetKernelArg(
            active_kernel, 4, sizeof(height), &height);
        if (status == CL_SUCCESS) status = clSetKernelArg(
            active_kernel, 5, sizeof(mixed_view_width_scale), &mixed_view_width_scale);
        if (status == CL_SUCCESS) status = clSetKernelArg(
            active_kernel, 6, sizeof(view_width.exponent), &view_width.exponent);
        if (status == CL_SUCCESS) status = clSetKernelArg(
            active_kernel, 7, sizeof(mixed_view_height_scale), &mixed_view_height_scale);
        if (status == CL_SUCCESS) status = clSetKernelArg(
            active_kernel, 8, sizeof(view_height.exponent), &view_height.exponent);
        if (status == CL_SUCCESS) status = clSetKernelArg(
            active_kernel, 9, sizeof(max_iter), &max_iter);
        if (status == CL_SUCCESS) status = clSetKernelArg(
            active_kernel, 10, sizeof(mixed_output_bias), &mixed_output_bias);
        if (status == CL_SUCCESS) status = clSetKernelArg(
            active_kernel, 11, sizeof(bailout_exponent), &bailout_exponent);
        if (status == CL_SUCCESS) status = clSetKernelArg(
            active_kernel, 12, sizeof(mixed_bailout_mantissa), &mixed_bailout_mantissa);
        if (status == CL_SUCCESS) status = clSetKernelArg(
            active_kernel, 13, sizeof(options.coordinate_mode), &options.coordinate_mode);
        if (status == CL_SUCCESS) status = clSetKernelArg(
            active_kernel, 14, sizeof(cl_mem), &mixed_device_reference_norms);
        if (status == CL_SUCCESS) status = clSetKernelArg(
            active_kernel, 15, sizeof(cl_mem), &mixed_device_bla);
        if (status == CL_SUCCESS) status = clSetKernelArg(
            active_kernel, 16, sizeof(cl_mem), &mixed_device_bla_offsets);
        if (status == CL_SUCCESS) status = clSetKernelArg(
            active_kernel, 17, sizeof(cl_mem), &mixed_device_bla_counts);
        const int mixed_bla_level_count =
            runtime.mixed_bla_level_count > 0
                && runtime.mixed_bla_generation == context.generation
                ? runtime.mixed_bla_level_count : 0;
        if (status == CL_SUCCESS) status = clSetKernelArg(
            active_kernel, 18, sizeof(mixed_bla_level_count), &mixed_bla_level_count);
        if (!use_mixed_mandelbrot_kernel) {
            if (status == CL_SUCCESS) status = clSetKernelArg(
                active_kernel, 19, sizeof(formula), &formula);
            if (status == CL_SUCCESS) status = clSetKernelArg(
                active_kernel, 20, sizeof(mixed_parameter_real), &mixed_parameter_real);
            if (status == CL_SUCCESS) status = clSetKernelArg(
                active_kernel, 21, sizeof(mixed_parameter_imag), &mixed_parameter_imag);
            if (status == CL_SUCCESS) status = clSetKernelArg(
                active_kernel, 22, sizeof(parameter_exponent), &parameter_exponent);
        }
    } else if (use_simple_scalar_kernel) {
        if (status == CL_SUCCESS) status = clSetKernelArg(active_kernel, 0, sizeof(device_output), &device_output);
        if (status == CL_SUCCESS) status = clSetKernelArg(active_kernel, 1, sizeof(device_reference), &device_reference);
        if (status == CL_SUCCESS) status = clSetKernelArg(active_kernel, 2, sizeof(reference_count), &reference_count);
        if (status == CL_SUCCESS) status = clSetKernelArg(active_kernel, 3, sizeof(width), &width);
        if (status == CL_SUCCESS) status = clSetKernelArg(active_kernel, 4, sizeof(height), &height);
        if (status == CL_SUCCESS) status = clSetKernelArg(active_kernel, 5, sizeof(view_width.mantissa), &view_width.mantissa);
        if (status == CL_SUCCESS) status = clSetKernelArg(active_kernel, 6, sizeof(view_width.exponent), &view_width.exponent);
        if (status == CL_SUCCESS) status = clSetKernelArg(active_kernel, 7, sizeof(view_height.mantissa), &view_height.mantissa);
        if (status == CL_SUCCESS) status = clSetKernelArg(active_kernel, 8, sizeof(view_height.exponent), &view_height.exponent);
        if (status == CL_SUCCESS) status = clSetKernelArg(active_kernel, 9, sizeof(max_iter), &max_iter);
        if (status == CL_SUCCESS) status = clSetKernelArg(active_kernel, 10, sizeof(options.output_bias), &options.output_bias);
        if (status == CL_SUCCESS) status = clSetKernelArg(active_kernel, 11, sizeof(bailout_exponent), &bailout_exponent);
        if (status == CL_SUCCESS) status = clSetKernelArg(active_kernel, 12, sizeof(bailout_mantissa), &bailout_mantissa);
        if (status == CL_SUCCESS) status = clSetKernelArg(active_kernel, 13, sizeof(options.coordinate_mode), &options.coordinate_mode);
        if (status == CL_SUCCESS) status = clSetKernelArg(active_kernel, 14, sizeof(cl_mem), &device_reference_norms);
        if (use_scalar_bla_kernel) {
            if (status == CL_SUCCESS) status = clSetKernelArg(
                active_kernel, 15, sizeof(cl_mem), &device_bla);
            if (status == CL_SUCCESS) status = clSetKernelArg(
                active_kernel, 16, sizeof(cl_mem), &device_bla_offsets);
            if (status == CL_SUCCESS) status = clSetKernelArg(
                active_kernel, 17, sizeof(cl_mem), &device_bla_counts);
            if (status == CL_SUCCESS) status = clSetKernelArg(
                active_kernel, 18, sizeof(bla_level_count), &bla_level_count);
        }
    } else {
        if (status == CL_SUCCESS) status = clSetKernelArg(active_kernel, 0, sizeof(device_output), &device_output);
        if (status == CL_SUCCESS) status = clSetKernelArg(active_kernel, 1, sizeof(device_reference), &device_reference);
        if (status == CL_SUCCESS) status = clSetKernelArg(active_kernel, 2, sizeof(reference_count), &reference_count);
        if (status == CL_SUCCESS) status = clSetKernelArg(active_kernel, 3, sizeof(width), &width);
        if (status == CL_SUCCESS) status = clSetKernelArg(active_kernel, 4, sizeof(height), &height);
        if (status == CL_SUCCESS) status = clSetKernelArg(active_kernel, 5, sizeof(view_width.mantissa), &view_width.mantissa);
        if (status == CL_SUCCESS) status = clSetKernelArg(active_kernel, 6, sizeof(view_width.exponent), &view_width.exponent);
        if (status == CL_SUCCESS) status = clSetKernelArg(active_kernel, 7, sizeof(view_height.mantissa), &view_height.mantissa);
        if (status == CL_SUCCESS) status = clSetKernelArg(active_kernel, 8, sizeof(view_height.exponent), &view_height.exponent);
        if (status == CL_SUCCESS) status = clSetKernelArg(active_kernel, 9, sizeof(max_iter), &max_iter);
        if (status == CL_SUCCESS) status = clSetKernelArg(active_kernel, 10, sizeof(options.output_bias), &options.output_bias);
        if (status == CL_SUCCESS) status = clSetKernelArg(active_kernel, 11, sizeof(bailout_exponent), &bailout_exponent);
        if (status == CL_SUCCESS) status = clSetKernelArg(active_kernel, 12, sizeof(bailout_mantissa), &bailout_mantissa);
        if (status == CL_SUCCESS) status = clSetKernelArg(active_kernel, 13, sizeof(options.coordinate_mode), &options.coordinate_mode);
        if (status == CL_SUCCESS) status = clSetKernelArg(active_kernel, 14, sizeof(pixel_spacing.mantissa), &pixel_spacing.mantissa);
        if (status == CL_SUCCESS) status = clSetKernelArg(active_kernel, 15, sizeof(pixel_spacing.exponent), &pixel_spacing.exponent);
        if (status == CL_SUCCESS) status = clSetKernelArg(active_kernel, 16, sizeof(write_planes), &write_planes);
        for (size_t index = 0; status == CL_SUCCESS && index < device_planes.size(); ++index) {
            status = clSetKernelArg(active_kernel, static_cast<cl_uint>(17 + index),
                                    sizeof(cl_mem), &device_planes[index]);
        }
        if (status == CL_SUCCESS) status = clSetKernelArg(active_kernel, 23, sizeof(cl_mem), &device_bla);
        if (status == CL_SUCCESS) status = clSetKernelArg(active_kernel, 24, sizeof(cl_mem), &device_bla_offsets);
        if (status == CL_SUCCESS) status = clSetKernelArg(active_kernel, 25, sizeof(cl_mem), &device_bla_counts);
        if (status == CL_SUCCESS) status = clSetKernelArg(active_kernel, 26, sizeof(bla_level_count), &bla_level_count);
        if (status == CL_SUCCESS) status = clSetKernelArg(active_kernel, 27, sizeof(use_linear_bla), &use_linear_bla);
        if (status == CL_SUCCESS) status = clSetKernelArg(active_kernel, 28, sizeof(formula), &formula);
        if (status == CL_SUCCESS) status = clSetKernelArg(active_kernel, 29, sizeof(parameter_real), &parameter_real);
        if (status == CL_SUCCESS) status = clSetKernelArg(active_kernel, 30, sizeof(parameter_imag), &parameter_imag);
        if (status == CL_SUCCESS) status = clSetKernelArg(active_kernel, 31, sizeof(parameter_exponent), &parameter_exponent);
        if (status == CL_SUCCESS) status = clSetKernelArg(
            active_kernel, 32, sizeof(cl_mem), &device_point_offsets);
        if (status == CL_SUCCESS) status = clSetKernelArg(
            active_kernel, 33, sizeof(point_mode), &point_mode);
        if (status == CL_SUCCESS) status = clSetKernelArg(
            active_kernel, 34, sizeof(cl_mem), &device_reference_norms);
    }
    const size_t workgroup = use_mixed_precision
        ? (use_mixed_mandelbrot_kernel
            ? runtime.deep_mixed_mandelbrot_workgroup_size
            : runtime.deep_mixed_workgroup_size)
        : (use_scalar_bla_kernel
            ? runtime.deep_scalar_bla_workgroup_size
            : (use_scalar_kernel
                ? runtime.deep_scalar_workgroup_size : runtime.workgroup_size));
    const size_t global_size = workgroup > 0
        ? ((count + workgroup - 1U) / workgroup) * workgroup
        : count;
    const size_t* local_work_size = workgroup > 0 ? &workgroup : nullptr;
    if (status == CL_SUCCESS) status = clEnqueueNDRangeKernel(
        runtime.queue, active_kernel, 1, nullptr, &global_size,
        local_work_size, 0, nullptr, nullptr);
    if (status == CL_SUCCESS) status = clEnqueueReadBuffer(
        runtime.queue, device_output, CL_TRUE, 0, count * sizeof(float),
        output, 0, nullptr, nullptr);
    for (size_t index = 0; status == CL_SUCCESS && planes != nullptr
         && index < device_planes.size(); ++index) {
        status = clEnqueueReadBuffer(runtime.queue, device_planes[index], CL_TRUE, 0,
                                     count * plane_sizes[index], plane_outputs[index],
                                     0, nullptr, nullptr);
    }
    // The blocking output reads above already establish completion for this
    // synchronous ABI call; do not serialize the driver a second time.
    cleanup_planes();
    cleanup();
    if (status != CL_SUCCESS) throw std::runtime_error(opencl_error_text(status));
}
#endif

void stabilize_alternate_reference_cycle(
    std::vector<FloatExpComplex>& fast_orbit,
    ReferenceOrbitData& orbit,
    int escape_radius_mode = ESCAPE_RADIUS_MODE_CLASSIC
) {
    // A decimal deep-zoom target can be a periodic point whose last supplied
    // digits are still amplified by a repelling cycle. MPFR faithfully
    // iterates those digits, but the projected reference then drifts away
    // from the intended target and makes every pixel follow the wrong
    // perturbation path. Mirror the Python alternate renderer's conservative
    // cycle test and repeat the settled projected cycle before building the
    // native linear hierarchy.
    const size_t count = std::min(fast_orbit.size(), orbit.scaled.size());
    if (count < 6 || orbit.real_double.size() != count
        || orbit.imag_double.size() != count) {
        return;
    }
    size_t last_finite = count;
    while (last_finite > 0) {
        const size_t index = last_finite - 1;
        if (std::isfinite(orbit.real_double[index])
            && std::isfinite(orbit.imag_double[index])) {
            break;
        }
        --last_finite;
    }
    if (last_finite < 6) return;
    const size_t scan_limit = std::min<size_t>(last_finite - 1, 1024);
    const int max_period = std::min<int>(
        64, static_cast<int>((scan_limit + 1) / 4));
    constexpr double tolerance = 1.0e-12;
    for (int period = 1; period <= max_period; ++period) {
        for (size_t end = static_cast<size_t>(4 * period - 1);
             end <= scan_limit;
             ++end) {
            const size_t cycle_start = end
                - static_cast<size_t>(3 * period) + 1U;
            bool matches = true;
            for (int offset = 0; offset < period && matches; ++offset) {
                const size_t previous = cycle_start + static_cast<size_t>(offset);
                const size_t current = previous + static_cast<size_t>(period);
                const size_t following = current + static_cast<size_t>(period);
                const double previous_real = orbit.real_double[previous];
                const double previous_imag = orbit.imag_double[previous];
                const double current_real = orbit.real_double[current];
                const double current_imag = orbit.imag_double[current];
                const double following_real = orbit.real_double[following];
                const double following_imag = orbit.imag_double[following];
                const double previous_radius = std::hypot(
                    previous_real, previous_imag);
                const double current_radius = std::hypot(
                    current_real, current_imag);
                const double following_radius = std::hypot(
                    following_real, following_imag);
                if (!std::isfinite(previous_radius)
                    || !std::isfinite(current_radius)
                    || !std::isfinite(following_radius)
                    || previous_radius * previous_radius
                        >= escape_radius_squared_double(escape_radius_mode)
                    || current_radius * current_radius
                        >= escape_radius_squared_double(escape_radius_mode)
                    || following_radius * following_radius
                        >= escape_radius_squared_double(escape_radius_mode)) {
                    matches = false;
                    break;
                }
                const double scale = std::max(
                    1.0,
                    std::max(
                        previous_radius,
                        std::max(current_radius, following_radius)));
                const double current_difference = std::hypot(
                    current_real - previous_real,
                    current_imag - previous_imag);
                const double following_difference = std::hypot(
                    following_real - current_real,
                    following_imag - current_imag);
                if (!(current_difference <= tolerance * scale
                      && following_difference <= tolerance * scale)) {
                    matches = false;
                }
            }
            if (!matches) continue;
            for (size_t target = cycle_start; target < count; ++target) {
                const size_t source = cycle_start
                    + (target - cycle_start) % static_cast<size_t>(period);
                fast_orbit[target] = fast_orbit[source];
                orbit.scaled[target] = orbit.scaled[source];
                orbit.real_double[target] = orbit.real_double[source];
                orbit.imag_double[target] = orbit.imag_double[source];
                if (target < orbit.escape_margin.size()
                    && source < orbit.escape_margin.size()) {
                    orbit.escape_margin[target] = orbit.escape_margin[source];
                }
            }
            return;
        }
    }
}

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
        shared->generation = static_cast<std::uint64_t>(token);
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
    int threads,
    int formula,
    double julia_real,
    double julia_imag,
    double output_bias,
    int escape_radius_mode
) {
    const double escape_squared = escape_radius_squared_double(escape_radius_mode);
    const double safe_escape_squared = escape_squared + 1.0e-7;
    const double log_formula_power = std::log(
        static_cast<double>(formula_power(formula)));
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
            if (formula == FRACTAL_FORMULA_MANDELBROT) {
                for (int lane = 0; lane < 4; ++lane) {
                    const double cx = cx_values[lane];
                    const double q = (cx - 0.25) * (cx - 0.25)
                        + cy_scalar * cy_scalar;
                    const bool in_cardioid = q * (q + cx - 0.25)
                        <= 0.25 * cy_scalar * cy_scalar;
                    const bool in_bulb = (cx + 1.0) * (cx + 1.0)
                        + cy_scalar * cy_scalar <= 0.0625;
                    if (!in_cardioid && !in_bulb) active_bits |= 1 << lane;
                    else output[py * width + px + lane] = encode_render_iteration(
                        max_iter, output_bias);
                }
            } else {
                active_bits = 0x0f;
            }
            if (active_bits == 0) continue;

            const __m256d cx = _mm256_loadu_pd(cx_values);
            const __m256d cy = _mm256_set1_pd(cy_scalar);
            const bool julia = formula == FRACTAL_FORMULA_JULIA;
            __m256d zr = julia ? cx : _mm256_setzero_pd();
            __m256d zi = julia ? cy : _mm256_setzero_pd();
            const __m256d parameter_real = julia
                ? _mm256_set1_pd(julia_real) : cx;
            const __m256d parameter_imag = julia
                ? _mm256_set1_pd(julia_imag) : cy;
            const __m256d sign_mask = _mm256_set1_pd(-0.0);
            int escaped_iteration[4] = {
                max_iter + 1,
                max_iter + 1,
                max_iter + 1,
                max_iter + 1,
            };
            double escaped_norm[4] = {0.0, 0.0, 0.0, 0.0};
            for (int iteration = 0; iteration < max_iter && active_bits != 0; ++iteration) {
                const __m256d formula_real = formula == FRACTAL_FORMULA_BURNING_SHIP
                    ? _mm256_andnot_pd(sign_mask, zr) : zr;
                const __m256d formula_imag = formula == FRACTAL_FORMULA_BURNING_SHIP
                    ? _mm256_andnot_pd(sign_mask, zi) : zi;
                const __m256d zr_squared = _mm256_mul_pd(formula_real, formula_real);
                const __m256d zi_squared = _mm256_mul_pd(formula_imag, formula_imag);
                const __m256d next_real = _mm256_add_pd(
                    _mm256_sub_pd(zr_squared, zi_squared),
                    parameter_real);
                __m256d cross = _mm256_mul_pd(
                    _mm256_add_pd(formula_real, formula_real),
                    formula_imag);
                if (formula == FRACTAL_FORMULA_TRICORN) {
                    cross = _mm256_sub_pd(_mm256_setzero_pd(), cross);
                }
                const __m256d next_imag = _mm256_add_pd(cross, parameter_imag);
                zr = next_real;
                zi = next_imag;
                const __m256d norm = _mm256_add_pd(
                    _mm256_mul_pd(zr, zr),
                    _mm256_mul_pd(zi, zi));
                alignas(32) double norm_values[4];
                _mm256_store_pd(norm_values, norm);
                int escaped = 0;
                for (int lane = 0; lane < 4; ++lane) {
                    if ((active_bits & (1 << lane))
                        && norm_values[lane] > escape_squared) {
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
                    output[index] = encode_render_iteration(max_iter, output_bias);
                    continue;
                }
                const double magnitude = std::sqrt(
                    std::max(escaped_norm[lane], safe_escape_squared));
                output[index] = encode_render_value(
                    static_cast<long double>(escaped_iteration[lane])
                        - std::log(std::log(magnitude)) / static_cast<double>(LOG_TWO),
                    output_bias);
            }
        }
        // Scalar cleanup handles a non-multiple-of-four width without a
        // masked load/store penalty in the common SIMD path.
        for (; px < width; ++px) {
            const double cx = x_coordinates[static_cast<size_t>(px)];
            const int index = py * width + px;
            if (formula == FRACTAL_FORMULA_MANDELBROT) {
                const double q = (cx - 0.25) * (cx - 0.25)
                    + cy_scalar * cy_scalar;
                const bool in_cardioid = q * (q + cx - 0.25)
                    <= 0.25 * cy_scalar * cy_scalar;
                const bool in_bulb = (cx + 1.0) * (cx + 1.0)
                    + cy_scalar * cy_scalar <= 0.0625;
                if (in_cardioid || in_bulb) {
                    output[index] = encode_render_iteration(max_iter, output_bias);
                    continue;
                }
            }
            double zr = formula == FRACTAL_FORMULA_JULIA ? cx : 0.0;
            double zi = formula == FRACTAL_FORMULA_JULIA ? cy_scalar : 0.0;
            const double parameter_real = formula == FRACTAL_FORMULA_JULIA
                ? julia_real : cx;
            const double parameter_imag = formula == FRACTAL_FORMULA_JULIA
                ? julia_imag : cy_scalar;
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
                if (magnitude_squared > escape_squared) {
                    const double magnitude = std::sqrt(
                        std::max(magnitude_squared, safe_escape_squared));
                    output[index] = encode_render_value(
                        static_cast<long double>(iteration + 1)
                            - std::log(std::log(magnitude))
                                / log_formula_power,
                        output_bias);
                    break;
                }
            }
            if (iteration == max_iter) {
                output[index] = encode_render_iteration(max_iter, output_bias);
            }
        }
    }
}
#endif

template<int Formula>
inline void iterate_direct_formula_static(
    double zr,
    double zi,
    double parameter_real,
    double parameter_imag,
    double& next_real,
    double& next_imag
) noexcept {
    if constexpr (Formula == FRACTAL_FORMULA_BURNING_SHIP) {
        const double absolute_real = std::abs(zr);
        const double absolute_imag = std::abs(zi);
        // Keep the piecewise alternate maps reproducible with the NumPy
        // fallback. A fused multiply-add changes a late orbit by a few ulps
        // and can move a boundary pixel to the other side of escape.
        const volatile double real_square = absolute_real * absolute_real;
        const volatile double imag_square = absolute_imag * absolute_imag;
        const volatile double cross = 2.0 * absolute_real * absolute_imag;
        const volatile double real_difference = real_square - imag_square;
        const volatile double imaginary_product = cross + parameter_imag;
        next_real = real_difference + parameter_real;
        next_imag = imaginary_product;
    } else if constexpr (Formula == FRACTAL_FORMULA_TRICORN) {
        const volatile double real_square = zr * zr;
        const volatile double imag_square = zi * zi;
        const volatile double cross = -2.0 * zr * zi;
        const volatile double real_difference = real_square - imag_square;
        const volatile double imaginary_product = cross + parameter_imag;
        next_real = real_difference + parameter_real;
        next_imag = imaginary_product;
    } else if constexpr (Formula == FRACTAL_FORMULA_JULIA) {
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

template<int Formula>
void render_direct_scalar_formula(
    float* output,
    int width,
    int height,
    const std::vector<double>& x_coordinates,
    const std::vector<double>& y_coordinates,
    int max_iter,
    int threads,
    double julia_real,
    double julia_imag,
    double output_bias,
    int escape_radius_mode,
    double log_formula_power
) {
    constexpr bool is_mandelbrot = Formula == FRACTAL_FORMULA_MANDELBROT;
    constexpr bool is_julia = Formula == FRACTAL_FORMULA_JULIA;
    const double escape_squared = escape_radius_squared_double(escape_radius_mode);
    const double safe_escape_squared = escape_squared + 1.0e-7;
#ifdef _OPENMP
    if (threads > 0) {
        omp_set_dynamic(0);
        omp_set_num_threads(threads);
    }
#pragma omp parallel for schedule(dynamic, 1)
#endif
    for (int py = 0; py < height; ++py) {
        const double cy = y_coordinates[static_cast<size_t>(py)];
        const size_t row_offset = static_cast<size_t>(py)
            * static_cast<size_t>(width);
        for (int px = 0; px < width; ++px) {
            const double cx = x_coordinates[static_cast<size_t>(px)];
            const size_t index = row_offset + static_cast<size_t>(px);
            if constexpr (is_mandelbrot) {
                const double q = (cx - 0.25) * (cx - 0.25) + cy * cy;
                const bool in_cardioid = q * (q + cx - 0.25)
                    <= 0.25 * cy * cy;
                const bool in_bulb = (cx + 1.0) * (cx + 1.0)
                    + cy * cy <= 0.0625;
                if (in_cardioid || in_bulb) {
                    output[index] = encode_render_iteration(max_iter, output_bias);
                    continue;
                }
            }

            double zr = is_julia ? cx : 0.0;
            double zi = is_julia ? cy : 0.0;
            const double parameter_real = is_julia ? julia_real : cx;
            const double parameter_imag = is_julia ? julia_imag : cy;
            int iteration = 0;
            for (; iteration < max_iter; ++iteration) {
                double next_real = 0.0;
                double next_imag = 0.0;
                iterate_direct_formula_static<Formula>(
                    zr,
                    zi,
                    parameter_real,
                    parameter_imag,
                    next_real,
                    next_imag);
                zr = next_real;
                zi = next_imag;
                const double magnitude_squared = zr * zr + zi * zi;
                if (magnitude_squared > escape_squared
                    || !std::isfinite(magnitude_squared)) {
                    const double safe_squared = std::isfinite(magnitude_squared)
                        ? std::max(magnitude_squared, safe_escape_squared)
                        : std::numeric_limits<double>::max();
                    const double magnitude = std::sqrt(safe_squared);
                    output[index] = encode_render_value(
                        static_cast<long double>(iteration + 1)
                            - std::log(std::log(magnitude))
                                / log_formula_power,
                        output_bias);
                    break;
                }
            }
            if (iteration == max_iter) {
                output[index] = encode_render_iteration(max_iter, output_bias);
            }
        }
    }
}

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
    double julia_imag = 0.0,
    double output_bias = 0.0,
    int escape_radius_mode = ESCAPE_RADIUS_MODE_CLASSIC,
    int coordinate_mode = COORDINATE_MODE_PROJECT
) {
    // This path is deliberately ordinary double precision.  The Python
    // layer routes only shallow views here; using long double for every
    // pixel made the inexpensive part of a zoom sequence disproportionately
    // slow on low-power CPUs.
    const double zoom_value = static_cast<double>(zoom);
    const double center_real = static_cast<double>(x_center);
    const double center_imag = static_cast<double>(y_center);
    if (!valid_escape_radius_mode(escape_radius_mode)) {
        throw std::runtime_error("unknown escape-radius mode");
    }
    if (!valid_coordinate_mode(coordinate_mode)) {
        throw std::runtime_error("unknown coordinate mode");
    }
    if (!valid_formula(formula)) {
        throw std::runtime_error("unknown fractal formula");
    }
    const double log_formula_power = std::log(
        static_cast<double>(formula_power(formula)));
    if (!std::isfinite(zoom_value) || zoom_value <= 0.0
        || !std::isfinite(center_real) || !std::isfinite(center_imag)) {
        throw std::runtime_error("direct native coordinates or zoom exceed double range");
    }
    const double height_span = viewport_height_factor(coordinate_mode) / zoom_value;
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
            pixel_axis_offset(px, width, coordinate_mode)
            * width_span / static_cast<double>(width);
        x_coordinates[static_cast<size_t>(px)] = center_real + x_offset;
    }
    for (int py = 0; py < height; ++py) {
        const double y_offset =
            -pixel_axis_offset(py, height, coordinate_mode)
            * height_span / static_cast<double>(height);
        y_coordinates[static_cast<size_t>(py)] = center_imag + y_offset;
    }
#if defined(__AVX2__)
    if (backend == 1 && avx2_runtime_available()) {
        render_direct_avx2(
            output,
            width,
            height,
            x_coordinates,
            y_coordinates,
            max_iter,
            threads,
            formula,
            julia_real,
            julia_imag,
            output_bias,
            escape_radius_mode);
        return;
    }
    if (backend == 1) {
        throw std::runtime_error("AVX2 backend requested but the CPU does not support AVX2");
    }
#else
    if (backend == 1) {
        throw std::runtime_error("AVX2 backend requested but this build has no AVX2 support");
    }
#endif
#ifdef FRACTAL_HAVE_OPENCL
    if (backend == 2) {
        render_direct_opencl(
            output,
            width,
            height,
            static_cast<double>(zoom),
            static_cast<double>(x_center),
            static_cast<double>(y_center),
            max_iter,
            output_bias,
            formula,
            julia_real,
            julia_imag,
            escape_radius_mode,
            coordinate_mode);
        return;
    }
#else
    if (backend == 2) {
        throw std::runtime_error("OpenCL backend is not available in this build");
    }
#endif
    // Formula selection is fixed for the whole frame. Dispatching once here
    // removes the predictable formula/Julia branches from every orbit step
    // in the shallow renderer used by live view and low-depth previews.
    switch (formula) {
        case FRACTAL_FORMULA_MANDELBROT:
            render_direct_scalar_formula<FRACTAL_FORMULA_MANDELBROT>(
                output, width, height, x_coordinates, y_coordinates,
                max_iter, threads, julia_real, julia_imag, output_bias,
                escape_radius_mode, log_formula_power);
            return;
        case FRACTAL_FORMULA_JULIA:
            render_direct_scalar_formula<FRACTAL_FORMULA_JULIA>(
                output, width, height, x_coordinates, y_coordinates,
                max_iter, threads, julia_real, julia_imag, output_bias,
                escape_radius_mode, log_formula_power);
            return;
        case FRACTAL_FORMULA_BURNING_SHIP:
            render_direct_scalar_formula<FRACTAL_FORMULA_BURNING_SHIP>(
                output, width, height, x_coordinates, y_coordinates,
                max_iter, threads, julia_real, julia_imag, output_bias,
                escape_radius_mode, log_formula_power);
            return;
        case FRACTAL_FORMULA_TRICORN:
            render_direct_scalar_formula<FRACTAL_FORMULA_TRICORN>(
                output, width, height, x_coordinates, y_coordinates,
                max_iter, threads, julia_real, julia_imag, output_bias,
                escape_radius_mode, log_formula_power);
            return;
        default:
            throw std::runtime_error("unknown fractal formula");
    }
}

template<int Formula, bool NeedDerivative>
void render_direct_with_planes_formula(
    float* output,
    int width,
    int height,
    const std::vector<double>& x_coordinates,
    const std::vector<double>& y_coordinates,
    int max_iter,
    int threads,
    double julia_real,
    double julia_imag,
    double output_bias,
    int escape_radius_mode,
    double log_formula_power,
    const FloatExp& pixel_spacing,
    std::uint32_t plane_flags,
    FractalRenderPlanes* planes
) {
    constexpr bool is_mandelbrot = Formula == FRACTAL_FORMULA_MANDELBROT;
    constexpr bool is_julia = Formula == FRACTAL_FORMULA_JULIA;
    constexpr bool has_analytic_de = is_mandelbrot || is_julia;
    constexpr bool parameter_plane = !is_julia;
    const double escape_squared = escape_radius_squared_double(escape_radius_mode);
    const double safe_escape_squared = escape_squared + 1.0e-7;
#ifdef _OPENMP
    if (threads > 0) {
        omp_set_dynamic(0);
        omp_set_num_threads(threads);
    }
#pragma omp parallel for schedule(dynamic, 1)
#endif
    for (int py = 0; py < height; ++py) {
        const double cy = y_coordinates[static_cast<size_t>(py)];
        const size_t row_offset = static_cast<size_t>(py)
            * static_cast<size_t>(width);
        for (int px = 0; px < width; ++px) {
            const double cx = x_coordinates[static_cast<size_t>(px)];
            const size_t index = row_offset + static_cast<size_t>(px);
            auto mark_inside = [&] {
                output[index] = encode_render_iteration(max_iter, output_bias);
                planes->orbit_iteration[index] = max_iter;
                planes->phase[index] = 0.0;
                planes->de_x[index] = 0.0;
                planes->de_y[index] = 0.0;
                planes->test1[index] = 0.0;
                planes->test2[index] = 0.0;
            };
            if constexpr (is_mandelbrot) {
                const double q = (cx - 0.25) * (cx - 0.25) + cy * cy;
                const bool in_cardioid = q * (q + cx - 0.25)
                    <= 0.25 * cy * cy;
                const bool in_bulb = (cx + 1.0) * (cx + 1.0)
                    + cy * cy <= 0.0625;
                if (in_cardioid || in_bulb) {
                    mark_inside();
                    continue;
                }
            }

            double zr = 0.0;
            double zi = 0.0;
            double parameter_real = 0.0;
            double parameter_imag = 0.0;
            double derivative_real = 0.0;
            if constexpr (is_julia) {
                zr = cx;
                zi = cy;
                parameter_real = julia_real;
                parameter_imag = julia_imag;
                derivative_real = 1.0;
            } else {
                parameter_real = cx;
                parameter_imag = cy;
            }
            double derivative_imag = 0.0;
            double test1 = 0.0;
            double test2 = 0.0;
            int iteration = 0;
            for (; iteration < max_iter; ++iteration) {
                double next_real = 0.0;
                double next_imag = 0.0;
                double next_derivative_real = 0.0;
                double next_derivative_imag = 0.0;
                iterate_direct_formula_with_derivative<Formula, parameter_plane, NeedDerivative>(
                    zr,
                    zi,
                    parameter_real,
                    parameter_imag,
                    derivative_real,
                    derivative_imag,
                    next_real,
                    next_imag,
                    next_derivative_real,
                    next_derivative_imag);
                zr = next_real;
                zi = next_imag;
                derivative_real = next_derivative_real;
                derivative_imag = next_derivative_imag;
                const double magnitude_squared = zr * zr + zi * zi;
                test2 = test1;
                test1 = magnitude_squared;
                if (magnitude_squared > escape_squared
                    || !std::isfinite(magnitude_squared)) {
                    const double safe_squared = std::isfinite(magnitude_squared)
                        ? std::max(magnitude_squared, safe_escape_squared)
                        : std::numeric_limits<double>::max();
                    const double magnitude = std::sqrt(safe_squared);
                    output[index] = encode_render_value(
                        static_cast<long double>(iteration + 1)
                            - std::log(std::log(magnitude))
                                / log_formula_power,
                        output_bias);
                    // Kalles increments `antal` after each tested sample;
                    // the first escaped z^2+c sample therefore carries
                    // `antal == 0`, matching this zero-based loop variable.
                    planes->orbit_iteration[index] = iteration;
                    planes->phase[index] = 0.0;
                    if ((plane_flags & FRACTAL_RENDER_HINT_SKIP_PHASE) == 0) {
                        planes->phase[index] = std::isfinite(zr)
                            && std::isfinite(zi)
                            ? [&] {
                                double value = std::atan2(zi, zr)
                                    / 6.283185307179586476925286766559005768;
                                value -= std::floor(value);
                                return value;
                            }()
                            : 0.0;
                    }
                    planes->de_x[index] = 0.0;
                    planes->de_y[index] = 0.0;
                    if constexpr (NeedDerivative && has_analytic_de) {
                        if (std::isfinite(zr) && std::isfinite(zi)
                            && std::isfinite(derivative_real)
                            && std::isfinite(derivative_imag)) {
                            const ScaledComplex total =
                                ScaledComplex::from_float_exp(
                                    FloatExp::from_parts(zr, 0),
                                    FloatExp::from_parts(zi, 0));
                            const ScaledComplex derivative =
                                ScaledComplex::from_float_exp(
                                    FloatExp::from_parts(derivative_real, 0),
                                    FloatExp::from_parts(derivative_imag, 0));
                            (void)render_plane_de(
                                total,
                                derivative,
                                pixel_spacing,
                                planes->de_x[index],
                                planes->de_y[index]);
                        }
                    }
                    planes->test1[index] = test1;
                    planes->test2[index] = test2;
                    break;
                }
            }
            if (iteration == max_iter) {
                mark_inside();
            }
        }
    }
}

void render_direct_with_planes(
    float* output,
    int width,
    int height,
    long double zoom,
    long double x_center,
    long double y_center,
    int max_iter,
    int threads,
    int formula,
    double julia_real,
    double julia_imag,
    double output_bias,
    int escape_radius_mode,
    int coordinate_mode,
    std::uint32_t plane_flags,
    FractalRenderPlanes* planes
) {
    if (!planes || !planes->orbit_iteration || !planes->phase
        || !planes->de_x || !planes->de_y
        || !planes->test1 || !planes->test2) {
        throw std::runtime_error("native render metadata planes are incomplete");
    }
    const double zoom_value = static_cast<double>(zoom);
    const double center_real = static_cast<double>(x_center);
    const double center_imag = static_cast<double>(y_center);
    if (!valid_escape_radius_mode(escape_radius_mode)) {
        throw std::runtime_error("unknown escape-radius mode");
    }
    if (!valid_coordinate_mode(coordinate_mode)) {
        throw std::runtime_error("unknown coordinate mode");
    }
    if (!valid_formula(formula)) {
        throw std::runtime_error("unknown fractal formula");
    }
    const double log_formula_power = std::log(
        static_cast<double>(formula_power(formula)));
    if (!std::isfinite(zoom_value) || zoom_value <= 0.0
        || !std::isfinite(center_real) || !std::isfinite(center_imag)) {
        throw std::runtime_error("direct native coordinates or zoom exceed double range");
    }
    const double height_span = viewport_height_factor(coordinate_mode) / zoom_value;
    const double width_span = height_span * static_cast<double>(width)
        / static_cast<double>(height);
    if (!std::isfinite(height_span) || !std::isfinite(width_span)) {
        throw std::runtime_error("direct native viewport is outside double range");
    }
    // This is the screen-space scale used by Kalles' analytic DE formula.
    // Keep it beside the direct viewport geometry so shallow metadata and
    // deep/BLA metadata use the same unit convention.
    const FloatExp pixel_spacing = FloatExp::from_parts(
        height_span / static_cast<double>(height),
        0);
    std::vector<double> x_coordinates(static_cast<size_t>(width));
    std::vector<double> y_coordinates(static_cast<size_t>(height));
    for (int px = 0; px < width; ++px) {
        const double x_offset =
            pixel_axis_offset(px, width, coordinate_mode)
            * width_span / static_cast<double>(width);
        x_coordinates[static_cast<size_t>(px)] = center_real + x_offset;
    }
    for (int py = 0; py < height; ++py) {
        const double y_offset =
            -pixel_axis_offset(py, height, coordinate_mode)
            * height_span / static_cast<double>(height);
        y_coordinates[static_cast<size_t>(py)] = center_imag + y_offset;
    }
    const bool skip_analytic_de =
        (plane_flags & FRACTAL_RENDER_HINT_SKIP_ANALYTIC_DE) != 0;
    switch (formula) {
        case FRACTAL_FORMULA_MANDELBROT:
            if (skip_analytic_de) {
                render_direct_with_planes_formula<
                    FRACTAL_FORMULA_MANDELBROT, false>(
                    output, width, height, x_coordinates, y_coordinates,
                    max_iter, threads, julia_real, julia_imag, output_bias,
                    escape_radius_mode, log_formula_power, pixel_spacing,
                    plane_flags, planes);
            } else {
                render_direct_with_planes_formula<
                    FRACTAL_FORMULA_MANDELBROT, true>(
                    output, width, height, x_coordinates, y_coordinates,
                    max_iter, threads, julia_real, julia_imag, output_bias,
                    escape_radius_mode, log_formula_power, pixel_spacing,
                    plane_flags, planes);
            }
            return;
        case FRACTAL_FORMULA_JULIA:
            if (skip_analytic_de) {
                render_direct_with_planes_formula<FRACTAL_FORMULA_JULIA, false>(
                    output, width, height, x_coordinates, y_coordinates,
                    max_iter, threads, julia_real, julia_imag, output_bias,
                    escape_radius_mode, log_formula_power, pixel_spacing,
                    plane_flags, planes);
            } else {
                render_direct_with_planes_formula<FRACTAL_FORMULA_JULIA, true>(
                    output, width, height, x_coordinates, y_coordinates,
                    max_iter, threads, julia_real, julia_imag, output_bias,
                    escape_radius_mode, log_formula_power, pixel_spacing,
                    plane_flags, planes);
            }
            return;
        case FRACTAL_FORMULA_BURNING_SHIP:
            render_direct_with_planes_formula<
                FRACTAL_FORMULA_BURNING_SHIP, false>(
                output, width, height, x_coordinates, y_coordinates,
                max_iter, threads, julia_real, julia_imag, output_bias,
                escape_radius_mode, log_formula_power, pixel_spacing,
                plane_flags, planes);
            return;
        case FRACTAL_FORMULA_TRICORN:
            render_direct_with_planes_formula<FRACTAL_FORMULA_TRICORN, false>(
                output, width, height, x_coordinates, y_coordinates,
                max_iter, threads, julia_real, julia_imag, output_bias,
                escape_radius_mode, log_formula_power, pixel_spacing,
                plane_flags, planes);
            return;
        default:
            throw std::runtime_error("unknown fractal formula");
    }
}

#ifdef FRACTAL_HAVE_MPFR

struct MpfrWorkspace {
    mpfr_t cx, cy, viewport_zoom, viewport_radius;
    mpfr_t parameter_real, parameter_imag;
    mpfr_t zr, zi, next_real, next_imag, temporary;
    mpfr_t derivative_real, derivative_imag;
    mpfr_t next_derivative_real, next_derivative_imag, temporary2;
    mpfr_t absolute_real, absolute_imag, norm_squared, margin;
    // Scratch values for recovering the repelling fixed point used by Julia
    // catalogue targets.  Iterating a finite decimal approximation is not a
    // valid deep reference: the last supplied digit is eventually amplified
    // and turns the target itself into an apparent escape.
    mpfr_t discriminant_real, discriminant_imag, discriminant_magnitude;
    mpfr_t root_real, root_imag;
    mpfr_t root_a_real, root_a_imag, root_b_real, root_b_imag;
    mpfr_t root_a_norm, root_b_norm, distance_squared, multiplier_squared;
    mpfr_t fixed_tolerance, viewport_log10;

    explicit MpfrWorkspace(mpfr_prec_t precision_bits) {
        mpfr_init2(cx, precision_bits);
        mpfr_init2(cy, precision_bits);
        mpfr_init2(viewport_zoom, precision_bits);
        mpfr_init2(viewport_radius, precision_bits);
        mpfr_init2(parameter_real, precision_bits);
        mpfr_init2(parameter_imag, precision_bits);
        mpfr_init2(zr, precision_bits);
        mpfr_init2(zi, precision_bits);
        mpfr_init2(next_real, precision_bits);
        mpfr_init2(next_imag, precision_bits);
        mpfr_init2(temporary, precision_bits);
        mpfr_init2(derivative_real, precision_bits);
        mpfr_init2(derivative_imag, precision_bits);
        mpfr_init2(next_derivative_real, precision_bits);
        mpfr_init2(next_derivative_imag, precision_bits);
        mpfr_init2(temporary2, precision_bits);
        mpfr_init2(absolute_real, precision_bits);
        mpfr_init2(absolute_imag, precision_bits);
        mpfr_init2(norm_squared, precision_bits);
        mpfr_init2(margin, precision_bits);
        mpfr_init2(discriminant_real, precision_bits);
        mpfr_init2(discriminant_imag, precision_bits);
        mpfr_init2(discriminant_magnitude, precision_bits);
        mpfr_init2(root_real, precision_bits);
        mpfr_init2(root_imag, precision_bits);
        mpfr_init2(root_a_real, precision_bits);
        mpfr_init2(root_a_imag, precision_bits);
        mpfr_init2(root_b_real, precision_bits);
        mpfr_init2(root_b_imag, precision_bits);
        mpfr_init2(root_a_norm, precision_bits);
        mpfr_init2(root_b_norm, precision_bits);
        mpfr_init2(distance_squared, precision_bits);
        mpfr_init2(multiplier_squared, precision_bits);
        mpfr_init2(fixed_tolerance, precision_bits);
        mpfr_init2(viewport_log10, precision_bits);
    }

    ~MpfrWorkspace() {
        mpfr_clears(
            cx, cy, viewport_zoom, viewport_radius,
            parameter_real, parameter_imag, zr, zi,
            next_real, next_imag, temporary,
            derivative_real, derivative_imag,
            next_derivative_real, next_derivative_imag, temporary2,
            absolute_real, absolute_imag, norm_squared, margin,
            discriminant_real, discriminant_imag, discriminant_magnitude,
            root_real, root_imag,
            root_a_real, root_a_imag, root_b_real, root_b_imag,
            root_a_norm, root_b_norm, distance_squared, multiplier_squared,
            fixed_tolerance, viewport_log10, nullptr);
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
    int precision_bits,
    int formula,
    const char* julia_real_text,
    const char* julia_imag_text,
    int escape_radius_mode,
    int coordinate_mode
) {
    if (!valid_c_string(x_text) || !valid_c_string(y_text)
        || (viewport_zoom_text && !valid_c_string(viewport_zoom_text))
        || !valid_formula(formula)
        || !valid_escape_radius_mode(escape_radius_mode)
        || !valid_coordinate_mode(coordinate_mode)
        || !valid_c_string(julia_real_text)
        || !valid_c_string(julia_imag_text)) {
        throw std::runtime_error("native reference text is too long or null");
    }
    context.requested_max_iter = max_iter;
    context.escape_radius_mode = escape_radius_mode;
    context.x_center_text = x_text;
    context.y_center_text = y_text;
    precision_bits = std::max(128, precision_bits);
    MpfrWorkspace workspace(static_cast<mpfr_prec_t>(precision_bits));
    mpfr_ptr cx = workspace.cx;
    mpfr_ptr cy = workspace.cy;
    mpfr_ptr viewport_zoom = workspace.viewport_zoom;
    mpfr_ptr viewport_radius = workspace.viewport_radius;
    mpfr_ptr parameter_real = workspace.parameter_real;
    mpfr_ptr parameter_imag = workspace.parameter_imag;
    mpfr_ptr zr = workspace.zr;
    mpfr_ptr zi = workspace.zi;
    mpfr_ptr next_real = workspace.next_real;
    mpfr_ptr next_imag = workspace.next_imag;
    mpfr_ptr temporary = workspace.temporary;
    mpfr_ptr absolute_real = workspace.absolute_real;
    mpfr_ptr absolute_imag = workspace.absolute_imag;
    mpfr_ptr norm_squared = workspace.norm_squared;
    mpfr_ptr margin = workspace.margin;
    mpfr_ptr discriminant_real = workspace.discriminant_real;
    mpfr_ptr discriminant_imag = workspace.discriminant_imag;
    mpfr_ptr discriminant_magnitude = workspace.discriminant_magnitude;
    mpfr_ptr root_real = workspace.root_real;
    mpfr_ptr root_imag = workspace.root_imag;
    mpfr_ptr root_a_real = workspace.root_a_real;
    mpfr_ptr root_a_imag = workspace.root_a_imag;
    mpfr_ptr root_b_real = workspace.root_b_real;
    mpfr_ptr root_b_imag = workspace.root_b_imag;
    mpfr_ptr root_a_norm = workspace.root_a_norm;
    mpfr_ptr root_b_norm = workspace.root_b_norm;
    mpfr_ptr distance_squared = workspace.distance_squared;
    mpfr_ptr multiplier_squared = workspace.multiplier_squared;
    mpfr_ptr fixed_tolerance = workspace.fixed_tolerance;
    mpfr_ptr viewport_log10 = workspace.viewport_log10;
    if (mpfr_set_str(cx, x_text, 10, MPFR_RNDN) != 0 || mpfr_set_str(cy, y_text, 10, MPFR_RNDN) != 0) {
        throw std::runtime_error("invalid MPFR Mandelbrot centre");
    }
    if (viewport_zoom_text && mpfr_set_str(viewport_zoom, viewport_zoom_text, 10, MPFR_RNDN) == 0
        && mpfr_sgn(viewport_zoom) > 0) {
        // Approximate diagonal half-span of the view.  BLA composition uses
        // this as the bound for |delta c|, so every cropped source frame is
        // covered by the same reusable approximation table.
        mpfr_set_d(
            viewport_radius,
            viewport_height_factor(coordinate_mode),
            MPFR_RNDN);
        mpfr_div(viewport_radius, viewport_radius, viewport_zoom, MPFR_RNDN);
        context.bla.input_radius = FloatExp::from_mpfr(viewport_radius);
    } else {
        context.bla.input_radius = FloatExp::from_parts(1.0, 0);
    }
    if (formula == FRACTAL_FORMULA_JULIA) {
        if (mpfr_set_str(parameter_real, julia_real_text, 10, MPFR_RNDN) != 0
            || mpfr_set_str(parameter_imag, julia_imag_text, 10, MPFR_RNDN) != 0) {
            throw std::runtime_error("invalid MPFR Julia constant");
        }
        mpfr_set(zr, cx, MPFR_RNDN);
        mpfr_set(zi, cy, MPFR_RNDN);
    } else {
        mpfr_set(parameter_real, cx, MPFR_RNDN);
        mpfr_set(parameter_imag, cy, MPFR_RNDN);
        mpfr_set_zero(zr, 0);
        mpfr_set_zero(zi, 0);
    }
    context.formula = formula;
    context.julia_real = parse_coordinate(julia_real_text, "Julia real");
    context.julia_imag = parse_coordinate(julia_imag_text, "Julia imaginary");
    context.parameter = ScaledComplex::from_float_exp(
        FloatExp::from_mpfr(parameter_real),
        FloatExp::from_mpfr(parameter_imag));
    context.precision_bits = static_cast<mpfr_prec_t>(precision_bits);

    bool fixed_point_reference = false;
    if (formula == FRACTAL_FORMULA_JULIA) {
        // A Julia catalogue centre is exported as a decimal approximation to
        // the repelling fixed point of z^2+c.  Letting MPFR iterate that
        // approximation is still wrong at deep zooms: its final decimal bit
        // is amplified until the reference escapes. Recover the algebraic
        // root exactly at the working precision, matching the Python fallback
        // and keeping the centre bounded while pixel deltas remain active.
        mpfr_set_ui(temporary, 4, MPFR_RNDN);
        mpfr_mul(discriminant_real, parameter_real, temporary, MPFR_RNDN);
        mpfr_ui_sub(discriminant_real, 1, discriminant_real, MPFR_RNDN);
        mpfr_mul(discriminant_imag, parameter_imag, temporary, MPFR_RNDN);
        mpfr_neg(discriminant_imag, discriminant_imag, MPFR_RNDN);

        mpfr_mul(norm_squared, discriminant_real, discriminant_real, MPFR_RNDN);
        mpfr_mul(temporary, discriminant_imag, discriminant_imag, MPFR_RNDN);
        mpfr_add(norm_squared, norm_squared, temporary, MPFR_RNDN);
        mpfr_sqrt(discriminant_magnitude, norm_squared, MPFR_RNDN);

        mpfr_add(root_real, discriminant_magnitude, discriminant_real, MPFR_RNDN);
        mpfr_div_2ui(root_real, root_real, 1, MPFR_RNDN);
        mpfr_sqrt(root_real, root_real, MPFR_RNDN);
        mpfr_sub(root_imag, discriminant_magnitude, discriminant_real, MPFR_RNDN);
        mpfr_div_2ui(root_imag, root_imag, 1, MPFR_RNDN);
        mpfr_sqrt(root_imag, root_imag, MPFR_RNDN);
        if (mpfr_sgn(discriminant_imag) < 0) {
            mpfr_neg(root_imag, root_imag, MPFR_RNDN);
        }

        mpfr_add_ui(root_a_real, root_real, 1, MPFR_RNDN);
        mpfr_div_2ui(root_a_real, root_a_real, 1, MPFR_RNDN);
        mpfr_div_2ui(root_a_imag, root_imag, 1, MPFR_RNDN);
        mpfr_set_ui(temporary, 1, MPFR_RNDN);
        mpfr_sub(root_b_real, temporary, root_real, MPFR_RNDN);
        mpfr_div_2ui(root_b_real, root_b_real, 1, MPFR_RNDN);
        mpfr_neg(root_b_imag, root_a_imag, MPFR_RNDN);

        mpfr_mul(root_a_norm, root_a_real, root_a_real, MPFR_RNDN);
        mpfr_mul(temporary, root_a_imag, root_a_imag, MPFR_RNDN);
        mpfr_add(root_a_norm, root_a_norm, temporary, MPFR_RNDN);
        mpfr_mul(root_b_norm, root_b_real, root_b_real, MPFR_RNDN);
        mpfr_mul(temporary, root_b_imag, root_b_imag, MPFR_RNDN);
        mpfr_add(root_b_norm, root_b_norm, temporary, MPFR_RNDN);

        if (mpfr_cmp(root_a_norm, root_b_norm) >= 0) {
            mpfr_set(next_real, root_a_real, MPFR_RNDN);
            mpfr_set(next_imag, root_a_imag, MPFR_RNDN);
            mpfr_set(norm_squared, root_a_norm, MPFR_RNDN);
        } else {
            mpfr_set(next_real, root_b_real, MPFR_RNDN);
            mpfr_set(next_imag, root_b_imag, MPFR_RNDN);
            mpfr_set(norm_squared, root_b_norm, MPFR_RNDN);
        }
        mpfr_mul_ui(multiplier_squared, norm_squared, 4, MPFR_RNDN);

        mpfr_log10(viewport_log10, viewport_zoom, MPFR_RNDN);
        const double log_zoom = mpfr_get_d(viewport_log10, MPFR_RNDN);
        const int required_digits = std::max(
            32,
            static_cast<int>(std::ceil(std::max(0.0, log_zoom))) + 16);
        mpfr_set_ui(fixed_tolerance, 10, MPFR_RNDN);
        mpfr_pow_si(
            fixed_tolerance,
            fixed_tolerance,
            -static_cast<long>(required_digits),
            MPFR_RNDN);

        mpfr_sub(temporary, cx, next_real, MPFR_RNDN);
        mpfr_mul(distance_squared, temporary, temporary, MPFR_RNDN);
        mpfr_sub(temporary, cy, next_imag, MPFR_RNDN);
        mpfr_mul(norm_squared, temporary, temporary, MPFR_RNDN);
        mpfr_add(distance_squared, distance_squared, norm_squared, MPFR_RNDN);
        mpfr_mul(norm_squared, fixed_tolerance, fixed_tolerance, MPFR_RNDN);
        if (mpfr_cmp_ui(multiplier_squared, 1) > 0
            && mpfr_cmp(distance_squared, norm_squared) <= 0) {
            mpfr_set(zr, next_real, MPFR_RNDN);
            mpfr_set(zi, next_imag, MPFR_RNDN);
            fixed_point_reference = true;
        }
    }
    context.fast_orbit.clear();
    context.fast_orbit.reserve(static_cast<size_t>(max_iter) + 1U);
    std::vector<FloatExp> escape_margins;
    escape_margins.reserve(static_cast<size_t>(max_iter) + 1U);
    FloatExp orbit_real_value = FloatExp::from_mpfr(zr);
    FloatExp orbit_imag_value = FloatExp::from_mpfr(zi);
    size_t finite_orbit_size = 0;
    for (int i = 0; i <= max_iter; ++i) {
        if (!orbit_real_value.finite() || !orbit_imag_value.finite()) break;
        context.fast_orbit.push_back({orbit_real_value, orbit_imag_value});
        mpfr_mul(norm_squared, zr, zr, MPFR_RNDN);
        mpfr_mul(temporary, zi, zi, MPFR_RNDN);
        mpfr_add(norm_squared, norm_squared, temporary, MPFR_RNDN);
        mpfr_set_d(
            temporary,
            static_cast<double>(escape_radius_squared_for_mode(escape_radius_mode)),
            MPFR_RNDN);
        mpfr_sub(margin, norm_squared, temporary, MPFR_RNDN);
        escape_margins.push_back(FloatExp::from_mpfr(margin));
        finite_orbit_size = context.fast_orbit.size();
        if (fixed_point_reference) continue;
        if (formula == FRACTAL_FORMULA_BURNING_SHIP) {
            mpfr_abs(absolute_real, zr, MPFR_RNDN);
            mpfr_abs(absolute_imag, zi, MPFR_RNDN);
            mpfr_mul(next_real, absolute_real, absolute_real, MPFR_RNDN);
            mpfr_mul(temporary, absolute_imag, absolute_imag, MPFR_RNDN);
            mpfr_sub(next_real, next_real, temporary, MPFR_RNDN);
            mpfr_add(next_real, next_real, parameter_real, MPFR_RNDN);
            mpfr_mul(next_imag, absolute_real, absolute_imag, MPFR_RNDN);
            mpfr_mul_ui(next_imag, next_imag, 2, MPFR_RNDN);
            mpfr_add(next_imag, next_imag, parameter_imag, MPFR_RNDN);
        } else {
            mpfr_mul(next_real, zr, zr, MPFR_RNDN);
            mpfr_mul(temporary, zi, zi, MPFR_RNDN);
            mpfr_sub(next_real, next_real, temporary, MPFR_RNDN);
            if (formula == FRACTAL_FORMULA_TRICORN) {
                mpfr_mul(next_imag, zr, zi, MPFR_RNDN);
                mpfr_mul_ui(next_imag, next_imag, 2, MPFR_RNDN);
                mpfr_neg(next_imag, next_imag, MPFR_RNDN);
            } else {
                mpfr_mul(next_imag, zr, zi, MPFR_RNDN);
                mpfr_mul_ui(next_imag, next_imag, 2, MPFR_RNDN);
            }
            mpfr_add(next_real, next_real, parameter_real, MPFR_RNDN);
            mpfr_add(next_imag, next_imag, parameter_imag, MPFR_RNDN);
        }
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
    const FloatExp& escape_radius_squared =
        escape_radius_squared_float_exp(escape_radius_mode);
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
    render_orbit->escape_margin = std::move(escape_margins);
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
    if (formula != FRACTAL_FORMULA_MANDELBROT) {
        stabilize_alternate_reference_cycle(
            context.fast_orbit,
            *render_orbit,
            escape_radius_mode);
        // A false projected escape before cycle stabilization must not
        // shorten the alternate linear hierarchy. Recompute its endpoint
        // from the stabilized orbit; the exact perturbation loop still owns
        // the final escape classification.
        context.bla.map_end = static_cast<int>(render_orbit->scaled.size()) - 1;
        for (size_t index = 1; index < render_orbit->scaled.size(); ++index) {
            if (fe_compare(
                    fec_norm_squared(context.fast_orbit[index]),
                    escape_radius_squared) > 0) {
                context.bla.map_end = static_cast<int>(index);
                break;
            }
        }
    }
    context.orbit = std::move(render_orbit);
}

// A compact reference is deliberately stored with double mantissas, so a
// parameter that diverges through a near-zero orbit can eventually cancel the
// reference state more deeply than that representation can preserve.  A
// strict atlas render must not turn that event into a guessed interior value.
// Re-evaluate only the affected pixel with MPFR instead.  This is uncommon,
// keeps the normal BLA path fast, and also gives the KFP colouriser the same
// orbit/DE metadata as a successful perturbation render.
bool render_exact_mandelbrot_pixel(
    float& output,
    size_t index,
    const ReferenceContext& context,
    const ScaledComplex& dc,
    int max_iter,
    double output_bias,
    int escape_radius_mode,
    const FloatExp& pixel_spacing,
    FractalRenderPlanes* planes
) {
    if (context.x_center_text.empty() || context.y_center_text.empty()
        || context.precision_bits < 128
        || !sc_finite(dc)) {
        return false;
    }

    // A point repair can contain thousands of glitch pixels. Reuse one MPFR
    // scratch set per OpenMP worker instead of allocating and clearing a full
    // high-precision workspace for every pixel. The viewport spacing also
    // gives a tighter precision floor than the number of decorative decimal
    // digits in the stored centre: retain a generous binary guard, but do not
    // make every e100 repair pay for 170 decimal places that cannot affect a
    // displayed pixel.
    mpfr_prec_t exact_precision = context.precision_bits;
    if (pixel_spacing.exponent < 0) {
        const mpfr_prec_t spacing_precision = static_cast<mpfr_prec_t>(
            -static_cast<long long>(pixel_spacing.exponent) + 128);
        exact_precision = std::min(
            exact_precision,
            std::max<mpfr_prec_t>(256, spacing_precision));
    }
    thread_local std::unique_ptr<MpfrWorkspace> cached_workspace;
    thread_local mpfr_prec_t cached_precision = 0;
    if (!cached_workspace || cached_precision != exact_precision) {
        cached_workspace = std::make_unique<MpfrWorkspace>(exact_precision);
        cached_precision = exact_precision;
    }
    MpfrWorkspace& workspace = *cached_workspace;
    if (mpfr_set_str(
            workspace.parameter_real,
            context.x_center_text.c_str(),
            10,
            MPFR_RNDN) != 0
        || mpfr_set_str(
            workspace.parameter_imag,
            context.y_center_text.c_str(),
            10,
            MPFR_RNDN) != 0) {
        return false;
    }

    // The point ABI supplies dc as an exact binary mantissa/exponent pair.
    // Add that pair to the original decimal reference centre at the same
    // precision as the MPFR orbit, instead of routing it through long double.
    mpfr_set_d(workspace.temporary, dc.real, MPFR_RNDN);
    mpfr_mul_2si(
        workspace.temporary,
        workspace.temporary,
        static_cast<long>(dc.exponent),
        MPFR_RNDN);
    mpfr_add(
        workspace.parameter_real,
        workspace.parameter_real,
        workspace.temporary,
        MPFR_RNDN);
    mpfr_set_d(workspace.temporary, dc.imag, MPFR_RNDN);
    mpfr_mul_2si(
        workspace.temporary,
        workspace.temporary,
        static_cast<long>(dc.exponent),
        MPFR_RNDN);
    mpfr_add(
        workspace.parameter_imag,
        workspace.parameter_imag,
        workspace.temporary,
        MPFR_RNDN);

    mpfr_set_zero(workspace.zr, 0);
    mpfr_set_zero(workspace.zi, 0);
    mpfr_set_zero(workspace.derivative_real, 0);
    mpfr_set_zero(workspace.derivative_imag, 0);
    ScaledNorm previous_norm{};
    const double escape_squared = escape_radius_squared_double(escape_radius_mode);

    for (int iteration = 1; iteration <= max_iter; ++iteration) {
        // z' = z^2 + c
        mpfr_mul(workspace.next_real, workspace.zr, workspace.zr, MPFR_RNDN);
        mpfr_mul(workspace.temporary, workspace.zi, workspace.zi, MPFR_RNDN);
        mpfr_sub(workspace.next_real, workspace.next_real, workspace.temporary, MPFR_RNDN);
        mpfr_mul(workspace.next_imag, workspace.zr, workspace.zi, MPFR_RNDN);
        mpfr_mul_ui(workspace.next_imag, workspace.next_imag, 2, MPFR_RNDN);
        mpfr_add(
            workspace.next_real,
            workspace.next_real,
            workspace.parameter_real,
            MPFR_RNDN);
        mpfr_add(
            workspace.next_imag,
            workspace.next_imag,
            workspace.parameter_imag,
            MPFR_RNDN);

        // dz'/dc = 2 z dz/dc + 1.  Keep the old z and derivative live until
        // both components have been written to their separate scratch slots.
        mpfr_mul(
            workspace.next_derivative_real,
            workspace.zr,
            workspace.derivative_real,
            MPFR_RNDN);
        mpfr_mul(
            workspace.temporary,
            workspace.zi,
            workspace.derivative_imag,
            MPFR_RNDN);
        mpfr_sub(
            workspace.next_derivative_real,
            workspace.next_derivative_real,
            workspace.temporary,
            MPFR_RNDN);
        mpfr_mul_ui(
            workspace.next_derivative_real,
            workspace.next_derivative_real,
            2,
            MPFR_RNDN);
        mpfr_add_ui(
            workspace.next_derivative_real,
            workspace.next_derivative_real,
            1,
            MPFR_RNDN);
        mpfr_mul(
            workspace.next_derivative_imag,
            workspace.zr,
            workspace.derivative_imag,
            MPFR_RNDN);
        mpfr_mul(
            workspace.temporary,
            workspace.zi,
            workspace.derivative_real,
            MPFR_RNDN);
        mpfr_add(
            workspace.next_derivative_imag,
            workspace.next_derivative_imag,
            workspace.temporary,
            MPFR_RNDN);
        mpfr_mul_ui(
            workspace.next_derivative_imag,
            workspace.next_derivative_imag,
            2,
            MPFR_RNDN);

        mpfr_set(workspace.zr, workspace.next_real, MPFR_RNDN);
        mpfr_set(workspace.zi, workspace.next_imag, MPFR_RNDN);
        mpfr_set(
            workspace.derivative_real,
            workspace.next_derivative_real,
            MPFR_RNDN);
        mpfr_set(
            workspace.derivative_imag,
            workspace.next_derivative_imag,
            MPFR_RNDN);

        mpfr_mul(workspace.norm_squared, workspace.zr, workspace.zr, MPFR_RNDN);
        mpfr_mul(workspace.temporary, workspace.zi, workspace.zi, MPFR_RNDN);
        mpfr_add(
            workspace.norm_squared,
            workspace.norm_squared,
            workspace.temporary,
            MPFR_RNDN);
        const ScaledNorm current_norm = sc_norm_squared(
            ScaledComplex::from_float_exp(
                FloatExp::from_mpfr(workspace.zr),
                FloatExp::from_mpfr(workspace.zi)));
        if (mpfr_cmp_d(workspace.norm_squared, escape_squared) > 0) {
            // MPFR computes the logarithm before narrowing to the long-double
            // smooth value, so a 32-bit field still receives the correct
            // fractional escape iteration at a high bailout radius.
            mpfr_log(workspace.temporary, workspace.norm_squared, MPFR_RNDN);
            const long double log_magnitude = 0.5L * mpfr_get_ld(
                workspace.temporary,
                MPFR_RNDN);
            if (!(log_magnitude > 0.0L)
                || !std::isfinite(log_magnitude)) {
                return false;
            }
            const long double smooth = static_cast<long double>(iteration)
                - std::log(log_magnitude) / LOG_TWO;
            output = encode_render_value(smooth, output_bias);
            const ScaledComplex total = ScaledComplex::from_float_exp(
                FloatExp::from_mpfr(workspace.zr),
                FloatExp::from_mpfr(workspace.zi));
            const ScaledComplex derivative = ScaledComplex::from_float_exp(
                FloatExp::from_mpfr(workspace.derivative_real),
                FloatExp::from_mpfr(workspace.derivative_imag));
            store_render_planes_escape(
                planes,
                index,
                kalles_mandelbrot_raw_iteration(iteration),
                total,
                current_norm,
                previous_norm,
                &derivative,
                &pixel_spacing);
            return true;
        }
        previous_norm = current_norm;
    }

    output = encode_render_iteration(max_iter, output_bias);
    clear_render_planes_pixel(planes, index, max_iter);
    return true;
}

#else

void make_reference_orbit(
    ReferenceContext&, const char*, const char*, const char*, int, int,
    int, const char*, const char*, int, int
) {
    throw std::runtime_error("deep rendering requires MPFR/GMP; rebuild with make");
}

bool render_exact_mandelbrot_pixel(
    float&, size_t, const ReferenceContext&, const ScaledComplex&, int,
    double, int, const FloatExp&, FractalRenderPlanes*
) {
    return false;
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
    const FloatExp& tolerance_squared,
    int escape_radius_mode = ESCAPE_RADIUS_MODE_CLASSIC
) {
    const FloatExpComplex error = fec_sub(approximate_delta, exact_delta);
    FloatExp scale = fec_norm_squared(exact_delta);
    if (fe_compare(scale, minimum_scale_squared) < 0) scale = minimum_scale_squared;
    const FloatExp allowed = fe_mul(scale, tolerance_squared);
    if (fe_compare(fec_norm_squared(error), allowed) > 0) return false;

    const bool exact_inside = fe_compare(
        fec_escape_margin_with_delta(
            reference, exact_delta, escape_radius_mode),
        FloatExp{0.0, 0}) <= 0;
    const bool approximate_inside = fe_compare(
        fec_escape_margin_with_delta(
            reference, approximate_delta, escape_radius_mode),
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
    const bool julia = context.formula == FRACTAL_FORMULA_JULIA;
    if (context.formula != FRACTAL_FORMULA_MANDELBROT && !julia) return;
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
    if (julia) {
        // For a Julia frame the pixel offset is the initial-state
        // perturbation: delta_0 = dc, while the parameter remains fixed.
        // This is the same polynomial recurrence as Mandelbrot with the
        // per-step parameter term removed.
        coefficients[1] = {FloatExp::from_parts(1.0, 0), FloatExp{0.0, 0}};
        for (size_t probe = 0; probe < probes.size(); ++probe) {
            exact[probe] = probes[probe];
        }
    }
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
            if (!julia && term == 1) {
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
                fec_add(
                    fec_mul(exact[probe], exact[probe]),
                    julia
                        ? FloatExpComplex{FloatExp{0.0, 0}, FloatExp{0.0, 0}}
                        : dc));
            exact[probe] = exact_next;
            if (iteration + 1 >= order) {
                const FloatExpComplex approximate = series_evaluate(coefficients, dc);
                if (!series_probe_is_safe(
                        builder_orbit[static_cast<size_t>(iteration + 1)],
                        exact_next,
                        approximate,
                        minimum_scale_squared,
                        tolerance_squared,
                        context.escape_radius_mode)) {
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

inline FloatExp alternate_matrix_norm(
    const std::array<FloatExp, 8>& coefficients,
    int offset
) {
    FloatExp sum{0.0, 0};
    sum = fe_add(sum, fe_sqr(coefficients[static_cast<size_t>(offset)]));
    sum = fe_add(sum, fe_sqr(coefficients[static_cast<size_t>(offset + 1)]));
    sum = fe_add(sum, fe_sqr(coefficients[static_cast<size_t>(offset + 2)]));
    sum = fe_add(sum, fe_sqr(coefficients[static_cast<size_t>(offset + 3)]));
    return fe_sqrt(sum);
}

AlternateLinearBlaStep merge_alternate_linear_bla(
    const AlternateLinearBlaStep& y,
    const AlternateLinearBlaStep& x,
    const FloatExp& input_parameter_radius
) {
    const auto& yc = y.coefficients;
    const auto& xc = x.coefficients;
    std::array<FloatExp, 8> coefficients{};
    // M = My * Mx.
    coefficients[0] = fe_add(
        fe_mul(yc[0], xc[0]), fe_mul(yc[1], xc[2]));
    coefficients[1] = fe_add(
        fe_mul(yc[0], xc[1]), fe_mul(yc[1], xc[3]));
    coefficients[2] = fe_add(
        fe_mul(yc[2], xc[0]), fe_mul(yc[3], xc[2]));
    coefficients[3] = fe_add(
        fe_mul(yc[2], xc[1]), fe_mul(yc[3], xc[3]));
    // P = My * Px + Py.
    coefficients[4] = fe_add(
        fe_add(fe_mul(yc[0], xc[4]), fe_mul(yc[1], xc[6])), yc[4]);
    coefficients[5] = fe_add(
        fe_add(fe_mul(yc[0], xc[5]), fe_mul(yc[1], xc[7])), yc[5]);
    coefficients[6] = fe_add(
        fe_add(fe_mul(yc[2], xc[4]), fe_mul(yc[3], xc[6])), yc[6]);
    coefficients[7] = fe_add(
        fe_add(fe_mul(yc[2], xc[5]), fe_mul(yc[3], xc[7])), yc[7]);

    // The input parameter is constant across a block. Reserve enough of the
    // outer map's radius for its response to that parameter, then bound the
    // incoming state with the composed derivative. This is the matrix form
    // of the same conservative radius calculation used by Mandelbrot BLA.
    const FloatExp x_state_norm = alternate_matrix_norm(xc, 0);
    const FloatExp x_parameter_norm = alternate_matrix_norm(xc, 4);
    FloatExp radius = fe_sqrt(x.radius_squared);
    const FloatExp remaining = fe_sub(
        fe_sqrt(y.radius_squared),
        fe_mul(x_parameter_norm, input_parameter_radius));
    if (fe_compare(remaining, FloatExp{0.0, 0}) > 0
        && fe_compare(x_state_norm, FloatExp{0.0, 0}) > 0) {
        radius = fe_min(radius, fe_div(remaining, x_state_norm));
    } else {
        radius = FloatExp{0.0, 0};
    }
    radius = fe_mul(radius, 1.0 - std::ldexp(1.0, -40));
    return {coefficients, fe_sqr(radius), x.length + y.length};
}

inline ScaledComplex apply_alternate_linear_bla(
    const AlternateLinearBlaStep& step,
    const ScaledComplex& delta,
    const ScaledComplex& parameter
) {
    const auto& c = step.coefficients;
    const FloatExp delta_real = sc_component_as_float_exp(delta, false);
    const FloatExp delta_imag = sc_component_as_float_exp(delta, true);
    const FloatExp parameter_real = sc_component_as_float_exp(parameter, false);
    const FloatExp parameter_imag = sc_component_as_float_exp(parameter, true);
    const FloatExp result_real = fe_add(
        fe_add(fe_mul(c[0], delta_real), fe_mul(c[1], delta_imag)),
        fe_add(fe_mul(c[4], parameter_real), fe_mul(c[5], parameter_imag)));
    const FloatExp result_imag = fe_add(
        fe_add(fe_mul(c[2], delta_real), fe_mul(c[3], delta_imag)),
        fe_add(fe_mul(c[6], parameter_real), fe_mul(c[7], parameter_imag)));
    return ScaledComplex::from_float_exp(result_real, result_imag);
}

FloatExp alternate_reference_radius(const FloatExpComplex& reference) {
    const FloatExp magnitude = fe_sqrt(fec_norm_squared(reference));
    const FloatExp one = FloatExp::from_parts(1.0, 0);
    return fe_compare(magnitude, one) > 0 ? magnitude : one;
}

void build_alternate_linear_bla(
    ReferenceContext& context,
    bool retain_builder_orbit = false
) {
    context.alternate_bla.levels.clear();
    context.alternate_bla.levels.reserve(32);
    context.alternate_bla.input_radius = context.bla.input_radius;
    context.alternate_bla.input_radius_squared = fe_sqr(context.bla.input_radius);
    context.alternate_bla.start_index =
        context.formula == FRACTAL_FORMULA_JULIA ? 0 : 1;
    if (context.fast_orbit.size() < 2) return;

    const int first_index = context.alternate_bla.start_index;
    const int last_index = std::min(
        context.bla.map_end,
        static_cast<int>(context.fast_orbit.size()) - 1);
    const int base_count = last_index - first_index;
    if (base_count <= 0) return;

    const FloatExp tolerance = FloatExp::from_long_double(
        std::ldexp(1.0L, -38));
    const FloatExp two = FloatExp::from_parts(2.0, 0);
    const FloatExp quarter = FloatExp::from_parts(0.25, 0);
    auto& base = context.alternate_bla.levels.emplace_back(
        static_cast<size_t>(base_count));
    for (int offset = 0; offset < base_count; ++offset) {
        const FloatExpComplex& reference = context.fast_orbit[
            static_cast<size_t>(first_index + offset)];
        const FloatExp& real = reference.real;
        const FloatExp& imag = reference.imag;
        std::array<FloatExp, 8> coefficients{};
        if (context.formula == FRACTAL_FORMULA_TRICORN) {
            coefficients[0] = fe_mul(real, two);
            coefficients[1] = fe_mul(imag, -2.0);
            coefficients[2] = fe_mul(imag, -2.0);
            coefficients[3] = fe_mul(real, -2.0);
        } else if (context.formula == FRACTAL_FORMULA_BURNING_SHIP) {
            const FloatExp absolute_real = fe_abs(real);
            const FloatExp absolute_imag = fe_abs(imag);
            const double real_sign = real.mantissa < 0.0 ? -1.0 : 1.0;
            const double imag_sign = imag.mantissa < 0.0 ? -1.0 : 1.0;
            coefficients[0] = fe_mul(absolute_real, 2.0 * real_sign);
            coefficients[1] = fe_mul(absolute_imag, -2.0 * imag_sign);
            coefficients[2] = fe_mul(absolute_imag, 2.0 * real_sign);
            coefficients[3] = fe_mul(absolute_real, 2.0 * imag_sign);
        } else {
            coefficients[0] = fe_mul(real, two);
            coefficients[1] = fe_mul(imag, -2.0);
            coefficients[2] = fe_mul(imag, two);
            coefficients[3] = fe_mul(real, two);
        }
        if (context.formula != FRACTAL_FORMULA_JULIA) {
            coefficients[4] = FloatExp::from_parts(1.0, 0);
            coefficients[7] = FloatExp::from_parts(1.0, 0);
        }

        FloatExp radius = fe_mul(
            tolerance,
            alternate_reference_radius(reference));
        if (context.formula == FRACTAL_FORMULA_BURNING_SHIP) {
            // The derivative above is valid only while the perturbation stays
            // on the same side of both absolute-value axes. A zero component
            // deliberately disables multi-step maps for that orbit state;
            // the exact formula-aware step handles the cusp without creating
            // a rectangular interior fill.
            if (real.zero() || imag.zero()) {
                radius = FloatExp{0.0, 0};
            } else {
                radius = fe_min(radius, fe_mul(fe_abs(real), quarter));
                radius = fe_min(radius, fe_mul(fe_abs(imag), quarter));
            }
        }
        base[static_cast<size_t>(offset)] = {
            coefficients,
            fe_sqr(radius),
            1,
        };
    }

    for (size_t level = 1; ; ++level) {
        if ((1ULL << level) > static_cast<unsigned long long>(MAX_SAFE_LINEAR_BLA_LENGTH)) {
            break;
        }
        const auto& previous = context.alternate_bla.levels[level - 1];
        const size_t current_size = (previous.size() + 1) / 2;
        if (current_size == 0) break;
        auto& current = context.alternate_bla.levels.emplace_back(current_size);
        for (size_t index = 0; index < current_size; ++index) {
            const size_t first = index * 2;
            if (first + 1 < previous.size()) {
                current[index] = merge_alternate_linear_bla(
                    previous[first + 1], previous[first],
                    context.alternate_bla.input_radius);
            } else {
                current[index] = previous[first];
            }
        }
    }
    if (!retain_builder_orbit) {
        context.fast_orbit.clear();
        context.fast_orbit.shrink_to_fit();
    }
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
    // boundary, while the validated BLA hierarchy avoids a severe e40--e60
    // throughput cliff when the perturbation enters the ordinary-double
    // range.
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

inline ScaledComplex scaled_formula_step(
    int formula,
    const ScaledComplex& value,
    const ScaledComplex& parameter
) {
    if (formula == FRACTAL_FORMULA_BURNING_SHIP) {
        const ScaledComplex absolute{
            std::abs(value.real), std::abs(value.imag), value.exponent};
        return sc_add(sc_mul(absolute, absolute), parameter);
    }
    const ScaledComplex squared = formula == FRACTAL_FORMULA_TRICORN
        ? sc_mul(sc_conjugate(value), sc_conjugate(value))
        : sc_mul(value, value);
    return sc_add(squared, parameter);
}

inline ScaledComplex alternate_delta_step(
    const ReferenceContext& context,
    int reference_index,
    const ScaledComplex& delta,
    const ScaledComplex& parameter_delta
) {
    const int formula = context.formula;
    const ScaledComplex& reference =
        context.orbit->scaled[static_cast<size_t>(reference_index)];
    if (formula == FRACTAL_FORMULA_TRICORN) {
        const ScaledComplex conjugate_reference = sc_conjugate(reference);
        const ScaledComplex conjugate_delta = sc_conjugate(delta);
        return sc_add(
            sc_double(sc_mul(conjugate_reference, conjugate_delta)),
            sc_add(sc_mul(conjugate_delta, conjugate_delta), parameter_delta));
    }
    if (formula == FRACTAL_FORMULA_BURNING_SHIP) {
        // Away from an axis the absolute-value map has a stable real 2x2
        // derivative. Keeping this as FloatExp components preserves a tiny
        // perturbation that would disappear if we reconstructed
        // |Z + delta| and subtracted the rounded reference orbit.
        const FloatExp reference_real = sc_component_as_float_exp(reference, false);
        const FloatExp reference_imag = sc_component_as_float_exp(reference, true);
        const FloatExp delta_real = sc_component_as_float_exp(delta, false);
        const FloatExp delta_imag = sc_component_as_float_exp(delta, true);
        const FloatExp absolute_real = fe_abs(reference_real);
        const FloatExp absolute_imag = fe_abs(reference_imag);
        // Match the piecewise map's actual sign transition, rather than
        // treating |delta| >= |reference| as a crossing.  A perturbation can
        // be larger than the reference while remaining on the same side of
        // an axis; forcing the exact fallback in that case introduces a
        // projected-centre subtraction on every step and slowly smears the
        // deep field.  The FloatExp sign test also handles reference==0 with
        // the same >= 0 convention as the Python reference path.
        const FloatExp actual_real = fe_add(reference_real, delta_real);
        const FloatExp actual_imag = fe_add(reference_imag, delta_imag);
        const bool crosses_real =
            (fe_compare(reference_real, FloatExp{0.0, 0}) >= 0)
            != (fe_compare(actual_real, FloatExp{0.0, 0}) >= 0);
        const bool crosses_imag =
            (fe_compare(reference_imag, FloatExp{0.0, 0}) >= 0)
            != (fe_compare(actual_imag, FloatExp{0.0, 0}) >= 0);
        if (crosses_real || crosses_imag) {
            // Do not form reference + delta as ScaledComplex here.  Its
            // aligned add intentionally drops terms more than 60 binary
            // exponents below the reference, which is exactly what happens
            // when a deep pixel crosses an absolute-value axis after a long
            // amplification.  FloatExp components retain the crossing
            // perturbation, and the projected reference/parameter terms are
            const FloatExp absolute_total_real = fe_abs(actual_real);
            const FloatExp absolute_total_imag = fe_abs(actual_imag);
            const ScaledComplex& next_reference =
                context.orbit->scaled[static_cast<size_t>(reference_index + 1)];
            const FloatExp parameter_offset_real = fe_sub(
                sc_component_as_float_exp(context.parameter, false),
                sc_component_as_float_exp(next_reference, false));
            const FloatExp parameter_offset_imag = fe_sub(
                sc_component_as_float_exp(context.parameter, true),
                sc_component_as_float_exp(next_reference, true));
            const FloatExp next_real = fe_add(
                fe_sub(
                    fe_sqr(absolute_total_real),
                    fe_sqr(absolute_total_imag)),
                fe_add(
                    parameter_offset_real,
                    sc_component_as_float_exp(parameter_delta, false)));
            const FloatExp next_imag = fe_add(
                fe_mul(
                    fe_mul(absolute_total_real, absolute_total_imag),
                    2.0),
                fe_add(
                    parameter_offset_imag,
                    sc_component_as_float_exp(parameter_delta, true)));
            return ScaledComplex::from_float_exp(
                next_real,
                next_imag);
        }
        const double real_sign = reference_real.mantissa < 0.0 ? -1.0 : 1.0;
        const double imag_sign = reference_imag.mantissa < 0.0 ? -1.0 : 1.0;
        const FloatExp real_linear = fe_sub(
            fe_mul(fe_mul(absolute_real, delta_real), 2.0 * real_sign),
            fe_mul(fe_mul(absolute_imag, delta_imag), 2.0 * imag_sign));
        const FloatExp imag_linear = fe_add(
            fe_mul(fe_mul(absolute_real, delta_imag), 2.0 * imag_sign),
            fe_mul(fe_mul(absolute_imag, delta_real), 2.0 * real_sign));
        const FloatExp delta_square_real = fe_sub(
            fe_sqr(delta_real), fe_sqr(delta_imag));
        const FloatExp delta_square_imag = fe_mul(
            fe_mul(delta_real, delta_imag), 2.0 * real_sign * imag_sign);
        return ScaledComplex::from_float_exp(
            fe_add(fe_add(real_linear, delta_square_real),
                   sc_component_as_float_exp(parameter_delta, false)),
            fe_add(fe_add(imag_linear, delta_square_imag),
                   sc_component_as_float_exp(parameter_delta, true)));
    }
    // Mandelbrot and Julia are holomorphic z² maps. Julia keeps c fixed, so
    // its parameter delta is zero and the incoming pixel offset is the
    // initial-state perturbation instead.
    return sc_add(
        sc_double(sc_mul(reference, delta)),
        sc_add(sc_mul(delta, delta), parameter_delta));
}

inline FloatExp alternate_escape_margin(
    const ReferenceContext& context,
    int reference_index,
    const ScaledComplex& delta
) {
    const auto& orbit = *context.orbit;
    if (reference_index >= 0
        && reference_index < static_cast<int>(orbit.escape_margin.size())) {
        return sc_escape_margin_with_reference_margin(
            orbit.escape_margin[static_cast<size_t>(reference_index)],
            orbit.scaled[static_cast<size_t>(reference_index)],
            delta);
    }
    return sc_escape_margin_with_delta(
        orbit.scaled[static_cast<size_t>(reference_index)],
        delta,
        context.escape_radius_mode);
}

inline ScaledNorm norm_from_escape_margin(
    const FloatExp& margin,
    int escape_radius_mode = ESCAPE_RADIUS_MODE_CLASSIC
) {
    const FloatExp norm = fe_add(
        escape_radius_squared_float_exp(escape_radius_mode), margin);
    return {norm.mantissa, norm.exponent};
}

void render_alternate_reference_impl(
    float* __restrict output,
    int width,
    int height,
    const char* zoom_text,
    const ReferenceContext& context,
    int max_iter,
    int threads,
    const FractalRenderOptions& options,
    RenderStats* stats_out,
    const std::vector<ScaledComplex>* point_offsets = nullptr,
    const FloatExp* point_radius = nullptr,
    FractalRenderPlanes* planes = nullptr
) {
    if (!context.orbit || context.orbit->scaled.size() < 2) {
        throw std::runtime_error("alternate reference orbit is incomplete");
    }
    const auto render_started = std::chrono::steady_clock::now();
    const int escape_radius_mode = options.escape_radius_mode;
    const FloatExp zoom = parse_zoom_float_exp(zoom_text, context.precision_bits);
    const FloatExp view_height = fe_mul(
        fe_div(FloatExp::from_parts(1.0, 0), zoom),
        viewport_height_factor(options.coordinate_mode));
    const FloatExp view_width = fe_mul(
        view_height, static_cast<double>(width) / static_cast<double>(height));
    const FloatExp pixel_spacing = fe_div(
        view_height,
        FloatExp::from_parts(static_cast<double>(height), 0));
    std::vector<FloatExp> x_offsets;
    std::vector<FloatExp> y_offsets;
    if (point_offsets == nullptr) {
        x_offsets.resize(static_cast<size_t>(width));
        y_offsets.resize(static_cast<size_t>(height));
        for (int px = 0; px < width; ++px) {
            const double fraction =
                pixel_axis_offset(px, width, options.coordinate_mode)
                / static_cast<double>(width);
            x_offsets[static_cast<size_t>(px)] = fe_mul(view_width, fraction);
        }
        for (int py = 0; py < height; ++py) {
            const double fraction =
                -pixel_axis_offset(py, height, options.coordinate_mode)
                / static_cast<double>(height);
            y_offsets[static_cast<size_t>(py)] = fe_mul(view_height, fraction);
        }
    } else if (point_offsets->size() != static_cast<size_t>(width * height)) {
        throw std::runtime_error("alternate point renderer dimensions do not match offset count");
    }

    RenderTimeBudget render_budget;
    if (options.time_budget_ms > 0) {
        render_budget.enabled = true;
        render_budget.deadline = std::chrono::steady_clock::now()
            + std::chrono::milliseconds(options.time_budget_ms);
    }
    RenderTimeBudget* time_budget = render_budget.enabled ? &render_budget : nullptr;
    const bool cycle_detection_enabled = options.disable_cycle == 0
        && (options.strict == 0 || options.strict_cycle != 0);
    const bool julia = context.formula == FRACTAL_FORMULA_JULIA;
    const bool image_series_available = julia
        && context.image_series.enabled
        && context.image_series.iteration < max_iter
        && (point_radius == nullptr
            || fe_compare(
                *point_radius,
                fe_sqrt(context.image_series.radius_squared)) <= 0);
    const bool alternate_bla_enabled = options.disable_bla == 0
        && !context.alternate_bla.levels.empty();
    const int max_alternate_block_length = std::clamp(
        options.max_linear_bla_length,
        2,
        MAX_SAFE_LINEAR_BLA_LENGTH);

#ifdef _OPENMP
    if (threads > 0) {
        omp_set_dynamic(0);
        omp_set_num_threads(threads);
    }
#pragma omp parallel for schedule(dynamic, 256)
#endif
    for (int linear_pixel = 0; linear_pixel < width * height; ++linear_pixel) {
        const int py = linear_pixel / width;
        const int px = linear_pixel - py * width;
        const int index = py * width + px;
        clear_render_planes_pixel(planes, static_cast<size_t>(index), max_iter);
        const ScaledComplex dc = point_offsets != nullptr
            ? (*point_offsets)[static_cast<size_t>(linear_pixel)]
            : ScaledComplex::from_float_exp(
                x_offsets[static_cast<size_t>(px)],
                y_offsets[static_cast<size_t>(py)]);
        const ScaledComplex parameter_delta = julia ? ScaledComplex{} : dc;
        const ScaledNorm parameter_norm = sc_norm_squared(parameter_delta);
        const FloatExp parameter_norm_squared{
            parameter_norm.mantissa,
            parameter_norm.exponent,
        };
        ScaledComplex delta = dc;
        AlternateJacobian jacobian = alternate_identity_jacobian();
        int reference_index = julia ? 0 : 1;
        int iteration = julia ? 0 : 1;
        if (image_series_available
            && sc_compare_norm(
                sc_norm_squared(dc),
                ScaledNorm{
                    context.image_series.radius_squared.mantissa,
                    context.image_series.radius_squared.exponent,
                }) <= 0) {
            ScaledComplex series_derivative{};
            delta = evaluate_image_series_with_derivative(
                context.image_series,
                dc,
                series_derivative);
            jacobian = alternate_holomorphic_jacobian(series_derivative);
            reference_index = context.image_series.iteration;
            iteration = context.image_series.iteration;
        }
        bool escaped = false;
        bool deadline_abort = false;
        bool unresolved_pixel = false;
        std::uint32_t budget_ticks = 0;
        ScaledComplex total = sc_add(
            context.orbit->scaled[static_cast<size_t>(reference_index)], delta);
        ScaledNorm total_norm = norm_from_escape_margin(
            alternate_escape_margin(context, reference_index, delta),
            escape_radius_mode);
        ScaledNorm palette_norm = planes != nullptr
            ? sc_norm_squared(total)
            : ScaledNorm{};

        const auto store_alternate_escape = [&](int escaped_iteration,
                                                const ScaledComplex& escaped_total,
                                                const ScaledNorm& previous_norm) {
            if (planes == nullptr) return;
            const ScaledNorm current_norm = sc_norm_squared(escaped_total);
            // Julia starts by testing the pixel's initial z at iteration 0;
            // the other alternate parameter-plane formulas start at z_1,
            // just like the deep Mandelbrot path.
            const int raw_iteration = julia
                ? escaped_iteration
                : kalles_mandelbrot_raw_iteration(escaped_iteration);
            store_render_planes_escape_jacobian(
                planes,
                static_cast<size_t>(index),
                raw_iteration,
                escaped_total,
                current_norm,
                previous_norm,
                jacobian,
                pixel_spacing);
        };

        if (sc_outside_escape(total_norm, escape_radius_mode)) {
            output[index] = smooth_escape_scaled(iteration, total_norm, options.output_bias);
            store_alternate_escape(iteration, total, ScaledNorm{});
            continue;
        }

        ScaledComplex cycle_tortoise{};
        int cycle_power = 1;
        int cycle_length = 0;
        bool cycle_ready = false;
        int cycle_hits = 0;
        while (iteration < max_iter) {
            if (render_time_budget_expired(time_budget, budget_ticks)) {
                output[index] = std::numeric_limits<float>::quiet_NaN();
                deadline_abort = true;
                unresolved_pixel = true;
                break;
            }

            const int next_index = reference_index + 1;
            if (next_index >= static_cast<int>(context.orbit->scaled.size())) {
                // A reference that escaped before max_iter cannot be extended
                // with the compact perturbation relation. Continue from the
                // reconstructed state using the exact scaled recurrence; this
                // branch is rare for boundary references and preserves a
                // finite answer instead of painting the remainder as inside.
                const ScaledComplex parameter = julia
                    ? context.parameter
                    : sc_add(context.parameter, dc);
                const ScaledNorm previous_palette_norm = palette_norm;
                const ScaledComplex previous_total = total;
                jacobian = alternate_jacobian_step(
                    context.formula,
                    previous_total,
                    jacobian,
                    !julia);
                total = scaled_formula_step(context.formula, total, parameter);
                ++iteration;
                total_norm = sc_norm_squared(total);
                if (sc_outside_escape(total_norm, escape_radius_mode)) {
                    output[index] = smooth_escape_scaled(iteration, total_norm, options.output_bias);
                    store_alternate_escape(iteration, total, previous_palette_norm);
                    escaped = true;
                    break;
                }
                if (planes != nullptr) palette_norm = sc_norm_squared(total);
                continue;
            }

            const ScaledNorm delta_norm = sc_norm_squared(delta);
            const AlternateLinearBlaStep* linear_step = alternate_bla_enabled
                ? context.alternate_bla.lookup(
                    reference_index,
                    FloatExp{delta_norm.mantissa, delta_norm.exponent},
                    parameter_norm_squared,
                    std::min(max_alternate_block_length, max_iter - iteration))
                : nullptr;
            if (linear_step != nullptr
                && reference_index + linear_step->length
                    < static_cast<int>(context.orbit->scaled.size())) {
                const ScaledComplex previous_delta = delta;
                const AlternateJacobian previous_jacobian = jacobian;
                const int previous_reference_index = reference_index;
                const int previous_iteration = iteration;
                const int endpoint_index = reference_index + linear_step->length;
                const ScaledComplex candidate_delta = apply_alternate_linear_bla(
                    *linear_step, delta, parameter_delta);
                const FloatExp candidate_margin = alternate_escape_margin(
                    context, endpoint_index, candidate_delta);
                const ScaledNorm candidate_norm = norm_from_escape_margin(
                    candidate_margin, escape_radius_mode);
                const ScaledComplex candidate_total = sc_add(
                    context.orbit->scaled[static_cast<size_t>(endpoint_index)],
                    candidate_delta);
                const bool candidate_bad = !sc_finite(candidate_delta)
                    || !candidate_margin.finite()
                    || !sc_finite(candidate_total)
                    || fe_compare(candidate_margin, FloatExp{0.0, 0}) > 0
                    || sc_compare_norm(candidate_norm, ScaledNorm{0.75, 2}) >= 0;
                if (!candidate_bad) {
                    delta = candidate_delta;
                    jacobian = alternate_linear_bla_jacobian(
                        linear_step->coefficients,
                        jacobian,
                        !julia);
                    reference_index = endpoint_index;
                    iteration = previous_iteration + linear_step->length;
                    total = candidate_total;
                    total_norm = candidate_norm;
                    if (planes != nullptr) palette_norm = sc_norm_squared(total);
                    continue;
                }

                // A linear block is a throughput optimization, never a
                // classification shortcut. Replay a block that approaches
                // an escape boundary or leaves its conservative domain one
                // formula-aware step at a time. This is also where Burning
                // Ship axis crossings are handled exactly.
                delta = previous_delta;
                jacobian = previous_jacobian;
                reference_index = previous_reference_index;
                iteration = previous_iteration;
                for (int replay = 0;
                     replay < linear_step->length && iteration < max_iter;
                     ++replay) {
                    const ScaledNorm previous_replay_norm = palette_norm;
                    const ScaledComplex previous_total = sc_add(
                        context.orbit->scaled[static_cast<size_t>(reference_index)],
                        delta);
                    jacobian = alternate_jacobian_step(
                        context.formula,
                        previous_total,
                        jacobian,
                        !julia);
                    delta = alternate_delta_step(
                        context, reference_index, delta, parameter_delta);
                    ++reference_index;
                    ++iteration;
                    if (!sc_finite(delta)) {
                        output[index] = options.strict
                            ? std::numeric_limits<float>::quiet_NaN()
                            : encode_render_iteration(iteration, options.output_bias);
                        unresolved_pixel = true;
                        break;
                    }
                    const FloatExp replay_margin = alternate_escape_margin(
                        context, reference_index, delta);
                    total_norm = norm_from_escape_margin(
                        replay_margin, escape_radius_mode);
                    total = sc_add(
                        context.orbit->scaled[static_cast<size_t>(reference_index)],
                        delta);
                    if (fe_compare(replay_margin, FloatExp{0.0, 0}) > 0
                        || sc_outside_escape(total_norm, escape_radius_mode)) {
                        output[index] = smooth_escape_scaled(iteration, total_norm, options.output_bias);
                        store_alternate_escape(iteration, total, previous_replay_norm);
                        escaped = true;
                        break;
                    }
                    if (planes != nullptr) palette_norm = sc_norm_squared(total);
                }
                if (escaped || unresolved_pixel) break;
                continue;
            }

            const ScaledNorm previous_palette_norm = palette_norm;
            const ScaledComplex previous_total = sc_add(
                context.orbit->scaled[static_cast<size_t>(reference_index)],
                delta);
            jacobian = alternate_jacobian_step(
                context.formula,
                previous_total,
                jacobian,
                !julia);
            delta = alternate_delta_step(
                context, reference_index, delta, parameter_delta);
            reference_index = next_index;
            ++iteration;
            if (!sc_finite(delta)) {
                output[index] = options.strict
                    ? std::numeric_limits<float>::quiet_NaN()
                    : encode_render_iteration(iteration, options.output_bias);
                unresolved_pixel = true;
                break;
            }
            const FloatExp margin = alternate_escape_margin(
                context, reference_index, delta);
            total_norm = norm_from_escape_margin(margin, escape_radius_mode);
            total = sc_add(
                context.orbit->scaled[static_cast<size_t>(reference_index)], delta);
            if (fe_compare(margin, FloatExp{0.0, 0}) > 0
                || sc_outside_escape(total_norm, escape_radius_mode)) {
                output[index] = smooth_escape_scaled(iteration, total_norm, options.output_bias);
                store_alternate_escape(iteration, total, previous_palette_norm);
                escaped = true;
                break;
            }
            if (planes != nullptr) palette_norm = sc_norm_squared(total);

            // Confirm a bounded cycle several times before declaring the
            // pixel interior. The perturbation is included in the comparison,
            // so a repelling fixed point cannot look periodic merely because
            // the rounded reference state repeats.
            if (cycle_detection_enabled
                && iteration >= 2048
                && (iteration & 31) == 0
                && sc_compare_norm(total_norm, ScaledNorm{0.75, 2}) < 0) {
                if (!cycle_ready) {
                    cycle_tortoise = total;
                    cycle_power = 1;
                    cycle_length = 0;
                    cycle_hits = 0;
                    cycle_ready = true;
                } else {
                    const ScaledNorm distance = sc_norm_squared(
                        sc_sub(total, cycle_tortoise));
                    const int scale_exponent = std::max(total_norm.exponent, 0);
                    if (distance.mantissa == 0.0
                        || distance.exponent <= scale_exponent - 78) {
                        ++cycle_hits;
                        if (cycle_hits >= 3) {
                            output[index] = encode_render_iteration(max_iter, options.output_bias);
                            clear_render_planes_pixel(
                                planes,
                                static_cast<size_t>(index),
                                max_iter);
                            escaped = true;
                            break;
                        }
                    } else {
                        cycle_hits = 0;
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
        if (!escaped && !deadline_abort && !unresolved_pixel) {
            output[index] = encode_render_iteration(max_iter, options.output_bias);
            clear_render_planes_pixel(planes, static_cast<size_t>(index), max_iter);
        }
    }
    if (stats_out != nullptr) {
        stats_out->pixels = static_cast<std::uint64_t>(width)
            * static_cast<std::uint64_t>(height);
        stats_out->render_ns = static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - render_started).count());
    }
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
    ScaledComplex& derivative,
    bool disable_cycle_detection,
    bool strict_cycle_detection,
    bool strict_render,
    double output_bias,
    int escape_radius_mode,
    RenderTimeBudget* time_budget,
    std::uint32_t& budget_ticks,
    bool& deadline_abort,
    bool& unresolved_tail,
    RenderStats* stats,
    FractalRenderPlanes* planes = nullptr,
    size_t plane_index = 0,
    const ScaledNorm* initial_norm = nullptr,
    const FloatExp* pixel_spacing = nullptr
) {
    constexpr int MAX_TAIL_REBASES = 64;
    // A normal 960x540 probe at this location already needs about 4.6k
    // double-tail steps for its slowest pixel. Keep the emergency ceiling
    // above the supported video iteration range so ordinary tails do not
    // take the slower restart path.
    constexpr int MAX_TAIL_STEPS = 65536;
    const double dc_real = sc_to_double(dc);
    const double dc_imag = std::ldexp(dc.imag, dc.exponent);
    const double escape_squared = escape_radius_squared_double(escape_radius_mode);
    const double safe_escape_squared = escape_squared + 1.0e-7;
    double delta_real = sc_to_double(delta);
    double delta_imag = std::ldexp(delta.imag, delta.exponent);
    output = encode_render_iteration(max_iter, output_bias);
    double tortoise_real = 0.0;
    double tortoise_imag = 0.0;
    int cycle_power = 1;
    int cycle_length = 0;
    int tail_steps = 0;
    int tail_iterations = 0;
    int tail_rebases = 0;
    ScaledNorm previous_norm = initial_norm != nullptr
        ? *initial_norm : ScaledNorm{};
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
        // The double tail still needs the same derivative recurrence as the
        // scaled perturbation path.  Build the current total before taking
        // the next step; d(z²+c)/dc = 2*z*dz/dc + 1.
        ScaledComplex prior_total{
            reference_real + delta_real,
            reference_imag + delta_imag,
            0,
        };
        prior_total.normalize();
        derivative = sc_add(
            sc_double(sc_mul(prior_total, derivative)),
            ScaledComplex{1.0, 0.0, 0});
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
                : encode_render_iteration(iteration, output_bias);
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
                : encode_render_iteration(iteration, output_bias);
            return false;
        }
        if (magnitude_squared > escape_squared) {
            const double magnitude = std::sqrt(
                std::max(magnitude_squared, safe_escape_squared));
            output = encode_render_value(
                static_cast<long double>(iteration)
                    - std::log(std::log(magnitude)) / static_cast<double>(LOG_TWO),
                output_bias);
            ScaledComplex escaped_total{total_real, total_imag, 0};
            escaped_total.normalize();
            const ScaledNorm current_norm = scaled_norm_from_double(magnitude_squared);
            store_render_planes_escape(
                planes,
                plane_index,
                kalles_mandelbrot_raw_iteration(iteration),
                escaped_total,
                current_norm,
                previous_norm,
                &derivative,
                pixel_spacing);
            return false;
        }
        previous_norm = scaled_norm_from_double(magnitude_squared);
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
                    clear_render_planes_pixel(planes, plane_index, max_iter);
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
                : encode_render_iteration(iteration, output_bias);
            clear_render_planes_pixel(planes, plane_index, max_iter);
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
            : encode_render_iteration(iteration, output_bias);
        clear_render_planes_pixel(planes, plane_index, max_iter);
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
    const FloatExp* point_radius = nullptr,
    FractalRenderPlanes* planes = nullptr
) {
    const auto render_started = std::chrono::steady_clock::now();
    const int escape_radius_mode = options.escape_radius_mode;
    const FloatExp zoom = parse_zoom_float_exp(zoom_text, context.precision_bits);
    const FloatExp inverse_zoom = fe_div(FloatExp::from_parts(1.0, 0), zoom);
    const FloatExp view_height = fe_mul(
        inverse_zoom,
        viewport_height_factor(options.coordinate_mode));
    const FloatExp view_width = fe_mul(
        view_height, static_cast<double>(width) / static_cast<double>(height));
    const FloatExp pixel_spacing = fe_div(
        view_height,
        FloatExp::from_parts(static_cast<double>(height), 0));
    const FloatExp current_input_radius = point_radius != nullptr
        ? *point_radius
        : fe_mul(
            inverse_zoom,
            viewport_height_factor(options.coordinate_mode));
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
                -pixel_axis_offset(py, height, options.coordinate_mode)
                / static_cast<double>(height);
            y_offsets[static_cast<size_t>(py)] = fe_mul(view_height, y_fraction);
        }
        for (int px = 0; px < width; ++px) {
            const double x_fraction =
                pixel_axis_offset(px, width, options.coordinate_mode)
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
    // one row take orders of magnitude longer than its neighbours. A row is
    // therefore too coarse a unit of work. Use moderately sized contiguous
    // pixel chunks so hard regions are still redistributed while the OpenMP
    // scheduler does not perform an atomic work assignment for every few
    // dozen pixels in a multi-million-pixel 4K tile.
#pragma omp parallel for schedule(dynamic, 256)
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
            clear_render_planes_pixel(planes, static_cast<size_t>(index), max_iter);
            ScaledComplex delta = dc;
            // Mandelbrot parameter-plane derivative dz/dc at z_1=c is one.
            // It is carried in the same scaled representation as delta so a
            // Kalles analytic-DE palette remains useful at e100 and beyond.
            ScaledComplex derivative{1.0, 0.0, 0};
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
                delta = evaluate_image_series_with_derivative(
                    context.image_series,
                    dc,
                    derivative);
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
            ScaledNorm previous_total_norm{};
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
                    output[index] = encode_render_iteration(iteration, options.output_bias);
                    store_render_planes_escape(
                        planes,
                        static_cast<size_t>(index),
                        kalles_mandelbrot_raw_iteration(iteration),
                        total,
                        total_norm,
                        previous_total_norm,
                        &derivative,
                        &pixel_spacing);
                    escaped = true;
                    break;
                }
                if (!have_total) {
                    total = sc_add(
                        context.orbit->scaled[static_cast<size_t>(reference_index)], delta);
                    total_norm = sc_norm_squared_with_delta(
                        context.orbit->scaled[static_cast<size_t>(reference_index)],
                        delta,
                        escape_radius_mode);
                }
                have_total = false;
                // A compact scaled reference keeps only a double mantissa,
                // so cancellation between the reference and perturbation can
                // lose the low bits that decide the orbit. Kalles marks this
                // condition as a perturbation glitch when the reconstructed
                // orbit is much smaller than the reference state, then asks
                // for another reference. Do the same here instead of turning
                // the first bad escape into a rectangular fill.
                const ScaledComplex& reference_state =
                    context.orbit->scaled[static_cast<size_t>(reference_index)];
                const ScaledNorm reference_norm_value = sc_norm_squared(reference_state);
                const FloatExp reference_norm = FloatExp{
                    reference_norm_value.mantissa,
                    reference_norm_value.exponent,
                };
                if (reference_norm.mantissa != 0.0
                    && reference_norm.finite()
                    && fe_compare(
                        FloatExp{total_norm.mantissa, total_norm.exponent},
                        fe_mul(reference_norm, 1.0e-7)) < 0) {
                    if constexpr (CollectStats) ++stats->glitch_count;
                    if (point_offsets != nullptr
                        && options.strict != 0
                        && static_cast<std::size_t>(width)
                            * static_cast<std::size_t>(height)
                            <= MAX_EXACT_POINT_REPAIR_PIXELS
                        && render_exact_mandelbrot_pixel(
                            output[index],
                            static_cast<size_t>(index),
                            context,
                            dc,
                            max_iter,
                            options.output_bias,
                            escape_radius_mode,
                            pixel_spacing,
                            planes)) {
                        // The compact path found a genuine cancellation, but
                        // the MPFR rebase produced the exact pixel and its
                        // KFP metadata. Treat it as complete so the Python
                        // atlas does not split a correct single-pixel repair
                        // into a much slower reference tree.
                        escaped = true;
                        break;
                    }
                    // The compact reference has lost enough low bits for a
                    // direct subtraction to be unsafe.  Live/draft callers
                    // explicitly allow recovery, so rebase from the already
                    // reconstructed total instead of returning a NaN that
                    // forces the Python layer into a full secondary-reference
                    // tree.  Strict export calls keep the old sentinel and
                    // are repaired by the quality-preserving atlas path.
                    if (options.allow_recovery != 0
                        && sc_finite(total)
                        && ++perturbation_rebases <= 64) {
                        delta = total;
                        reference_index = 0;
                        have_total = false;
                        continue;
                    }
                    output[index] = std::numeric_limits<float>::quiet_NaN();
                    unresolved_pixel = true;
                    if constexpr (CollectStats) {
                        ++stats->unresolved_pixels;
                    }
                    break;
                }
                if (sc_outside_escape(total_norm, escape_radius_mode)
                    || sc_outside_escape_with_delta(
                        reference_state,
                        delta,
                        escape_radius_mode)) {
                    output[index] = smooth_escape_scaled(iteration, total_norm, options.output_bias);
                    store_render_planes_escape(
                        planes,
                        static_cast<size_t>(index),
                        kalles_mandelbrot_raw_iteration(iteration),
                        total,
                        total_norm,
                        previous_total_norm,
                        &derivative,
                        &pixel_spacing);
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
                                output[index] = encode_render_iteration(max_iter, options.output_bias);
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
                        ScaledComplex exact_cycle_tortoise{};
                        int exact_cycle_power = 1;
                        int exact_cycle_length = 0;
                        bool exact_cycle_ready = false;
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
                            const ScaledNorm prior_norm = total_norm;
                            derivative = sc_add(
                                sc_double(sc_mul(total, derivative)),
                                ScaledComplex{1.0, 0.0, 0});
                            total = sc_add(sc_mul(total, total), parameter);
                            ++iteration;
                            if constexpr (CollectStats) {
                                ++stats->logical_iterations;
                                ++stats->exact_steps;
                            }
                            total_norm = sc_norm_squared(total);
                            if (sc_outside_escape(total_norm, escape_radius_mode)) {
                                output[index] = smooth_escape_scaled(iteration, total_norm, options.output_bias);
                                store_render_planes_escape(
                                    planes,
                                    static_cast<size_t>(index),
                                    kalles_mandelbrot_raw_iteration(iteration),
                                    total,
                                    total_norm,
                                    prior_norm,
                                    &derivative,
                                    &pixel_spacing);
                                escaped = true;
                                break;
                            }
                            previous_total_norm = prior_norm;
                            // Once a pixel has left the compact reference
                            // orbit, it still needs an interior exit. Without
                            // this check the live path iterates every settled
                            // pixel all the way to its draft cap, even though
                            // the surrounding perturbation/BLA path already
                            // has a conservative cycle detector. Strict
                            // exports keep cycle_detection_enabled false.
                            if (cycle_detection_enabled
                                && iteration >= cycle_minimum_iteration
                                && (iteration & 31) == 0
                                && sc_compare_norm(total_norm, ScaledNorm{0.75, 2}) < 0) {
                                if (!exact_cycle_ready) {
                                    exact_cycle_tortoise = total;
                                    exact_cycle_power = 1;
                                    exact_cycle_length = 0;
                                    exact_cycle_ready = true;
                                } else {
                                    const ScaledNorm exact_cycle_distance = sc_norm_squared(
                                        sc_sub(total, exact_cycle_tortoise));
                                    const int scale_exponent = std::max(total_norm.exponent, 0);
                                    if (exact_cycle_distance.mantissa == 0.0
                                        || exact_cycle_distance.exponent
                                            <= scale_exponent - cycle_exponent_margin) {
                                        output[index] = encode_render_iteration(
                                            max_iter,
                                            options.output_bias);
                                        escaped = true;
                                        break;
                                    }
                                    ++exact_cycle_length;
                                    if (exact_cycle_length >= exact_cycle_power) {
                                        exact_cycle_tortoise = total;
                                        exact_cycle_power = std::min(
                                            exact_cycle_power * 2,
                                            1 << 20);
                                        exact_cycle_length = 0;
                                    }
                                }
                            }
                        }
                        if (deadline_abort) break;
                        if (!escaped) {
                            output[index] = encode_render_iteration(max_iter, options.output_bias);
                        }
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
                    ScaledComplex exact_cycle_tortoise{};
                    int exact_cycle_power = 1;
                    int exact_cycle_length = 0;
                    bool exact_cycle_ready = false;
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
                        const ScaledNorm prior_norm = total_norm;
                        derivative = sc_add(
                            sc_double(sc_mul(total, derivative)),
                            ScaledComplex{1.0, 0.0, 0});
                        total = sc_add(
                            sc_mul(total, total),
                            parameter);
                        ++iteration;
                        if constexpr (CollectStats) {
                            ++stats->logical_iterations;
                            ++stats->exact_steps;
                        }
                        total_norm = sc_norm_squared(total);
                        if (sc_outside_escape(total_norm, escape_radius_mode)) {
                            output[index] = smooth_escape_scaled(iteration, total_norm, options.output_bias);
                            store_render_planes_escape(
                                planes,
                                static_cast<size_t>(index),
                                kalles_mandelbrot_raw_iteration(iteration),
                                total,
                                total_norm,
                                prior_norm,
                                &derivative,
                                &pixel_spacing);
                            escaped = true;
                            break;
                        }
                        previous_total_norm = prior_norm;
                        if (cycle_detection_enabled
                            && iteration >= cycle_minimum_iteration
                            && (iteration & 31) == 0
                            && sc_compare_norm(total_norm, ScaledNorm{0.75, 2}) < 0) {
                            if (!exact_cycle_ready) {
                                exact_cycle_tortoise = total;
                                exact_cycle_power = 1;
                                exact_cycle_length = 0;
                                exact_cycle_ready = true;
                            } else {
                                const ScaledNorm exact_cycle_distance = sc_norm_squared(
                                    sc_sub(total, exact_cycle_tortoise));
                                const int scale_exponent = std::max(total_norm.exponent, 0);
                                if (exact_cycle_distance.mantissa == 0.0
                                    || exact_cycle_distance.exponent
                                        <= scale_exponent - cycle_exponent_margin) {
                                    output[index] = encode_render_iteration(
                                        max_iter,
                                        options.output_bias);
                                    escaped = true;
                                    break;
                                }
                                ++exact_cycle_length;
                                if (exact_cycle_length >= exact_cycle_power) {
                                    exact_cycle_tortoise = total;
                                    exact_cycle_power = std::min(
                                        exact_cycle_power * 2,
                                        1 << 20);
                                    exact_cycle_length = 0;
                                }
                            }
                        }
                    }
                    if (deadline_abort) break;
                    if (!escaped) {
                        output[index] = encode_render_iteration(max_iter, options.output_bias);
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
                                derivative,
                                !cycle_detection_enabled, strict_cycle_detection,
                                options.strict != 0,
                                options.output_bias,
                                escape_radius_mode,
                                time_budget, budget_ticks, deadline_abort,
                                unresolved_pixel,
                                stats,
                                planes,
                                static_cast<size_t>(index),
                                &total_norm,
                                &pixel_spacing)
                            : render_scaled_double_tail<CollectStats, false>(
                                output[index], dc, context, max_iter,
                                iteration, reference_index, delta,
                                derivative,
                                !cycle_detection_enabled, strict_cycle_detection,
                                options.strict != 0,
                                options.output_bias,
                                escape_radius_mode,
                                time_budget, budget_ticks, deadline_abort,
                                unresolved_pixel,
                                stats,
                                planes,
                                static_cast<size_t>(index),
                                &total_norm,
                                &pixel_spacing);
                    } else {
                        tail_pathological = render_scaled_double_tail<CollectStats, false>(
                            output[index], dc, context, max_iter,
                            iteration, reference_index, delta,
                            derivative,
                            true, strict_cycle_detection,
                            options.strict != 0,
                            options.output_bias,
                            escape_radius_mode,
                            time_budget, budget_ticks, deadline_abort,
                            unresolved_pixel,
                            stats,
                            planes,
                            static_cast<size_t>(index),
                            &total_norm,
                            &pixel_spacing);
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
                        derivative = ScaledComplex{1.0, 0.0, 0};
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
                    const ScaledComplex previous_derivative = derivative;
                    const int previous_reference_index = reference_index;
                    const int previous_iteration = iteration;
                    const ScaledComplex input_delta = delta;
                    delta = linear_step != nullptr
                        ? sc_add(
                            sc_mul(linear_step->A, input_delta),
                            sc_mul(linear_step->B, dc))
                        : apply_bla_series(*step, input_delta, dc, effective_order);
                    derivative = linear_step != nullptr
                        ? sc_add(
                            sc_mul(linear_step->A, previous_derivative),
                            linear_step->B)
                        : apply_bla_series_derivative(
                            *step,
                            input_delta,
                            dc,
                            previous_derivative,
                            effective_order);
                    reference_index += map_length;
                    iteration += map_length;
                    const ScaledComplex endpoint = sc_add(
                        context.orbit->scaled[static_cast<size_t>(reference_index)], delta);
                    const ScaledNorm endpoint_norm = sc_norm_squared_with_delta(
                        context.orbit->scaled[static_cast<size_t>(reference_index)],
                        delta,
                        escape_radius_mode);
                    // A block that approaches the escape boundary is replayed
                    // one iteration at a time so smooth colouring does not
                    // acquire broad BLA-sized bands.
                    if (!sc_finite(delta) || !sc_finite(endpoint)) {
                        // A bad approximation must be retried from the same
                        // state with BLA disabled for this pixel.  Previously
                        // this path could write max_iter and create a large
                        // false black region.
                        delta = previous_delta;
                        derivative = previous_derivative;
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
                        derivative = previous_derivative;
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
                            const ScaledComplex prior_total = sc_add(reference, delta);
                            derivative = sc_add(
                                sc_double(sc_mul(prior_total, derivative)),
                                ScaledComplex{1.0, 0.0, 0});
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
                                delta,
                                escape_radius_mode);
                            if (!sc_finite(delta) || !sc_finite(replay_total)) {
                                retry_without_bla = true;
                                break;
                            }
                            if (sc_outside_escape(replay_norm, escape_radius_mode)
                                || sc_outside_escape_with_delta(
                                    context.orbit->scaled[static_cast<size_t>(reference_index)],
                                    delta,
                                    escape_radius_mode)) {
                                output[index] = smooth_escape_scaled(iteration, replay_norm, options.output_bias);
                                store_render_planes_escape(
                                    planes,
                                    static_cast<size_t>(index),
                                    kalles_mandelbrot_raw_iteration(iteration),
                                    replay_total,
                                    replay_norm,
                                    total_norm,
                                    &derivative,
                                    &pixel_spacing);
                                escaped = true;
                                break;
                            }
                        }
                        if (deadline_abort) break;
                        if (escaped) break;
                        if (retry_without_bla) {
                            delta = previous_delta;
                            derivative = previous_derivative;
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
                        previous_total_norm = total_norm;
                        have_total = true;
                    } else {
                        total = endpoint;
                        total_norm = endpoint_norm;
                        previous_total_norm = total_norm;
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
                const ScaledNorm prior_norm = total_norm;
                const ScaledComplex prior_total = sc_add(reference, delta);
                derivative = sc_add(
                    sc_double(sc_mul(prior_total, derivative)),
                    ScaledComplex{1.0, 0.0, 0});
                delta = sc_add(
                    sc_double(sc_mul(reference, delta)),
                    sc_add(sc_mul(delta, delta), dc));
                ++reference_index;
                ++iteration;
                previous_total_norm = prior_norm;
                if constexpr (CollectStats) {
                    ++stats->logical_iterations;
                    ++stats->exact_steps;
                }
            }
            if (!escaped && !deadline_abort && !unresolved_pixel) {
                output[index] = encode_render_iteration(max_iter, options.output_bias);
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
    const FloatExp* point_radius = nullptr,
    FractalRenderPlanes* planes = nullptr
) {
    const bool enable_cycle_detection = options.disable_cycle == 0
        && (options.strict == 0 || options.strict_cycle != 0);
    if (enable_cycle_detection) {
        render_bla_impl<CollectStats, true>(
            output, width, height, zoom_text, context, max_iter, threads,
            series_order, series_block, options, stats_out,
            point_offsets, point_radius, planes);
    } else {
        render_bla_impl<CollectStats, false>(
            output, width, height, zoom_text, context, max_iter, threads,
            series_order, series_block, options, stats_out,
            point_offsets, point_radius, planes);
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
    bool retain_builder_orbit,
    int formula = FRACTAL_FORMULA_MANDELBROT,
    const char* julia_real_text = "0",
    const char* julia_imag_text = "0",
    int escape_radius_mode = ESCAPE_RADIUS_MODE_CLASSIC,
    int coordinate_mode = COORDINATE_MODE_PROJECT
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
    if (!valid_formula(formula)
        || !valid_escape_radius_mode(escape_radius_mode)
        || !valid_coordinate_mode(coordinate_mode)
        || !valid_c_string(julia_real_text)
        || !valid_c_string(julia_imag_text)) {
        throw std::runtime_error("invalid native reference formula or Julia constant");
    }
    context->x_center = parse_coordinate(x_center, "real");
    context->y_center = parse_coordinate(y_center, "imaginary");
    context->formula = formula;
    context->escape_radius_mode = escape_radius_mode;
    context->coordinate_mode = coordinate_mode;
    context->julia_real = parse_coordinate(julia_real_text, "Julia real");
    context->julia_imag = parse_coordinate(julia_imag_text, "Julia imaginary");
    context->requested_series_order = std::clamp(series_order, 8, 32);
    const auto reference_started = std::chrono::steady_clock::now();
    make_reference_orbit(
        *context,
        x_center,
        y_center,
        viewport_zoom,
        max_iter,
        precision_bits,
        formula,
        julia_real_text,
        julia_imag_text,
        escape_radius_mode,
        coordinate_mode);
    context->reference_build_ns = static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - reference_started).count());
    const auto bla_started = std::chrono::steady_clock::now();
    if (formula == FRACTAL_FORMULA_MANDELBROT) {
        build_bla(*context, retain_builder_orbit);
    } else {
        // The ordinary complex BLA table models only the holomorphic
        // Mandelbrot parameter plane. Alternate formulas use their own
        // formula-aware real-linear hierarchy: Tricorn keeps conjugation in
        // its 2x2 derivative and Burning Ship disables blocks at cusps.
        if (formula == FRACTAL_FORMULA_JULIA) {
            const auto series_started = std::chrono::steady_clock::now();
            build_image_series(*context);
            context->series_build_ns = static_cast<std::uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now() - series_started).count());
        } else {
            context->series_build_ns = 0;
        }
        build_alternate_linear_bla(*context, retain_builder_orbit);
    }
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
    context->x_center_text = source.x_center_text;
    context->y_center_text = source.y_center_text;
    context->formula = source.formula;
    context->julia_real = source.julia_real;
    context->julia_imag = source.julia_imag;
    context->parameter = source.parameter;
    context->escape_radius_mode = source.escape_radius_mode;
    context->coordinate_mode = source.coordinate_mode;
    context->precision_bits = source.precision_bits;
    context->reference_build_ns = static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - reference_started).count());
    const auto bla_started = std::chrono::steady_clock::now();
    const FloatExp zoom = parse_zoom_float_exp(viewport_zoom, context->precision_bits);
    context->bla.input_radius = fe_div(
        FloatExp::from_parts(viewport_height_factor(context->coordinate_mode), 0),
        zoom);

    if (source.formula != FRACTAL_FORMULA_MANDELBROT) {
        // Alternate references use a real-linear hierarchy rather than the
        // holomorphic Mandelbrot BLA. The MPFR orbit is still reusable: only
        // the input-radius-dependent matrices need rebuilding for a clone.
        context->fast_orbit = source.fast_orbit;
        if (source.formula == FRACTAL_FORMULA_JULIA) {
            const auto series_started = std::chrono::steady_clock::now();
            build_image_series(*context);
            context->series_build_ns = static_cast<std::uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now() - series_started).count());
        }
        build_alternate_linear_bla(*context);
        if (source.formula != FRACTAL_FORMULA_JULIA) {
            context->series_build_ns = 0;
        }
        context->bla_build_ns = static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - bla_started).count());
        return context;
    }

    // Only the BLA/series input domain changes between depth tiers.  The
    // expensive MPFR recurrence and compact render orbit remain shared.
    build_retargeted_bla(*context, source);
    context->bla_build_ns = static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - bla_started).count());
    return context;
}
#endif

int colourise_field_impl(
    const float* field,
    std::uint8_t* output,
    int width,
    int height,
    int max_iter,
    double phase,
    double vocal,
    double instrumental,
    double pitch,
    int threads,
    const int* interior_color,
    const std::uint8_t* accents
) {
    try {
        if (!field || !output || !valid_pixel_dimensions(width, height)
            || !valid_iteration_count(max_iter) || !valid_thread_count(threads)
            || !valid_colour_controls(phase, vocal, instrumental, pitch)
            || (interior_color != nullptr
                && (interior_color[0] < 0 || interior_color[0] > 255
                    || interior_color[1] < 0 || interior_color[1] > 255
                    || interior_color[2] < 0 || interior_color[2] > 255))) {
            throw std::runtime_error("invalid native colour dimensions or palette");
        }
        const AuroraPalette& palette = aurora_palette_for(
            max_iter, phase, vocal, instrumental, pitch, accents);
        const int palette_size = static_cast<int>(palette.rgb.size());
        // Match the float32 indexing used by the Python fallback. A double
        // product can occasionally select the adjacent 65k-entry palette bin.
        const float index_scale = static_cast<float>(palette_size - 1)
            / static_cast<float>(max_iter);
        const int pixel_count = width * height;

#ifdef _OPENMP
        if (threads > 0) omp_set_num_threads(threads);
#pragma omp parallel for schedule(static)
#endif
        for (int pixel = 0; pixel < pixel_count; ++pixel) {
            write_colour_pixel(
                field[pixel],
                max_iter,
                palette,
                index_scale,
                output + static_cast<size_t>(pixel) * 3U,
                interior_color);
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

#ifdef FRACTAL_HAVE_OPENCL
int colourise_field_opencl_impl(
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
            throw std::runtime_error("invalid OpenCL colour dimensions or palette");
        }
        initialise_opencl();
        if (!opencl_colour_available()) {
            throw std::runtime_error(
                opencl_runtime && !opencl_runtime->error.empty()
                    ? opencl_runtime->error
                    : "OpenCL Aurora colourizer is unavailable");
        }
        const AuroraPalette& palette = aurora_palette_for(
            max_iter, phase, vocal, instrumental, pitch);
        const int palette_size = static_cast<int>(palette.rgb.size());
        if (palette_size <= 0) {
            throw std::runtime_error("OpenCL Aurora palette is empty");
        }
        static_assert(sizeof(std::array<std::uint8_t, 3>) == 3,
                      "Aurora RGB entries must be tightly packed");
        const size_t pixel_count = static_cast<size_t>(width)
            * static_cast<size_t>(height);
        const size_t field_bytes = pixel_count * sizeof(float);
        const size_t output_bytes = pixel_count * 3U * sizeof(std::uint8_t);
        const size_t palette_bytes = static_cast<size_t>(palette_size) * 3U;
        const float index_scale_float = static_cast<float>(palette_size - 1)
            / static_cast<float>(max_iter);
        const double index_scale = static_cast<double>(index_scale_float);
        OpenClRuntime& runtime = *opencl_runtime;
        std::lock_guard<std::mutex> lock(runtime.mutex);
        cl_int status = CL_SUCCESS;
        const auto ensure_buffer = [&] (
            cl_mem& buffer,
            size_t& capacity,
            cl_mem_flags flags,
            size_t bytes
        ) {
            if (buffer && capacity >= bytes) return true;
            cl_mem replacement = clCreateBuffer(
                runtime.context, flags, bytes, nullptr, &status);
            if (status != CL_SUCCESS || !replacement) return false;
            if (buffer) clReleaseMemObject(buffer);
            buffer = replacement;
            capacity = bytes;
            return true;
        };
        if (!ensure_buffer(runtime.colour_field, runtime.colour_field_capacity,
                           CL_MEM_READ_ONLY, field_bytes)
            || !ensure_buffer(runtime.colour_palette, runtime.colour_palette_capacity,
                              CL_MEM_READ_ONLY, palette_bytes)
            || !ensure_buffer(runtime.colour_output, runtime.colour_output_capacity,
                              CL_MEM_WRITE_ONLY, output_bytes)) {
            throw std::runtime_error(opencl_error_text(status));
        }
        status = clEnqueueWriteBuffer(runtime.queue, runtime.colour_field, CL_TRUE,
            0, field_bytes, field, 0, nullptr, nullptr);
        if (status == CL_SUCCESS) {
            status = clEnqueueWriteBuffer(
                runtime.queue, runtime.colour_palette, CL_TRUE, 0,
                palette_bytes,
                reinterpret_cast<const std::uint8_t*>(palette.rgb.data()),
                0, nullptr, nullptr);
        }
        const int pixel_count_int = static_cast<int>(pixel_count);
        if (status == CL_SUCCESS) status = clSetKernelArg(
            runtime.colour_kernel, 0, sizeof(runtime.colour_field), &runtime.colour_field);
        if (status == CL_SUCCESS) status = clSetKernelArg(
            runtime.colour_kernel, 1, sizeof(runtime.colour_palette), &runtime.colour_palette);
        if (status == CL_SUCCESS) status = clSetKernelArg(
            runtime.colour_kernel, 2, sizeof(runtime.colour_output), &runtime.colour_output);
        if (status == CL_SUCCESS) status = clSetKernelArg(
            runtime.colour_kernel, 3, sizeof(pixel_count_int), &pixel_count_int);
        if (status == CL_SUCCESS) status = clSetKernelArg(
            runtime.colour_kernel, 4, sizeof(max_iter), &max_iter);
        if (status == CL_SUCCESS) status = clSetKernelArg(
            runtime.colour_kernel, 5, sizeof(palette_size), &palette_size);
        if (status == CL_SUCCESS) status = clSetKernelArg(
            runtime.colour_kernel, 6, sizeof(index_scale), &index_scale);
        const size_t workgroup = runtime.colour_workgroup_size;
        const size_t global_size = workgroup > 0
            ? ((pixel_count + workgroup - 1U) / workgroup) * workgroup
            : pixel_count;
        const size_t* local_work_size = workgroup > 0 ? &workgroup : nullptr;
        if (status == CL_SUCCESS) status = clEnqueueNDRangeKernel(
            runtime.queue, runtime.colour_kernel, 1, nullptr, &global_size,
            local_work_size, 0, nullptr, nullptr);
        if (status == CL_SUCCESS) status = clEnqueueReadBuffer(
            runtime.queue, runtime.colour_output, CL_TRUE, 0, output_bytes,
            output, 0, nullptr, nullptr);
        if (status != CL_SUCCESS) throw std::runtime_error(opencl_error_text(status));
        set_error("");
        return 0;
    } catch (const std::exception& error) {
        set_error(error.what());
        return 1;
    } catch (...) {
        set_error("OpenCL colourizer failed with an unknown exception");
        return 1;
    }
}

int atlas_colourise_opencl_impl(
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
    double child_zoom,
    double parent_field_bias,
    double child_field_bias,
    double output_field_bias,
    int palette_max_iter,
    int feather,
    double phase,
    double vocal,
    double instrumental,
    double pitch,
    int threads,
    std::uint64_t parent_cache_token,
    std::uint64_t child_cache_token,
    const std::uint8_t* accents,
    const int* interior_color
) {
    try {
        if (!parent || !output
            || !valid_pixel_dimensions(parent_width, parent_height)
            || !valid_pixel_dimensions(output_width, output_height)
            || !valid_iteration_count(parent_max_iter)
            || !valid_thread_count(threads)
            || !valid_colour_controls(phase, vocal, instrumental, pitch)
            || !std::isfinite(parent_zoom) || parent_zoom <= 0.0
            || !std::isfinite(child_fraction)
            || child_fraction < 0.0 || child_fraction > 1.0
            || !std::isfinite(child_zoom) || child_zoom <= 0.0
            || !std::isfinite(parent_field_bias) || parent_field_bias < 0.0
            || parent_field_bias > static_cast<double>(parent_max_iter)
            || !std::isfinite(child_field_bias) || child_field_bias < 0.0
            || !std::isfinite(output_field_bias) || output_field_bias < 0.0
            || feather < 0
            || !valid_iteration_count(palette_max_iter)
            || (interior_color != nullptr
                && (interior_color[0] < 0 || interior_color[0] > 255
                    || interior_color[1] < 0 || interior_color[1] > 255
                    || interior_color[2] < 0 || interior_color[2] > 255))) {
            throw std::runtime_error("invalid OpenCL atlas colour dimensions");
        }
        const bool use_child = child != nullptr && child_fraction > 0.0;
        if (use_child && (!valid_pixel_dimensions(child_width, child_height)
                          || !valid_iteration_count(child_max_iter)
                          || child_field_bias > static_cast<double>(child_max_iter))) {
            throw std::runtime_error("invalid OpenCL atlas child tile");
        }
        const int effective_iter = std::max(
            palette_max_iter,
            std::max(parent_max_iter, use_child ? child_max_iter : 0));
        if (!valid_iteration_count(effective_iter)
            || output_field_bias > static_cast<double>(effective_iter)) {
            throw std::runtime_error("invalid OpenCL atlas iteration cap");
        }
        initialise_opencl();
        if (!opencl_atlas_colour_available()) {
            throw std::runtime_error(
                opencl_runtime && !opencl_runtime->error.empty()
                    ? opencl_runtime->error
                    : "OpenCL fused atlas colourizer is unavailable");
        }
        const AuroraPalette& palette = aurora_palette_for(
            effective_iter, phase, vocal, instrumental, pitch, accents);
        const int palette_size = static_cast<int>(palette.rgb.size());
        if (palette_size <= 0) {
            throw std::runtime_error("OpenCL Aurora palette is empty");
        }
        static_assert(sizeof(std::array<std::uint8_t, 3>) == 3,
                      "Aurora RGB entries must be tightly packed");
        const size_t parent_bytes = static_cast<size_t>(parent_width)
            * static_cast<size_t>(parent_height) * sizeof(float);
        const size_t child_bytes = use_child
            ? static_cast<size_t>(child_width)
                * static_cast<size_t>(child_height) * sizeof(float)
            : sizeof(float);
        const size_t output_bytes = static_cast<size_t>(output_width)
            * static_cast<size_t>(output_height) * 3U
            * sizeof(std::uint8_t);
        const size_t output_pixel_count = static_cast<size_t>(output_width)
            * static_cast<size_t>(output_height);
        // Pageable host uploads can be queued asynchronously for live/FHD
        // surfaces, but on the discrete 4K path the driver's staging copy is
        // faster when the small metadata uploads are completed synchronously.
        const cl_bool metadata_upload_blocking = output_pixel_count
            > static_cast<size_t>(1920) * 1080 ? CL_TRUE : CL_FALSE;
        const size_t palette_bytes = static_cast<size_t>(palette_size) * 3U;
        const float index_scale_float = static_cast<float>(palette_size - 1)
            / static_cast<float>(effective_iter);
        const float palette_index_scale = index_scale_float;
        const int interior_red = interior_color != nullptr ? interior_color[0] : 0;
        const int interior_green = interior_color != nullptr ? interior_color[1] : 0;
        const int interior_blue = interior_color != nullptr ? interior_color[2] : 0;
        OpenClRuntime& runtime = *opencl_runtime;
        // Hardware image coordinates are float-valued.  Keep the sampler
        // path for shallow/normal crops, and retain the double-coordinate
        // buffer kernel for the very deep zooms where float coordinates would
        // collapse several atlas pixels onto the same sample.
        bool image_path = runtime.atlas_images
            && runtime.atlas_image_colour_kernel != nullptr
            && std::abs(parent_field_bias) <= 1.0e-12
            && std::abs(child_field_bias) <= 1.0e-12
            && std::abs(output_field_bias) <= 1.0e-12
            && parent_zoom <= 1.0e6
            && (!use_child || child_zoom <= 1.0e6);
        std::vector<float> parent_image_data;
        std::vector<float> child_image_data;
        std::vector<OpenClAtlasAxis> parent_x_axis;
        std::vector<OpenClAtlasAxis> parent_y_axis;
        std::vector<OpenClAtlasAxis> child_x_axis;
        std::vector<OpenClAtlasAxis> child_y_axis;
        if (runtime.atlas_axis_maps) {
            fill_opencl_atlas_axis(
                parent_x_axis, output_width, parent_width, parent_zoom);
            fill_opencl_atlas_axis(
                parent_y_axis, output_height, parent_height, parent_zoom);
            if (use_child) {
                const int child_destination_width = child_fraction >= 0.999999
                    ? output_width
                    : std::max(
                        1,
                        static_cast<int>(std::floor(
                            static_cast<double>(output_width) * child_fraction
                            + 0.5)));
                const int child_destination_height = child_fraction >= 0.999999
                    ? output_height
                    : std::max(
                        1,
                        static_cast<int>(std::floor(
                            static_cast<double>(output_height) * child_fraction
                            + 0.5)));
                fill_opencl_atlas_axis(
                    child_x_axis,
                    child_destination_width,
                    child_width,
                    child_zoom);
                fill_opencl_atlas_axis(
                    child_y_axis,
                    child_destination_height,
                    child_height,
                    child_zoom);
            }
        }
        std::lock_guard<std::mutex> lock(runtime.mutex);
        cl_command_queue atlas_queue = runtime.queue;
        const size_t tiled_workgroup = opencl_atlas_tiled_workgroup(
            runtime.atlas_tiled_colour_workgroup_limit,
            output_pixel_count);
        bool use_tiled_kernel = !image_path
            && (!runtime.atlas_axis_maps
                || output_pixel_count <= static_cast<size_t>(1920) * 1080)
            && runtime.atlas_tiled_colour_kernel != nullptr
            && tiled_workgroup > 0;
        cl_kernel atlas_kernel = use_tiled_kernel
            ? runtime.atlas_tiled_colour_kernel
            : (image_path
                ? runtime.atlas_image_colour_kernel
                : runtime.atlas_colour_kernel);
        cl_mem& atlas_parent = runtime.atlas_parent;
        size_t& atlas_parent_capacity = runtime.atlas_parent_capacity;
        cl_mem& atlas_child = runtime.atlas_child;
        size_t& atlas_child_capacity = runtime.atlas_child_capacity;
        cl_mem& atlas_palette = runtime.colour_palette;
        size_t& atlas_palette_capacity = runtime.colour_palette_capacity;
        cl_mem& atlas_output = runtime.atlas_output;
        size_t& atlas_output_capacity = runtime.atlas_output_capacity;
        std::uint64_t& atlas_parent_cache_token = runtime.atlas_parent_cache_token;
        size_t& atlas_parent_cache_bytes = runtime.atlas_parent_cache_bytes;
        std::uint64_t& atlas_child_cache_token = runtime.atlas_child_cache_token;
        size_t& atlas_child_cache_bytes = runtime.atlas_child_cache_bytes;
        cl_int status = CL_SUCCESS;
        const auto ensure_buffer = [&] (
            cl_mem& buffer,
            size_t& capacity,
            cl_mem_flags flags,
            size_t bytes
        ) {
            if (buffer && capacity >= bytes) return true;
            cl_mem replacement = clCreateBuffer(
                runtime.context, flags, bytes, nullptr, &status);
            if (status != CL_SUCCESS || !replacement) return false;
            if (buffer) clReleaseMemObject(buffer);
            buffer = replacement;
            capacity = bytes;
            if (&buffer == &atlas_parent) {
                atlas_parent_cache_token = 0;
                atlas_parent_cache_bytes = 0;
            } else if (&buffer == &atlas_child) {
                atlas_child_cache_token = 0;
                atlas_child_cache_bytes = 0;
            }
            return true;
        };
        if (image_path
            && (!ensure_opencl_atlas_image(
                    runtime,
                    runtime.atlas_parent_image,
                    runtime.atlas_parent_image_width,
                    runtime.atlas_parent_image_height,
                    parent_width,
                    parent_height,
                    runtime.atlas_parent_image_cache_token,
                    status)
                || (use_child && !ensure_opencl_atlas_image(
                    runtime,
                    runtime.atlas_child_image,
                    runtime.atlas_child_image_width,
                    runtime.atlas_child_image_height,
                    child_width,
                    child_height,
                    runtime.atlas_child_image_cache_token,
                    status)))) {
            // Some OpenCL implementations expose the atlas kernels but not
            // float RGBA image objects.  Fall back to the proven buffer path
            // in the same call instead of failing the render.
            image_path = false;
            use_tiled_kernel = (!runtime.atlas_axis_maps
                || output_pixel_count <= static_cast<size_t>(1920) * 1080)
                && runtime.atlas_tiled_colour_kernel != nullptr
                && tiled_workgroup > 0;
            atlas_kernel = use_tiled_kernel
                ? runtime.atlas_tiled_colour_kernel
                : runtime.atlas_colour_kernel;
            status = CL_SUCCESS;
        }
        if ((!image_path
                && (!ensure_buffer(atlas_parent, atlas_parent_capacity,
                                   CL_MEM_READ_ONLY, parent_bytes)
                    || !ensure_buffer(atlas_child, atlas_child_capacity,
                                      CL_MEM_READ_ONLY, child_bytes)))
            || !ensure_buffer(atlas_palette,
                              atlas_palette_capacity,
                              CL_MEM_READ_ONLY, palette_bytes)
            || !ensure_buffer(
                atlas_output,
                atlas_output_capacity,
                CL_MEM_WRITE_ONLY,
                output_bytes)) {
            throw std::runtime_error(opencl_error_text(status));
        }
        if (runtime.atlas_axis_maps) {
            if (!ensure_buffer(
                    runtime.atlas_parent_x_axis,
                    runtime.atlas_parent_x_axis_capacity,
                    CL_MEM_READ_ONLY,
                    parent_x_axis.size() * sizeof(OpenClAtlasAxis))
                || !ensure_buffer(
                    runtime.atlas_parent_y_axis,
                    runtime.atlas_parent_y_axis_capacity,
                    CL_MEM_READ_ONLY,
                    parent_y_axis.size() * sizeof(OpenClAtlasAxis))
                || (use_child && !ensure_buffer(
                    runtime.atlas_child_x_axis,
                    runtime.atlas_child_x_axis_capacity,
                    CL_MEM_READ_ONLY,
                    child_x_axis.size() * sizeof(OpenClAtlasAxis)))
                || (use_child && !ensure_buffer(
                    runtime.atlas_child_y_axis,
                    runtime.atlas_child_y_axis_capacity,
                    CL_MEM_READ_ONLY,
                    child_y_axis.size() * sizeof(OpenClAtlasAxis)))) {
                throw std::runtime_error(opencl_error_text(status));
            }
        }
        if (image_path) {
            const bool parent_cached = parent_cache_token != 0
                && runtime.atlas_parent_image_cache_token == parent_cache_token;
            if (!parent_cached) {
                parent_image_data = pack_opencl_atlas_image(
                    parent, parent_width, parent_height, parent_max_iter);
                const size_t origin[3] = {0, 0, 0};
                const size_t region[3] = {
                    static_cast<size_t>(parent_width),
                    static_cast<size_t>(parent_height),
                    1,
                };
                status = clEnqueueWriteImage(
                    atlas_queue,
                    runtime.atlas_parent_image,
                    CL_TRUE,
                    origin,
                    region,
                    static_cast<size_t>(parent_width) * 4U * sizeof(float),
                    0,
                    parent_image_data.data(),
                    0,
                    nullptr,
                    nullptr);
                runtime.atlas_parent_image_cache_token =
                    parent_cache_token;
            }
            const bool child_cached = !use_child || (
                child_cache_token != 0
                && runtime.atlas_child_image_cache_token == child_cache_token);
            if (status == CL_SUCCESS && use_child && !child_cached) {
                child_image_data = pack_opencl_atlas_image(
                    child, child_width, child_height, child_max_iter);
                const size_t origin[3] = {0, 0, 0};
                const size_t region[3] = {
                    static_cast<size_t>(child_width),
                    static_cast<size_t>(child_height),
                    1,
                };
                status = clEnqueueWriteImage(
                    atlas_queue,
                    runtime.atlas_child_image,
                    CL_TRUE,
                    origin,
                    region,
                    static_cast<size_t>(child_width) * 4U * sizeof(float),
                    0,
                    child_image_data.data(),
                    0,
                    nullptr,
                    nullptr);
                runtime.atlas_child_image_cache_token = child_cache_token;
            }
        } else {
            const bool parent_cached = parent_cache_token != 0
                && atlas_parent_cache_token == parent_cache_token
                && atlas_parent_cache_bytes == parent_bytes;
            if (!parent_cached) {
                status = clEnqueueWriteBuffer(
                    atlas_queue, atlas_parent, CL_TRUE, 0,
                    parent_bytes, parent, 0, nullptr, nullptr);
                atlas_parent_cache_token = parent_cache_token;
                atlas_parent_cache_bytes = parent_cache_token != 0
                    ? parent_bytes : 0;
            }
            const bool child_cached = !use_child || (
                child_cache_token != 0
                && atlas_child_cache_token == child_cache_token
                && atlas_child_cache_bytes == child_bytes);
            if (status == CL_SUCCESS && use_child && !child_cached) {
                status = clEnqueueWriteBuffer(
                    atlas_queue, atlas_child, CL_TRUE, 0,
                    child_bytes, child, 0, nullptr, nullptr);
                atlas_child_cache_token = child_cache_token;
                atlas_child_cache_bytes = child_cache_token != 0
                    ? child_bytes : 0;
            }
        }
        if (status == CL_SUCCESS) status = clEnqueueWriteBuffer(
            atlas_queue, atlas_palette, metadata_upload_blocking, 0,
            palette_bytes,
            reinterpret_cast<const std::uint8_t*>(palette.rgb.data()),
            0, nullptr, nullptr);
        if (status == CL_SUCCESS && runtime.atlas_axis_maps) {
            status = clEnqueueWriteBuffer(
                atlas_queue,
                runtime.atlas_parent_x_axis,
                metadata_upload_blocking,
                0,
                parent_x_axis.size() * sizeof(OpenClAtlasAxis),
                parent_x_axis.data(),
                0,
                nullptr,
                nullptr);
        }
        if (status == CL_SUCCESS && runtime.atlas_axis_maps) {
            status = clEnqueueWriteBuffer(
                atlas_queue,
                runtime.atlas_parent_y_axis,
                metadata_upload_blocking,
                0,
                parent_y_axis.size() * sizeof(OpenClAtlasAxis),
                parent_y_axis.data(),
                0,
                nullptr,
                nullptr);
        }
        if (status == CL_SUCCESS && runtime.atlas_axis_maps && use_child) {
            status = clEnqueueWriteBuffer(
                atlas_queue,
                runtime.atlas_child_x_axis,
                metadata_upload_blocking,
                0,
                child_x_axis.size() * sizeof(OpenClAtlasAxis),
                child_x_axis.data(),
                0,
                nullptr,
                nullptr);
        }
        if (status == CL_SUCCESS && runtime.atlas_axis_maps && use_child) {
            status = clEnqueueWriteBuffer(
                atlas_queue,
                runtime.atlas_child_y_axis,
                metadata_upload_blocking,
                0,
                child_y_axis.size() * sizeof(OpenClAtlasAxis),
                child_y_axis.data(),
                0,
                nullptr,
                nullptr);
        }
        cl_mem parent_buffer = image_path
            ? runtime.atlas_parent_image : atlas_parent;
        cl_mem child_buffer = image_path
            ? (use_child ? runtime.atlas_child_image : runtime.atlas_parent_image)
            : (use_child ? atlas_child : atlas_parent);
        cl_mem palette_buffer = atlas_palette;
        cl_mem output_buffer = atlas_output;
        cl_mem parent_x_map = runtime.atlas_parent_x_axis;
        cl_mem parent_y_map = runtime.atlas_parent_y_axis;
        cl_mem child_x_map = use_child
            ? runtime.atlas_child_x_axis : runtime.atlas_parent_x_axis;
        cl_mem child_y_map = use_child
            ? runtime.atlas_child_y_axis : runtime.atlas_parent_y_axis;
        const int use_child_int = use_child ? 1 : 0;
        if (status == CL_SUCCESS) status = clSetKernelArg(
            atlas_kernel, 0, sizeof(parent_buffer), &parent_buffer);
        if (status == CL_SUCCESS) status = clSetKernelArg(
            atlas_kernel, 1, sizeof(parent_width), &parent_width);
        if (status == CL_SUCCESS) status = clSetKernelArg(
            atlas_kernel, 2, sizeof(parent_height), &parent_height);
        if (status == CL_SUCCESS) status = clSetKernelArg(
            atlas_kernel, 3, sizeof(parent_max_iter), &parent_max_iter);
        if (status == CL_SUCCESS) status = clSetKernelArg(
            atlas_kernel, 4, sizeof(child_buffer), &child_buffer);
        if (status == CL_SUCCESS) status = clSetKernelArg(
            atlas_kernel, 5, sizeof(child_width), &child_width);
        if (status == CL_SUCCESS) status = clSetKernelArg(
            atlas_kernel, 6, sizeof(child_height), &child_height);
        if (status == CL_SUCCESS) status = clSetKernelArg(
            atlas_kernel, 7, sizeof(child_max_iter), &child_max_iter);
        if (status == CL_SUCCESS) status = clSetKernelArg(
            atlas_kernel, 8, sizeof(palette_buffer), &palette_buffer);
        if (status == CL_SUCCESS) status = clSetKernelArg(
            atlas_kernel, 9, sizeof(output_buffer), &output_buffer);
        if (status == CL_SUCCESS) status = clSetKernelArg(
            atlas_kernel, 10, sizeof(output_width), &output_width);
        if (status == CL_SUCCESS) status = clSetKernelArg(
            atlas_kernel, 11, sizeof(output_height), &output_height);
        if (status == CL_SUCCESS) status = clSetKernelArg(
            atlas_kernel, 12, sizeof(parent_zoom), &parent_zoom);
        if (status == CL_SUCCESS) status = clSetKernelArg(
            atlas_kernel, 13, sizeof(child_fraction), &child_fraction);
        if (status == CL_SUCCESS) status = clSetKernelArg(
            atlas_kernel, 14, sizeof(child_zoom), &child_zoom);
        if (status == CL_SUCCESS) status = clSetKernelArg(
            atlas_kernel, 15, sizeof(parent_field_bias),
            &parent_field_bias);
        if (status == CL_SUCCESS) status = clSetKernelArg(
            atlas_kernel, 16, sizeof(child_field_bias),
            &child_field_bias);
        if (status == CL_SUCCESS) status = clSetKernelArg(
            atlas_kernel, 17, sizeof(output_field_bias),
            &output_field_bias);
        if (status == CL_SUCCESS) status = clSetKernelArg(
            atlas_kernel, 18, sizeof(effective_iter), &effective_iter);
        if (status == CL_SUCCESS) status = clSetKernelArg(
            atlas_kernel, 19, sizeof(feather), &feather);
        if (status == CL_SUCCESS) status = clSetKernelArg(
            atlas_kernel, 20, sizeof(palette_size), &palette_size);
        if (status == CL_SUCCESS) status = clSetKernelArg(
            atlas_kernel, 21, sizeof(palette_index_scale),
            &palette_index_scale);
        if (status == CL_SUCCESS) status = clSetKernelArg(
            atlas_kernel, 22, sizeof(interior_red), &interior_red);
        if (status == CL_SUCCESS) status = clSetKernelArg(
            atlas_kernel, 23, sizeof(interior_green), &interior_green);
        if (status == CL_SUCCESS) status = clSetKernelArg(
            atlas_kernel, 24, sizeof(interior_blue), &interior_blue);
        if (status == CL_SUCCESS) status = clSetKernelArg(
            atlas_kernel, 25, sizeof(use_child_int), &use_child_int);
        const cl_ulong output_offset = 0;
        const cl_ulong palette_offset = 0;
        if (status == CL_SUCCESS) status = clSetKernelArg(
            atlas_kernel, 26, sizeof(output_offset), &output_offset);
        if (status == CL_SUCCESS) status = clSetKernelArg(
            atlas_kernel, 27, sizeof(palette_offset), &palette_offset);
        if (status == CL_SUCCESS && runtime.atlas_axis_maps) {
            status = clSetKernelArg(
                atlas_kernel, 28, sizeof(parent_x_map), &parent_x_map);
        }
        if (status == CL_SUCCESS && runtime.atlas_axis_maps) {
            status = clSetKernelArg(
                atlas_kernel, 29, sizeof(parent_y_map), &parent_y_map);
        }
        if (status == CL_SUCCESS && runtime.atlas_axis_maps) {
            status = clSetKernelArg(
                atlas_kernel, 30, sizeof(child_x_map), &child_x_map);
        }
        if (status == CL_SUCCESS && runtime.atlas_axis_maps) {
            status = clSetKernelArg(
                atlas_kernel, 31, sizeof(child_y_map), &child_y_map);
        }
        const int parent_x_map_offset = 0;
        const int parent_y_map_offset = 0;
        const int child_x_map_offset = 0;
        const int child_y_map_offset = 0;
        if (status == CL_SUCCESS && runtime.atlas_axis_maps) {
            status = clSetKernelArg(
                atlas_kernel, 32, sizeof(parent_x_map_offset),
                &parent_x_map_offset);
        }
        if (status == CL_SUCCESS && runtime.atlas_axis_maps) {
            status = clSetKernelArg(
                atlas_kernel, 33, sizeof(parent_y_map_offset),
                &parent_y_map_offset);
        }
        if (status == CL_SUCCESS && runtime.atlas_axis_maps) {
            status = clSetKernelArg(
                atlas_kernel, 34, sizeof(child_x_map_offset),
                &child_x_map_offset);
        }
        if (status == CL_SUCCESS && runtime.atlas_axis_maps) {
            status = clSetKernelArg(
                atlas_kernel, 35, sizeof(child_y_map_offset),
                &child_y_map_offset);
        }
        const size_t pixel_count = output_pixel_count;
        size_t workgroup = runtime.atlas_colour_workgroup_size;
        bool use_2d_launch = false;
        size_t global_size_2d[2] = {
            static_cast<size_t>(output_width),
            static_cast<size_t>(output_height),
        };
        size_t global_size_1d = pixel_count;
        size_t local_size_2d[2] = {0, 0};
        if (use_tiled_kernel) {
            workgroup = tiled_workgroup;
            const size_t local_y = workgroup / 16U;
            global_size_2d[0] =
                ((static_cast<size_t>(output_width) + 15U) / 16U) * 16U;
            global_size_2d[1] =
                ((static_cast<size_t>(output_height) + local_y - 1U)
                    / local_y) * local_y;
            local_size_2d[0] = 16U;
            local_size_2d[1] = local_y;
            use_2d_launch = true;
        } else {
            // On the tested NVIDIA path, the atlas kernel benefits from a
            // wider local group at preview/FHD sizes, while a smaller group
            // wins at 4K. Keep the old launch selection for image/map paths.
            if (!std::getenv("FRACTAL_OPENCL_WORKGROUP")
                && runtime.device_is_gpu
                && workgroup > 0
                && runtime.atlas_colour_workgroup_limit > 0) {
                const size_t target = pixel_count
                    <= static_cast<size_t>(1920) * 1080 ? 256U : 128U;
                if (target >= workgroup
                    && target <= runtime.atlas_colour_workgroup_limit
                    && target % workgroup == 0) {
                    workgroup = target;
                } else if (pixel_count <= static_cast<size_t>(1920) * 1080
                           && workgroup <= runtime.atlas_colour_workgroup_limit / 2) {
                    workgroup *= 2;
                }
            }
            use_2d_launch = pixel_count
                <= static_cast<size_t>(1920) * 1080
                || opencl_atlas_force_2d_launch();
            global_size_2d[0] = workgroup > 0
                ? ((static_cast<size_t>(output_width) + workgroup - 1U)
                    / workgroup) * workgroup
                : static_cast<size_t>(output_width);
            global_size_2d[1] = static_cast<size_t>(output_height);
            global_size_1d = workgroup > 0
                ? ((pixel_count + workgroup - 1U) / workgroup) * workgroup
                : pixel_count;
            local_size_2d[0] = workgroup;
            local_size_2d[1] = 1U;
        }
        const size_t* local_work_size_2d =
            workgroup > 0 ? local_size_2d : nullptr;
        const size_t* local_work_size_1d =
            workgroup > 0 ? &workgroup : nullptr;
        if (status == CL_SUCCESS) status = clEnqueueNDRangeKernel(
            atlas_queue, atlas_kernel, use_2d_launch ? 2 : 1, nullptr,
            use_2d_launch ? global_size_2d : &global_size_1d,
            use_2d_launch ? local_work_size_2d : local_work_size_1d,
            0, nullptr, nullptr);
        if (status == CL_SUCCESS) status = clEnqueueReadBuffer(
            atlas_queue, atlas_output, CL_TRUE, 0,
            output_bytes, output, 0, nullptr, nullptr);
        if (status != CL_SUCCESS) throw std::runtime_error(opencl_error_text(status));
        set_error("");
        return 0;
    } catch (const std::exception& error) {
        set_error(error.what());
        return 1;
    } catch (...) {
        set_error("OpenCL fused atlas colourizer failed with an unknown exception");
        return 1;
    }
}

int atlas_colourise_opencl_batch_impl(
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
    const FractalAtlasColourFrame* frames,
    int frame_count,
    int threads,
    std::uint64_t parent_cache_token,
    std::uint64_t child_cache_token,
    const std::uint8_t* accents,
    const int* interior_color
) {
    try {
        if (!parent || !output || !frames
            || !valid_pixel_dimensions(parent_width, parent_height)
            || !valid_pixel_dimensions(output_width, output_height)
            || !valid_iteration_count(parent_max_iter)
            || !valid_thread_count(threads)
            || frame_count < 1 || frame_count > 64
            || (interior_color != nullptr
                && (interior_color[0] < 0 || interior_color[0] > 255
                    || interior_color[1] < 0 || interior_color[1] > 255
                    || interior_color[2] < 0 || interior_color[2] > 255))) {
            throw std::runtime_error("invalid batched OpenCL atlas dimensions");
        }
        const bool child_available = child != nullptr;
        if (child_available
            && (!valid_pixel_dimensions(child_width, child_height)
                || !valid_iteration_count(child_max_iter))) {
            throw std::runtime_error("invalid batched OpenCL atlas child tile");
        }

        static_assert(sizeof(std::array<std::uint8_t, 3>) == 3,
                      "Aurora RGB entries must be tightly packed");
        struct PaletteInfo {
            size_t offset = 0;
            int size = 0;
            float index_scale = 0.0f;
            int effective_iter = 0;
        };
        const size_t frame_output_bytes = static_cast<size_t>(output_width)
            * static_cast<size_t>(output_height) * 3U;
        if (frame_output_bytes == 0
            || static_cast<size_t>(frame_count)
                > std::numeric_limits<size_t>::max() / frame_output_bytes) {
            throw std::runtime_error("batched OpenCL atlas output is too large");
        }
        const size_t output_bytes = frame_output_bytes
            * static_cast<size_t>(frame_count);
        const size_t output_pixel_count = static_cast<size_t>(output_width)
            * static_cast<size_t>(output_height);
        const cl_bool metadata_upload_blocking = output_pixel_count
            > static_cast<size_t>(1920) * 1080 ? CL_TRUE : CL_FALSE;
        const size_t parent_bytes = static_cast<size_t>(parent_width)
            * static_cast<size_t>(parent_height) * sizeof(float);
        const size_t child_bytes = child_available
            ? static_cast<size_t>(child_width)
                * static_cast<size_t>(child_height) * sizeof(float)
            : sizeof(float);

        std::vector<PaletteInfo> palette_info;
        palette_info.reserve(static_cast<size_t>(frame_count));
        std::vector<std::uint8_t> palette_data;
        std::vector<int> use_child_values;
        use_child_values.reserve(static_cast<size_t>(frame_count));
        for (int index = 0; index < frame_count; ++index) {
            const FractalAtlasColourFrame& frame = frames[index];
            if (!valid_iteration_count(frame.palette_max_iter)
                || !std::isfinite(frame.parent_zoom)
                || frame.parent_zoom <= 0.0
                || !std::isfinite(frame.child_fraction)
                || frame.child_fraction < 0.0
                || frame.child_fraction > 1.0
                || !std::isfinite(frame.child_zoom)
                || frame.child_zoom <= 0.0
                || !std::isfinite(frame.parent_field_bias)
                || frame.parent_field_bias < 0.0
                || frame.parent_field_bias
                    > static_cast<double>(parent_max_iter)
                || !std::isfinite(frame.child_field_bias)
                || frame.child_field_bias < 0.0
                || !std::isfinite(frame.output_field_bias)
                || frame.output_field_bias < 0.0
                || frame.feather < 0
                || !valid_colour_controls(
                    frame.phase, frame.vocal,
                    frame.instrumental, frame.pitch)) {
                throw std::runtime_error(
                    "invalid batched OpenCL atlas frame controls");
            }
            const bool use_child = child_available
                && frame.child_fraction > 0.0;
            if (frame.child_fraction > 0.0 && !child_available) {
                throw std::runtime_error(
                    "batched OpenCL atlas frame requires a child tile");
            }
            if (use_child
                && frame.child_field_bias
                    > static_cast<double>(child_max_iter)) {
                throw std::runtime_error(
                    "invalid batched OpenCL atlas child bias");
            }
            const int effective_iter = std::max(
                static_cast<int>(frame.palette_max_iter),
                std::max(parent_max_iter, use_child ? child_max_iter : 0));
            if (!valid_iteration_count(effective_iter)
                || frame.output_field_bias
                    > static_cast<double>(effective_iter)) {
                throw std::runtime_error(
                    "invalid batched OpenCL atlas iteration cap");
            }
            const AuroraPalette& palette = aurora_palette_for(
                effective_iter,
                frame.phase,
                frame.vocal,
                frame.instrumental,
                frame.pitch,
                accents);
            const int palette_size = static_cast<int>(palette.rgb.size());
            if (palette_size <= 0) {
                throw std::runtime_error("OpenCL Aurora palette is empty");
            }
            const size_t palette_bytes = static_cast<size_t>(palette_size) * 3U;
            if (palette_data.size() > std::numeric_limits<size_t>::max()
                    - palette_bytes) {
                throw std::runtime_error("batched OpenCL atlas palette is too large");
            }
            const size_t palette_offset = palette_data.size();
            const auto* palette_bytes_ptr = reinterpret_cast<const std::uint8_t*>(
                palette.rgb.data());
            palette_data.insert(
                palette_data.end(),
                palette_bytes_ptr,
                palette_bytes_ptr + palette_bytes);
            const float index_scale_float = static_cast<float>(palette_size - 1)
                / static_cast<float>(effective_iter);
            palette_info.push_back({
                palette_offset,
                palette_size,
                index_scale_float,
                effective_iter,
            });
            use_child_values.push_back(use_child ? 1 : 0);
        }
        if (palette_data.empty()) {
            throw std::runtime_error("batched OpenCL atlas palette is empty");
        }

        initialise_opencl();
        if (!opencl_atlas_colour_available()) {
            throw std::runtime_error(
                opencl_runtime && !opencl_runtime->error.empty()
                    ? opencl_runtime->error
                    : "OpenCL fused atlas colourizer is unavailable");
        }
        OpenClRuntime& runtime = *opencl_runtime;
        bool image_path = runtime.atlas_images
            && runtime.atlas_image_colour_kernel != nullptr;
        for (int index = 0; image_path && index < frame_count; ++index) {
            const FractalAtlasColourFrame& frame = frames[index];
            const bool use_child = use_child_values[static_cast<size_t>(index)] != 0;
            image_path = std::abs(frame.parent_field_bias) <= 1.0e-12
                && std::abs(frame.child_field_bias) <= 1.0e-12
                && std::abs(frame.output_field_bias) <= 1.0e-12
                && frame.parent_zoom <= 1.0e6
                && (!use_child || frame.child_zoom <= 1.0e6);
        }
        std::vector<float> parent_image_data;
        std::vector<float> child_image_data;
        std::vector<OpenClAtlasAxis> parent_x_maps;
        std::vector<OpenClAtlasAxis> parent_y_maps;
        std::vector<OpenClAtlasAxis> child_x_maps;
        std::vector<OpenClAtlasAxis> child_y_maps;
        if (runtime.atlas_axis_maps) {
            const size_t frame_count_size = static_cast<size_t>(frame_count);
            const size_t width_size = static_cast<size_t>(output_width);
            const size_t height_size = static_cast<size_t>(output_height);
            if (frame_count_size > std::numeric_limits<size_t>::max()
                    / width_size
                || frame_count_size > std::numeric_limits<size_t>::max()
                    / height_size
                || frame_count_size * width_size
                    > static_cast<size_t>(std::numeric_limits<int>::max())
                || frame_count_size * height_size
                    > static_cast<size_t>(std::numeric_limits<int>::max())) {
                throw std::runtime_error(
                    "batched OpenCL atlas axis maps are too large");
            }
            parent_x_maps.resize(frame_count_size * width_size);
            parent_y_maps.resize(frame_count_size * height_size);
            if (child_available) {
                child_x_maps.resize(frame_count_size * width_size);
                child_y_maps.resize(frame_count_size * height_size);
            }
            std::vector<OpenClAtlasAxis> axis;
            for (int index = 0; index < frame_count; ++index) {
                const FractalAtlasColourFrame& frame = frames[index];
                fill_opencl_atlas_axis(
                    axis, output_width, parent_width, frame.parent_zoom);
                std::copy(
                    axis.begin(), axis.end(),
                    parent_x_maps.begin()
                        + static_cast<size_t>(index) * width_size);
                fill_opencl_atlas_axis(
                    axis, output_height, parent_height, frame.parent_zoom);
                std::copy(
                    axis.begin(), axis.end(),
                    parent_y_maps.begin()
                        + static_cast<size_t>(index) * height_size);
                if (child_available) {
                    const int child_destination_width =
                        frame.child_fraction >= 0.999999
                            ? output_width
                            : std::max(
                                1,
                                static_cast<int>(std::floor(
                                    static_cast<double>(output_width)
                                        * frame.child_fraction
                                    + 0.5)));
                    const int child_destination_height =
                        frame.child_fraction >= 0.999999
                            ? output_height
                            : std::max(
                                1,
                                static_cast<int>(std::floor(
                                    static_cast<double>(output_height)
                                        * frame.child_fraction
                                    + 0.5)));
                    fill_opencl_atlas_axis(
                        axis, child_destination_width,
                        child_width, frame.child_zoom);
                    std::copy(
                        axis.begin(), axis.end(),
                        child_x_maps.begin()
                            + static_cast<size_t>(index) * width_size);
                    fill_opencl_atlas_axis(
                        axis, child_destination_height,
                        child_height, frame.child_zoom);
                    std::copy(
                        axis.begin(), axis.end(),
                        child_y_maps.begin()
                            + static_cast<size_t>(index) * height_size);
                }
            }
        }
        std::lock_guard<std::mutex> lock(runtime.mutex);
        cl_int status = CL_SUCCESS;
        const auto ensure_buffer = [&] (
            cl_mem& buffer,
            size_t& capacity,
            cl_mem_flags flags,
            size_t bytes
        ) {
            if (buffer && capacity >= bytes) return true;
            cl_mem replacement = clCreateBuffer(
                runtime.context, flags, bytes, nullptr, &status);
            if (status != CL_SUCCESS || !replacement) return false;
            if (buffer) clReleaseMemObject(buffer);
            buffer = replacement;
            capacity = bytes;
            if (&buffer == &runtime.atlas_parent) {
                runtime.atlas_parent_cache_token = 0;
                runtime.atlas_parent_cache_bytes = 0;
            } else if (&buffer == &runtime.atlas_child) {
                runtime.atlas_child_cache_token = 0;
                runtime.atlas_child_cache_bytes = 0;
            }
            return true;
        };
        if (image_path
            && (!ensure_opencl_atlas_image(
                    runtime,
                    runtime.atlas_parent_image,
                    runtime.atlas_parent_image_width,
                    runtime.atlas_parent_image_height,
                    parent_width,
                    parent_height,
                    runtime.atlas_parent_image_cache_token,
                    status)
                || (child_available && !ensure_opencl_atlas_image(
                    runtime,
                    runtime.atlas_child_image,
                    runtime.atlas_child_image_width,
                    runtime.atlas_child_image_height,
                    child_width,
                    child_height,
                    runtime.atlas_child_image_cache_token,
                    status)))) {
            image_path = false;
            status = CL_SUCCESS;
        }
        if ((!image_path
                && (!ensure_buffer(
                    runtime.atlas_parent,
                    runtime.atlas_parent_capacity,
                    CL_MEM_READ_ONLY,
                    parent_bytes)
                    || (child_available && !ensure_buffer(
                        runtime.atlas_child,
                        runtime.atlas_child_capacity,
                        CL_MEM_READ_ONLY,
                        child_bytes))))
            || !ensure_buffer(
                runtime.colour_palette,
                runtime.colour_palette_capacity,
                CL_MEM_READ_ONLY,
                palette_data.size())
            || !ensure_buffer(
                runtime.atlas_output,
                runtime.atlas_output_capacity,
                CL_MEM_WRITE_ONLY,
                output_bytes)) {
            throw std::runtime_error(opencl_error_text(status));
        }
        if (runtime.atlas_axis_maps) {
            if (!ensure_buffer(
                    runtime.atlas_parent_x_axis,
                    runtime.atlas_parent_x_axis_capacity,
                    CL_MEM_READ_ONLY,
                    parent_x_maps.size() * sizeof(OpenClAtlasAxis))
                || !ensure_buffer(
                    runtime.atlas_parent_y_axis,
                    runtime.atlas_parent_y_axis_capacity,
                    CL_MEM_READ_ONLY,
                    parent_y_maps.size() * sizeof(OpenClAtlasAxis))
                || (child_available && !ensure_buffer(
                    runtime.atlas_child_x_axis,
                    runtime.atlas_child_x_axis_capacity,
                    CL_MEM_READ_ONLY,
                    child_x_maps.size() * sizeof(OpenClAtlasAxis)))
                || (child_available && !ensure_buffer(
                    runtime.atlas_child_y_axis,
                    runtime.atlas_child_y_axis_capacity,
                    CL_MEM_READ_ONLY,
                    child_y_maps.size() * sizeof(OpenClAtlasAxis)))) {
                throw std::runtime_error(opencl_error_text(status));
            }
        }
        if (image_path) {
            const bool parent_cached = parent_cache_token != 0
                && runtime.atlas_parent_image_cache_token == parent_cache_token;
            if (!parent_cached) {
                parent_image_data = pack_opencl_atlas_image(
                    parent, parent_width, parent_height, parent_max_iter);
                const size_t origin[3] = {0, 0, 0};
                const size_t region[3] = {
                    static_cast<size_t>(parent_width),
                    static_cast<size_t>(parent_height),
                    1,
                };
                status = clEnqueueWriteImage(
                    runtime.queue,
                    runtime.atlas_parent_image,
                    CL_TRUE,
                    origin,
                    region,
                    static_cast<size_t>(parent_width) * 4U * sizeof(float),
                    0,
                    parent_image_data.data(),
                    0,
                    nullptr,
                    nullptr);
                runtime.atlas_parent_image_cache_token = parent_cache_token;
            }
            const bool child_cached = !child_available || (
                child_cache_token != 0
                && runtime.atlas_child_image_cache_token == child_cache_token);
            if (status == CL_SUCCESS && child_available && !child_cached) {
                child_image_data = pack_opencl_atlas_image(
                    child, child_width, child_height, child_max_iter);
                const size_t origin[3] = {0, 0, 0};
                const size_t region[3] = {
                    static_cast<size_t>(child_width),
                    static_cast<size_t>(child_height),
                    1,
                };
                status = clEnqueueWriteImage(
                    runtime.queue,
                    runtime.atlas_child_image,
                    CL_TRUE,
                    origin,
                    region,
                    static_cast<size_t>(child_width) * 4U * sizeof(float),
                    0,
                    child_image_data.data(),
                    0,
                    nullptr,
                    nullptr);
                runtime.atlas_child_image_cache_token = child_cache_token;
            }
        } else {
            const bool parent_cached = parent_cache_token != 0
                && runtime.atlas_parent_cache_token == parent_cache_token
                && runtime.atlas_parent_cache_bytes == parent_bytes;
            if (!parent_cached) {
                status = clEnqueueWriteBuffer(
                    runtime.queue, runtime.atlas_parent, CL_TRUE, 0,
                    parent_bytes, parent, 0, nullptr, nullptr);
                runtime.atlas_parent_cache_token = parent_cache_token;
                runtime.atlas_parent_cache_bytes = parent_cache_token != 0
                    ? parent_bytes : 0;
            }
            const bool child_cached = !child_available || (
                child_cache_token != 0
                && runtime.atlas_child_cache_token == child_cache_token
                && runtime.atlas_child_cache_bytes == child_bytes);
            if (status == CL_SUCCESS && child_available && !child_cached) {
                status = clEnqueueWriteBuffer(
                    runtime.queue, runtime.atlas_child, CL_TRUE, 0,
                    child_bytes, child, 0, nullptr, nullptr);
                runtime.atlas_child_cache_token = child_cache_token;
                runtime.atlas_child_cache_bytes = child_cache_token != 0
                    ? child_bytes : 0;
            }
        }
        if (status == CL_SUCCESS) {
            status = clEnqueueWriteBuffer(
                runtime.queue,
                runtime.colour_palette,
                metadata_upload_blocking,
                0,
                palette_data.size(),
                palette_data.data(),
                0,
                nullptr,
                nullptr);
        }
        if (status == CL_SUCCESS && runtime.atlas_axis_maps) {
            status = clEnqueueWriteBuffer(
                runtime.queue,
                runtime.atlas_parent_x_axis,
                metadata_upload_blocking,
                0,
                parent_x_maps.size() * sizeof(OpenClAtlasAxis),
                parent_x_maps.data(),
                0,
                nullptr,
                nullptr);
        }
        if (status == CL_SUCCESS && runtime.atlas_axis_maps) {
            status = clEnqueueWriteBuffer(
                runtime.queue,
                runtime.atlas_parent_y_axis,
                metadata_upload_blocking,
                0,
                parent_y_maps.size() * sizeof(OpenClAtlasAxis),
                parent_y_maps.data(),
                0,
                nullptr,
                nullptr);
        }
        if (status == CL_SUCCESS && runtime.atlas_axis_maps && child_available) {
            status = clEnqueueWriteBuffer(
                runtime.queue,
                runtime.atlas_child_x_axis,
                metadata_upload_blocking,
                0,
                child_x_maps.size() * sizeof(OpenClAtlasAxis),
                child_x_maps.data(),
                0,
                nullptr,
                nullptr);
        }
        if (status == CL_SUCCESS && runtime.atlas_axis_maps && child_available) {
            status = clEnqueueWriteBuffer(
                runtime.queue,
                runtime.atlas_child_y_axis,
                metadata_upload_blocking,
                0,
                child_y_maps.size() * sizeof(OpenClAtlasAxis),
                child_y_maps.data(),
                0,
                nullptr,
                nullptr);
        }

        cl_mem parent_buffer = image_path
            ? runtime.atlas_parent_image : runtime.atlas_parent;
        cl_mem child_buffer = image_path
            ? (child_available
                ? runtime.atlas_child_image : runtime.atlas_parent_image)
            : (child_available ? runtime.atlas_child : runtime.atlas_parent);
        cl_mem palette_buffer = runtime.colour_palette;
        cl_mem output_buffer = runtime.atlas_output;
        cl_mem parent_x_map = runtime.atlas_parent_x_axis;
        cl_mem parent_y_map = runtime.atlas_parent_y_axis;
        cl_mem child_x_map = child_available
            ? runtime.atlas_child_x_axis : runtime.atlas_parent_x_axis;
        cl_mem child_y_map = child_available
            ? runtime.atlas_child_y_axis : runtime.atlas_parent_y_axis;
        const int child_width_arg = child_available ? child_width : 0;
        const int child_height_arg = child_available ? child_height : 0;
        const int child_max_iter_arg = child_available ? child_max_iter : 0;
        const int interior_red = interior_color != nullptr
            ? interior_color[0] : 0;
        const int interior_green = interior_color != nullptr
            ? interior_color[1] : 0;
        const int interior_blue = interior_color != nullptr
            ? interior_color[2] : 0;
        const auto set_kernel_arg = [&] (
            cl_kernel kernel,
            cl_uint index,
            size_t size,
            const void* value
        ) {
            if (status == CL_SUCCESS && kernel) {
                status = clSetKernelArg(
                    kernel, index, size, value);
            }
        };
        const size_t tiled_workgroup = opencl_atlas_tiled_workgroup(
            runtime.atlas_tiled_colour_workgroup_limit,
            output_pixel_count);
        const bool use_tiled_kernel = !image_path
            && (!runtime.atlas_axis_maps
                || output_pixel_count <= static_cast<size_t>(1920) * 1080)
            && runtime.atlas_tiled_colour_kernel != nullptr
            && tiled_workgroup > 0;
        cl_kernel atlas_kernel = use_tiled_kernel
            ? runtime.atlas_tiled_colour_kernel
            : (image_path
                ? runtime.atlas_image_colour_kernel
                : runtime.atlas_colour_kernel);
        cl_kernel atlas_kernel_secondary = use_tiled_kernel
            ? runtime.atlas_tiled_colour_kernel_secondary
            : (image_path
                ? runtime.atlas_image_colour_kernel_secondary
                : runtime.atlas_colour_kernel_secondary);
        const bool use_secondary_queue =
            frame_count >= 2
            && opencl_atlas_use_secondary_queue()
            && runtime.atlas_queue_secondary != nullptr
            && atlas_kernel_secondary != nullptr;
        const auto set_static_kernel_args = [&](cl_kernel kernel) {
            set_kernel_arg(kernel, 0, sizeof(parent_buffer), &parent_buffer);
            set_kernel_arg(kernel, 1, sizeof(parent_width), &parent_width);
            set_kernel_arg(kernel, 2, sizeof(parent_height), &parent_height);
            set_kernel_arg(kernel, 3, sizeof(parent_max_iter), &parent_max_iter);
            set_kernel_arg(kernel, 4, sizeof(child_buffer), &child_buffer);
            set_kernel_arg(kernel, 5, sizeof(child_width_arg), &child_width_arg);
            set_kernel_arg(kernel, 6, sizeof(child_height_arg), &child_height_arg);
            set_kernel_arg(kernel, 7, sizeof(child_max_iter_arg), &child_max_iter_arg);
            set_kernel_arg(kernel, 8, sizeof(palette_buffer), &palette_buffer);
            set_kernel_arg(kernel, 9, sizeof(output_buffer), &output_buffer);
            set_kernel_arg(kernel, 10, sizeof(output_width), &output_width);
            set_kernel_arg(kernel, 11, sizeof(output_height), &output_height);
            if (runtime.atlas_axis_maps) {
                set_kernel_arg(kernel, 28, sizeof(parent_x_map), &parent_x_map);
                set_kernel_arg(kernel, 29, sizeof(parent_y_map), &parent_y_map);
                set_kernel_arg(kernel, 30, sizeof(child_x_map), &child_x_map);
                set_kernel_arg(kernel, 31, sizeof(child_y_map), &child_y_map);
            }
        };
        set_static_kernel_args(atlas_kernel);
        if (use_secondary_queue) {
            set_static_kernel_args(atlas_kernel_secondary);
        }

        const size_t pixel_count = output_pixel_count;
        size_t workgroup = runtime.atlas_colour_workgroup_size;
        bool use_2d_launch = false;
        size_t global_size_2d[2] = {
            static_cast<size_t>(output_width),
            static_cast<size_t>(output_height),
        };
        size_t global_size_1d = pixel_count;
        size_t local_size_2d[2] = {0, 0};
        if (use_tiled_kernel) {
            workgroup = tiled_workgroup;
            const size_t local_y = workgroup / 16U;
            global_size_2d[0] =
                ((static_cast<size_t>(output_width) + 15U) / 16U) * 16U;
            global_size_2d[1] =
                ((static_cast<size_t>(output_height) + local_y - 1U)
                    / local_y) * local_y;
            local_size_2d[0] = 16U;
            local_size_2d[1] = local_y;
            use_2d_launch = true;
        } else {
            if (!std::getenv("FRACTAL_OPENCL_WORKGROUP")
                && runtime.device_is_gpu
                && workgroup > 0
                && runtime.atlas_colour_workgroup_limit > 0) {
                const size_t target = pixel_count
                    <= static_cast<size_t>(1920) * 1080 ? 256U : 128U;
                if (target >= workgroup
                    && target <= runtime.atlas_colour_workgroup_limit
                    && target % workgroup == 0) {
                    workgroup = target;
                } else if (pixel_count <= static_cast<size_t>(1920) * 1080
                           && workgroup <= runtime.atlas_colour_workgroup_limit / 2) {
                    workgroup *= 2;
                }
            }
            use_2d_launch = pixel_count
                <= static_cast<size_t>(1920) * 1080
                || opencl_atlas_force_2d_launch();
            global_size_2d[0] = workgroup > 0
                ? ((static_cast<size_t>(output_width) + workgroup - 1U)
                    / workgroup) * workgroup
                : static_cast<size_t>(output_width);
            global_size_2d[1] = static_cast<size_t>(output_height);
            global_size_1d = workgroup > 0
                ? ((pixel_count + workgroup - 1U) / workgroup) * workgroup
                : pixel_count;
            local_size_2d[0] = workgroup;
            local_size_2d[1] = 1U;
        }
        const size_t* local_work_size_2d =
            workgroup > 0 ? local_size_2d : nullptr;
        const size_t* local_work_size_1d =
            workgroup > 0 ? &workgroup : nullptr;
        const auto set_frame_kernel_args = [&](cl_kernel kernel, int index) {
            const FractalAtlasColourFrame& frame = frames[index];
            const PaletteInfo& palette = palette_info[static_cast<size_t>(index)];
            const int use_child = use_child_values[static_cast<size_t>(index)];
            const cl_ulong output_offset = static_cast<cl_ulong>(
                static_cast<size_t>(index) * frame_output_bytes);
            const cl_ulong palette_offset = static_cast<cl_ulong>(
                palette.offset);
            set_kernel_arg(kernel, 12, sizeof(frame.parent_zoom), &frame.parent_zoom);
            set_kernel_arg(kernel, 13, sizeof(frame.child_fraction), &frame.child_fraction);
            set_kernel_arg(kernel, 14, sizeof(frame.child_zoom), &frame.child_zoom);
            set_kernel_arg(
                kernel, 15, sizeof(frame.parent_field_bias), &frame.parent_field_bias);
            set_kernel_arg(
                kernel, 16, sizeof(frame.child_field_bias), &frame.child_field_bias);
            set_kernel_arg(
                kernel, 17, sizeof(frame.output_field_bias), &frame.output_field_bias);
            set_kernel_arg(
                kernel, 18, sizeof(palette.effective_iter), &palette.effective_iter);
            set_kernel_arg(kernel, 19, sizeof(frame.feather), &frame.feather);
            set_kernel_arg(kernel, 20, sizeof(palette.size), &palette.size);
            set_kernel_arg(
                kernel, 21, sizeof(palette.index_scale), &palette.index_scale);
            set_kernel_arg(kernel, 22, sizeof(interior_red), &interior_red);
            set_kernel_arg(kernel, 23, sizeof(interior_green), &interior_green);
            set_kernel_arg(kernel, 24, sizeof(interior_blue), &interior_blue);
            set_kernel_arg(kernel, 25, sizeof(use_child), &use_child);
            set_kernel_arg(kernel, 26, sizeof(output_offset), &output_offset);
            set_kernel_arg(kernel, 27, sizeof(palette_offset), &palette_offset);
            if (runtime.atlas_axis_maps) {
                const int parent_x_map_offset = index * output_width;
                const int parent_y_map_offset = index * output_height;
                const int child_x_map_offset = child_available
                    ? index * output_width : parent_x_map_offset;
                const int child_y_map_offset = child_available
                    ? index * output_height : parent_y_map_offset;
                set_kernel_arg(
                    kernel, 32, sizeof(parent_x_map_offset), &parent_x_map_offset);
                set_kernel_arg(
                    kernel, 33, sizeof(parent_y_map_offset), &parent_y_map_offset);
                set_kernel_arg(
                    kernel, 34, sizeof(child_x_map_offset), &child_x_map_offset);
                set_kernel_arg(
                    kernel, 35, sizeof(child_y_map_offset), &child_y_map_offset);
            }
        };
        const auto enqueue_frame = [&] (
            cl_command_queue queue,
            cl_kernel kernel,
            int index
        ) {
            set_frame_kernel_args(kernel, index);
            if (status == CL_SUCCESS) {
                status = clEnqueueNDRangeKernel(
                    queue,
                    kernel,
                    use_2d_launch ? 2 : 1,
                    nullptr,
                    use_2d_launch ? global_size_2d : &global_size_1d,
                    use_2d_launch ? local_work_size_2d : local_work_size_1d,
                    0,
                    nullptr,
                    nullptr);
            }
        };
        if (use_secondary_queue) {
            // Metadata may have been queued asynchronously on the primary
            // queue.  Finish it before the secondary queue starts consuming
            // the same buffers; the frame kernels themselves remain
            // independent because each writes a disjoint output range.
            if (status == CL_SUCCESS) {
                status = clFinish(runtime.queue);
            }
            if (opencl_atlas_use_pipelined_batch()) {
                // Queue every disjoint output slice before asking either
                // queue to read back. Set FRACTAL_OPENCL_ATLAS_PIPELINE=0
                // to retain the pairwise compatibility schedule on an older
                // ICD that is more stable when a read follows each pair.
                for (int index = 0;
                     index < frame_count && status == CL_SUCCESS;
                     ++index) {
                    const bool secondary = (index & 1) != 0;
                    enqueue_frame(
                        secondary ? runtime.atlas_queue_secondary
                                  : runtime.queue,
                        secondary ? atlas_kernel_secondary : atlas_kernel,
                        index);
                }
                std::vector<cl_event> read_events(
                    static_cast<size_t>(frame_count), nullptr);
                for (int index = 0;
                     index < frame_count && status == CL_SUCCESS;
                     ++index) {
                    const bool secondary = (index & 1) != 0;
                    const size_t offset = static_cast<size_t>(index)
                        * frame_output_bytes;
                    status = clEnqueueReadBuffer(
                        secondary ? runtime.atlas_queue_secondary
                                  : runtime.queue,
                        runtime.atlas_output,
                        CL_FALSE,
                        offset,
                        frame_output_bytes,
                        output + offset,
                        0,
                        nullptr,
                        &read_events[static_cast<size_t>(index)]);
                }
                if (status == CL_SUCCESS) {
                    status = clWaitForEvents(
                        static_cast<cl_uint>(frame_count),
                        read_events.data());
                }
                for (cl_event event : read_events) {
                    if (event) clReleaseEvent(event);
                }
            } else {
                for (int index = 0;
                     index < frame_count && status == CL_SUCCESS;
                     index += 2) {
                    enqueue_frame(
                        runtime.queue, atlas_kernel, index);
                    if (index + 1 < frame_count) {
                        enqueue_frame(
                            runtime.atlas_queue_secondary,
                            atlas_kernel_secondary,
                            index + 1);
                    }
                    if (status != CL_SUCCESS) break;
                    const size_t first_offset = static_cast<size_t>(index)
                        * frame_output_bytes;
                    if (index + 1 < frame_count) {
                        const size_t second_offset = static_cast<size_t>(index + 1)
                            * frame_output_bytes;
                        cl_event read_events[2] = {nullptr, nullptr};
                        status = clEnqueueReadBuffer(
                            runtime.queue,
                            runtime.atlas_output,
                            CL_FALSE,
                            first_offset,
                            frame_output_bytes,
                            output + first_offset,
                            0,
                            nullptr,
                            &read_events[0]);
                        if (status == CL_SUCCESS) {
                            status = clEnqueueReadBuffer(
                                runtime.atlas_queue_secondary,
                                runtime.atlas_output,
                                CL_FALSE,
                                second_offset,
                                frame_output_bytes,
                                output + second_offset,
                                0,
                                nullptr,
                                &read_events[1]);
                        }
                        if (status == CL_SUCCESS) {
                            status = clWaitForEvents(2, read_events);
                        }
                        if (read_events[0]) clReleaseEvent(read_events[0]);
                        if (read_events[1]) clReleaseEvent(read_events[1]);
                    } else {
                        status = clEnqueueReadBuffer(
                            runtime.queue,
                            runtime.atlas_output,
                            CL_TRUE,
                            first_offset,
                            frame_output_bytes,
                            output + first_offset,
                            0,
                            nullptr,
                            nullptr);
                    }
                }
            }
        } else {
            for (int index = 0;
                index < frame_count && status == CL_SUCCESS;
                 ++index) {
                enqueue_frame(runtime.queue, atlas_kernel, index);
            }
            if (status == CL_SUCCESS) status = clEnqueueReadBuffer(
                runtime.queue,
                runtime.atlas_output,
                CL_TRUE,
                0,
                output_bytes,
                output,
                0,
                nullptr,
                nullptr);
        }
        if (status != CL_SUCCESS) {
            throw std::runtime_error(opencl_error_text(status));
        }
        set_error("");
        return 0;
    } catch (const std::exception& error) {
        set_error(error.what());
        return 1;
    } catch (...) {
        set_error("batched OpenCL atlas colourizer failed with an unknown exception");
        return 1;
    }
}

int colourise_kfp_opencl_impl(
    const float* field,
    std::uint8_t* output,
    int width,
    int height,
    int max_iter,
    double phase,
    double vocal,
    double instrumental,
    double pitch,
    const FractalKfpOptions* options,
    const std::uint8_t* lut,
    int lut_size,
    int threads
) {
    try {
        if (!field || !output || !valid_pixel_dimensions(width, height)
            || !valid_iteration_count(max_iter) || !valid_thread_count(threads)
            || !valid_colour_controls(phase, vocal, instrumental, pitch)
            || !lut || !valid_kfp_options(options, lut_size)) {
            throw std::runtime_error("invalid OpenCL KFP colour dimensions or controls");
        }
        const FractalKfpOptions& transfer_options = *options;
        // The scalar OpenCL kernel deliberately covers the complete scalar
        // finite-difference/stencil contract, but it cannot carry Kalles'
        // orbit metadata, texture pixels, or multi-colour wave state.
        if (transfer_options.smooth_method != 0
            || transfer_options.multi_color != 0
            || transfer_options.texture_enabled != 0
            || std::abs(transfer_options.phase_color_strength) > 1.0e-12) {
            throw std::runtime_error(
                "OpenCL KFP colourizer supports scalar non-textured profiles only");
        }
        if (transfer_options.field_bias > static_cast<double>(max_iter)) {
            throw std::runtime_error("OpenCL KFP field bias exceeds iteration cap");
        }
        initialise_opencl();
        if (!opencl_kfp_colour_available()) {
            throw std::runtime_error(
                opencl_runtime && !opencl_runtime->error.empty()
                    ? opencl_runtime->error
                    : "OpenCL KFP colourizer is unavailable");
        }
        const double smooth_offset = kfp_smooth_offset(transfer_options);
        double transfer_minimum = 0.0;
        double transfer_maximum = 1.0;
        if (transfer_options.color_method == 4) {
            const KfpTransferBounds bounds = kfp_transfer_bounds(
                field,
                width,
                height,
                max_iter,
                transfer_options.field_bias,
                smooth_offset);
            transfer_minimum = bounds.minimum;
            transfer_maximum = bounds.maximum;
        }
        const KfpSlopeDirection slope_direction = kfp_slope_direction(
            transfer_options);
        const size_t pixel_count = static_cast<size_t>(width)
            * static_cast<size_t>(height);
        const size_t field_bytes = pixel_count * sizeof(float);
        const size_t output_bytes = pixel_count * 3U * sizeof(std::uint8_t);
        const size_t lut_bytes = static_cast<size_t>(lut_size) * 3U;
        OpenClRuntime& runtime = *opencl_runtime;
        std::lock_guard<std::mutex> lock(runtime.mutex);
        cl_int status = CL_SUCCESS;
        const auto ensure_buffer = [&] (
            cl_mem& buffer,
            size_t& capacity,
            cl_mem_flags flags,
            size_t bytes
        ) {
            if (buffer && capacity >= bytes) return true;
            cl_mem replacement = clCreateBuffer(
                runtime.context, flags, bytes, nullptr, &status);
            if (status != CL_SUCCESS || !replacement) return false;
            if (buffer) clReleaseMemObject(buffer);
            buffer = replacement;
            capacity = bytes;
            return true;
        };
        if (!ensure_buffer(runtime.colour_field, runtime.colour_field_capacity,
                           CL_MEM_READ_ONLY, field_bytes)
            || !ensure_buffer(runtime.colour_palette, runtime.colour_palette_capacity,
                              CL_MEM_READ_ONLY, lut_bytes)
            || !ensure_buffer(runtime.colour_output, runtime.colour_output_capacity,
                              CL_MEM_WRITE_ONLY, output_bytes)) {
            throw std::runtime_error(opencl_error_text(status));
        }
        status = clEnqueueWriteBuffer(
            runtime.queue, runtime.colour_field, CL_TRUE, 0,
            field_bytes, field, 0, nullptr, nullptr);
        if (status == CL_SUCCESS) {
            status = clEnqueueWriteBuffer(
                runtime.queue, runtime.colour_palette, CL_TRUE, 0,
                lut_bytes, lut, 0, nullptr, nullptr);
        }

        const int iter_division = transfer_options.iter_div != 1.0 ? 1 : 0;
        const int smooth = transfer_options.smooth;
        const int inverse_transition = transfer_options.inverse_transition;
        const int flat = transfer_options.flat;
        const int slopes = transfer_options.slopes;
        const int differences = transfer_options.differences;
        const int interior_red = transfer_options.interior_color[0];
        const int interior_green = transfer_options.interior_color[1];
        const int interior_blue = transfer_options.interior_color[2];
        const int lut_size_int = lut_size;
        if (status == CL_SUCCESS) status = clSetKernelArg(
            runtime.kfp_colour_kernel, 0, sizeof(runtime.colour_field),
            &runtime.colour_field);
        if (status == CL_SUCCESS) status = clSetKernelArg(
            runtime.kfp_colour_kernel, 1, sizeof(runtime.colour_palette),
            &runtime.colour_palette);
        if (status == CL_SUCCESS) status = clSetKernelArg(
            runtime.kfp_colour_kernel, 2, sizeof(runtime.colour_output),
            &runtime.colour_output);
        if (status == CL_SUCCESS) status = clSetKernelArg(
            runtime.kfp_colour_kernel, 3, sizeof(width), &width);
        if (status == CL_SUCCESS) status = clSetKernelArg(
            runtime.kfp_colour_kernel, 4, sizeof(height), &height);
        if (status == CL_SUCCESS) status = clSetKernelArg(
            runtime.kfp_colour_kernel, 5, sizeof(max_iter), &max_iter);
        if (status == CL_SUCCESS) status = clSetKernelArg(
            runtime.kfp_colour_kernel, 6, sizeof(transfer_options.field_bias),
            &transfer_options.field_bias);
        if (status == CL_SUCCESS) status = clSetKernelArg(
            runtime.kfp_colour_kernel, 7, sizeof(smooth_offset), &smooth_offset);
        if (status == CL_SUCCESS) status = clSetKernelArg(
            runtime.kfp_colour_kernel, 8, sizeof(iter_division), &iter_division);
        if (status == CL_SUCCESS) status = clSetKernelArg(
            runtime.kfp_colour_kernel, 9, sizeof(transfer_options.iter_div),
            &transfer_options.iter_div);
        if (status == CL_SUCCESS) status = clSetKernelArg(
            runtime.kfp_colour_kernel, 10, sizeof(transfer_options.color_offset),
            &transfer_options.color_offset);
        if (status == CL_SUCCESS) status = clSetKernelArg(
            runtime.kfp_colour_kernel, 11, sizeof(transfer_options.color_method),
            &transfer_options.color_method);
        if (status == CL_SUCCESS) status = clSetKernelArg(
            runtime.kfp_colour_kernel, 12, sizeof(smooth), &smooth);
        if (status == CL_SUCCESS) status = clSetKernelArg(
            runtime.kfp_colour_kernel, 13, sizeof(inverse_transition),
            &inverse_transition);
        if (status == CL_SUCCESS) status = clSetKernelArg(
            runtime.kfp_colour_kernel, 14, sizeof(flat), &flat);
        if (status == CL_SUCCESS) status = clSetKernelArg(
            runtime.kfp_colour_kernel, 15, sizeof(slopes), &slopes);
        if (status == CL_SUCCESS) status = clSetKernelArg(
            runtime.kfp_colour_kernel, 16, sizeof(transfer_options.slope_power),
            &transfer_options.slope_power);
        if (status == CL_SUCCESS) status = clSetKernelArg(
            runtime.kfp_colour_kernel, 17, sizeof(transfer_options.slope_ratio),
            &transfer_options.slope_ratio);
        if (status == CL_SUCCESS) status = clSetKernelArg(
            runtime.kfp_colour_kernel, 18, sizeof(slope_direction.cosine),
            &slope_direction.cosine);
        if (status == CL_SUCCESS) status = clSetKernelArg(
            runtime.kfp_colour_kernel, 19, sizeof(slope_direction.sine),
            &slope_direction.sine);
        if (status == CL_SUCCESS) status = clSetKernelArg(
            runtime.kfp_colour_kernel, 20, sizeof(differences), &differences);
        if (status == CL_SUCCESS) status = clSetKernelArg(
            runtime.kfp_colour_kernel, 21, sizeof(interior_red), &interior_red);
        if (status == CL_SUCCESS) status = clSetKernelArg(
            runtime.kfp_colour_kernel, 22, sizeof(interior_green), &interior_green);
        if (status == CL_SUCCESS) status = clSetKernelArg(
            runtime.kfp_colour_kernel, 23, sizeof(interior_blue), &interior_blue);
        if (status == CL_SUCCESS) status = clSetKernelArg(
            runtime.kfp_colour_kernel, 24, sizeof(transfer_minimum),
            &transfer_minimum);
        if (status == CL_SUCCESS) status = clSetKernelArg(
            runtime.kfp_colour_kernel, 25, sizeof(transfer_maximum),
            &transfer_maximum);
        if (status == CL_SUCCESS) status = clSetKernelArg(
            runtime.kfp_colour_kernel, 26, sizeof(lut_size_int), &lut_size_int);

        const size_t workgroup = runtime.kfp_colour_workgroup_size;
        const size_t global_size = workgroup > 0
            ? ((pixel_count + workgroup - 1U) / workgroup) * workgroup
            : pixel_count;
        const size_t* local_work_size = workgroup > 0 ? &workgroup : nullptr;
        if (status == CL_SUCCESS) status = clEnqueueNDRangeKernel(
            runtime.queue, runtime.kfp_colour_kernel, 1, nullptr,
            &global_size, local_work_size, 0, nullptr, nullptr);
        if (status == CL_SUCCESS) status = clEnqueueReadBuffer(
            runtime.queue, runtime.colour_output, CL_TRUE, 0, output_bytes,
            output, 0, nullptr, nullptr);
        if (status != CL_SUCCESS) {
            throw std::runtime_error(opencl_error_text(status));
        }
        set_error("");
        return 0;
    } catch (const std::exception& error) {
        set_error(error.what());
        return 1;
    } catch (...) {
        set_error("OpenCL KFP colourizer failed with an unknown exception");
        return 1;
    }
}

bool opencl_rgb_available() {
    return opencl_available()
        && opencl_runtime != nullptr
        && opencl_runtime->rgb_crop_kernel != nullptr
        && opencl_runtime->rgb_composite_kernel != nullptr;
}

int crop_rgb_opencl_impl(
    const std::uint8_t* source,
    int source_width,
    int source_height,
    std::uint8_t* output,
    int output_width,
    int output_height,
    double zoom_factor,
    int threads
) {
    try {
        if (!source || !output
            || !valid_pixel_dimensions(source_width, source_height)
            || !valid_pixel_dimensions(output_width, output_height)
            || !valid_thread_count(threads)
            || !std::isfinite(zoom_factor) || zoom_factor <= 0.0) {
            throw std::runtime_error("invalid OpenCL RGB crop dimensions");
        }
        initialise_opencl();
        if (!opencl_rgb_available()) {
            throw std::runtime_error(
                opencl_runtime && !opencl_runtime->error.empty()
                    ? opencl_runtime->error
                    : "OpenCL RGB compositor is unavailable");
        }
        OpenClRuntime& runtime = *opencl_runtime;
        std::lock_guard<std::mutex> lock(runtime.mutex);
        const size_t source_bytes = static_cast<size_t>(source_width)
            * static_cast<size_t>(source_height) * 3U;
        const size_t output_bytes = static_cast<size_t>(output_width)
            * static_cast<size_t>(output_height) * 3U;
        cl_int status = CL_SUCCESS;
        const auto ensure_buffer = [&] (
            cl_mem& buffer,
            size_t& capacity,
            cl_mem_flags flags,
            size_t bytes
        ) {
            if (buffer && capacity >= bytes) return true;
            cl_mem replacement = clCreateBuffer(
                runtime.context, flags, bytes, nullptr, &status);
            if (status != CL_SUCCESS || !replacement) return false;
            if (buffer) clReleaseMemObject(buffer);
            buffer = replacement;
            capacity = bytes;
            return true;
        };
        if (!ensure_buffer(runtime.rgb_parent, runtime.rgb_parent_capacity,
                           CL_MEM_READ_ONLY, source_bytes)
            || !ensure_buffer(runtime.rgb_output, runtime.rgb_output_capacity,
                              CL_MEM_WRITE_ONLY, output_bytes)) {
            throw std::runtime_error(opencl_error_text(status));
        }
        status = clEnqueueWriteBuffer(
            runtime.queue, runtime.rgb_parent, CL_TRUE, 0,
            source_bytes, source, 0, nullptr, nullptr);
        // This buffer is shared with the atlas compositor.  A crop call may
        // replace its contents, so invalidate the immutable-tile token before
        // any later cached atlas call can inspect it.
        runtime.rgb_parent_cache_token = 0;
        runtime.rgb_parent_cache_bytes = 0;
        cl_mem source_buffer = runtime.rgb_parent;
        if (status == CL_SUCCESS) status = clSetKernelArg(
            runtime.rgb_crop_kernel, 0, sizeof(source_buffer), &source_buffer);
        if (status == CL_SUCCESS) status = clSetKernelArg(
            runtime.rgb_crop_kernel, 1, sizeof(source_width), &source_width);
        if (status == CL_SUCCESS) status = clSetKernelArg(
            runtime.rgb_crop_kernel, 2, sizeof(source_height), &source_height);
        cl_mem output_buffer = runtime.rgb_output;
        const float zoom_value = static_cast<float>(zoom_factor);
        if (status == CL_SUCCESS) status = clSetKernelArg(
            runtime.rgb_crop_kernel, 3, sizeof(output_buffer), &output_buffer);
        if (status == CL_SUCCESS) status = clSetKernelArg(
            runtime.rgb_crop_kernel, 4, sizeof(output_width), &output_width);
        if (status == CL_SUCCESS) status = clSetKernelArg(
            runtime.rgb_crop_kernel, 5, sizeof(output_height), &output_height);
        if (status == CL_SUCCESS) status = clSetKernelArg(
            runtime.rgb_crop_kernel, 6, sizeof(zoom_value), &zoom_value);
        const size_t pixel_count = static_cast<size_t>(output_width)
            * static_cast<size_t>(output_height);
        const size_t workgroup = runtime.rgb_workgroup_size;
        const size_t global_size = workgroup > 0
            ? ((pixel_count + workgroup - 1U) / workgroup) * workgroup
            : pixel_count;
        const size_t* local_work_size = workgroup > 0 ? &workgroup : nullptr;
        if (status == CL_SUCCESS) status = clEnqueueNDRangeKernel(
            runtime.queue, runtime.rgb_crop_kernel, 1, nullptr,
            &global_size, local_work_size, 0, nullptr, nullptr);
        if (status == CL_SUCCESS) status = clEnqueueReadBuffer(
            runtime.queue, runtime.rgb_output, CL_TRUE, 0, output_bytes,
            output, 0, nullptr, nullptr);
        if (status != CL_SUCCESS) throw std::runtime_error(opencl_error_text(status));
        set_error("");
        return 0;
    } catch (const std::exception& error) {
        set_error(error.what());
        return 1;
    } catch (...) {
        set_error("OpenCL RGB crop failed with an unknown exception");
        return 1;
    }
}

int atlas_composite_rgb_opencl_impl(
    const std::uint8_t* parent,
    int parent_width,
    int parent_height,
    const std::uint8_t* child,
    int child_width,
    int child_height,
    std::uint8_t* output,
    int output_width,
    int output_height,
    double parent_zoom,
    double child_fraction,
    double child_zoom,
    int feather,
    int threads,
    std::uint64_t parent_cache_token = 0,
    std::uint64_t child_cache_token = 0,
    bool reuse_cached_inputs = false
) {
    try {
        if (!parent || !output
            || !valid_pixel_dimensions(parent_width, parent_height)
            || !valid_pixel_dimensions(output_width, output_height)
            || !valid_thread_count(threads)
            || !std::isfinite(parent_zoom) || parent_zoom <= 0.0
            || !std::isfinite(child_fraction)
            || child_fraction < 0.0 || child_fraction > 1.0
            || !std::isfinite(child_zoom) || child_zoom <= 0.0
            || feather < 0) {
            throw std::runtime_error("invalid OpenCL RGB atlas dimensions");
        }
        parent_zoom = std::max(parent_zoom, 1.0);
        child_zoom = std::max(child_zoom, 1.0);
        const bool use_child = child != nullptr && child_fraction > 0.0;
        if (!use_child && child_fraction > 0.0) {
            throw std::runtime_error("OpenCL RGB atlas child is missing");
        }
        if (use_child
            && (!valid_pixel_dimensions(child_width, child_height)
                || feather > std::min(output_width, output_height))) {
            throw std::runtime_error("invalid OpenCL RGB atlas child tile");
        }
        initialise_opencl();
        if (!opencl_rgb_available()) {
            throw std::runtime_error(
                opencl_runtime && !opencl_runtime->error.empty()
                    ? opencl_runtime->error
                    : "OpenCL RGB compositor is unavailable");
        }
        OpenClRuntime& runtime = *opencl_runtime;
        std::lock_guard<std::mutex> lock(runtime.mutex);
        const size_t parent_bytes = static_cast<size_t>(parent_width)
            * static_cast<size_t>(parent_height) * 3U;
        const size_t child_bytes = use_child
            ? static_cast<size_t>(child_width) * static_cast<size_t>(child_height) * 3U
            : 1U;
        const size_t output_bytes = static_cast<size_t>(output_width)
            * static_cast<size_t>(output_height) * 3U;
        cl_int status = CL_SUCCESS;
        const auto ensure_buffer = [&] (
            cl_mem& buffer,
            size_t& capacity,
            cl_mem_flags flags,
            size_t bytes
        ) {
            if (buffer && capacity >= bytes) return true;
            cl_mem replacement = clCreateBuffer(
                runtime.context, flags, bytes, nullptr, &status);
            if (status != CL_SUCCESS || !replacement) return false;
            if (buffer) clReleaseMemObject(buffer);
            buffer = replacement;
            capacity = bytes;
            if (&buffer == &runtime.rgb_parent) {
                runtime.rgb_parent_cache_token = 0;
                runtime.rgb_parent_cache_bytes = 0;
            } else if (&buffer == &runtime.rgb_child) {
                runtime.rgb_child_cache_token = 0;
                runtime.rgb_child_cache_bytes = 0;
            }
            return true;
        };
        if (!ensure_buffer(runtime.rgb_parent, runtime.rgb_parent_capacity,
                           CL_MEM_READ_ONLY, parent_bytes)
            || !ensure_buffer(runtime.rgb_child, runtime.rgb_child_capacity,
                              CL_MEM_READ_ONLY, child_bytes)
            || !ensure_buffer(runtime.rgb_output, runtime.rgb_output_capacity,
                              CL_MEM_WRITE_ONLY, output_bytes)) {
            throw std::runtime_error(opencl_error_text(status));
        }
        if (!reuse_cached_inputs) {
            runtime.rgb_parent_cache_token = 0;
            runtime.rgb_parent_cache_bytes = 0;
            runtime.rgb_child_cache_token = 0;
            runtime.rgb_child_cache_bytes = 0;
        }
        const bool parent_cached = reuse_cached_inputs
            && parent_cache_token != 0
            && runtime.rgb_parent_cache_token == parent_cache_token
            && runtime.rgb_parent_cache_bytes == parent_bytes;
        const bool child_cached = !use_child || (
            reuse_cached_inputs
            && child_cache_token != 0
            && runtime.rgb_child_cache_token == child_cache_token
            && runtime.rgb_child_cache_bytes == child_bytes);
        if (!parent_cached) {
            status = clEnqueueWriteBuffer(
                runtime.queue, runtime.rgb_parent, CL_TRUE, 0,
                parent_bytes, parent, 0, nullptr, nullptr);
            if (status == CL_SUCCESS && reuse_cached_inputs) {
                runtime.rgb_parent_cache_token = parent_cache_token;
                runtime.rgb_parent_cache_bytes = parent_bytes;
            }
        }
        if (status == CL_SUCCESS && use_child && !child_cached) {
            status = clEnqueueWriteBuffer(
                runtime.queue, runtime.rgb_child, CL_TRUE, 0,
                child_bytes, child, 0, nullptr, nullptr);
            if (status == CL_SUCCESS && reuse_cached_inputs) {
                runtime.rgb_child_cache_token = child_cache_token;
                runtime.rgb_child_cache_bytes = child_bytes;
            }
        }
        cl_mem parent_buffer = runtime.rgb_parent;
        cl_mem child_buffer = use_child ? runtime.rgb_child : runtime.rgb_parent;
        cl_mem output_buffer = runtime.rgb_output;
        const int visible_child = use_child ? 1 : 0;
        const float parent_zoom_value = static_cast<float>(parent_zoom);
        const float child_fraction_value = static_cast<float>(child_fraction);
        const float child_zoom_value = static_cast<float>(child_zoom);
        if (status == CL_SUCCESS) status = clSetKernelArg(
            runtime.rgb_composite_kernel, 0, sizeof(parent_buffer), &parent_buffer);
        if (status == CL_SUCCESS) status = clSetKernelArg(
            runtime.rgb_composite_kernel, 1, sizeof(parent_width), &parent_width);
        if (status == CL_SUCCESS) status = clSetKernelArg(
            runtime.rgb_composite_kernel, 2, sizeof(parent_height), &parent_height);
        if (status == CL_SUCCESS) status = clSetKernelArg(
            runtime.rgb_composite_kernel, 3, sizeof(child_buffer), &child_buffer);
        if (status == CL_SUCCESS) status = clSetKernelArg(
            runtime.rgb_composite_kernel, 4, sizeof(child_width), &child_width);
        if (status == CL_SUCCESS) status = clSetKernelArg(
            runtime.rgb_composite_kernel, 5, sizeof(child_height), &child_height);
        if (status == CL_SUCCESS) status = clSetKernelArg(
            runtime.rgb_composite_kernel, 6, sizeof(output_buffer), &output_buffer);
        if (status == CL_SUCCESS) status = clSetKernelArg(
            runtime.rgb_composite_kernel, 7, sizeof(output_width), &output_width);
        if (status == CL_SUCCESS) status = clSetKernelArg(
            runtime.rgb_composite_kernel, 8, sizeof(output_height), &output_height);
        if (status == CL_SUCCESS) status = clSetKernelArg(
            runtime.rgb_composite_kernel, 9, sizeof(parent_zoom_value),
            &parent_zoom_value);
        if (status == CL_SUCCESS) status = clSetKernelArg(
            runtime.rgb_composite_kernel, 10, sizeof(child_fraction_value),
            &child_fraction_value);
        if (status == CL_SUCCESS) status = clSetKernelArg(
            runtime.rgb_composite_kernel, 11, sizeof(child_zoom_value),
            &child_zoom_value);
        if (status == CL_SUCCESS) status = clSetKernelArg(
            runtime.rgb_composite_kernel, 12, sizeof(feather), &feather);
        if (status == CL_SUCCESS) status = clSetKernelArg(
            runtime.rgb_composite_kernel, 13, sizeof(visible_child), &visible_child);
        const size_t pixel_count = static_cast<size_t>(output_width)
            * static_cast<size_t>(output_height);
        const size_t workgroup = runtime.rgb_workgroup_size;
        const size_t global_size = workgroup > 0
            ? ((pixel_count + workgroup - 1U) / workgroup) * workgroup
            : pixel_count;
        const size_t* local_work_size = workgroup > 0 ? &workgroup : nullptr;
        if (status == CL_SUCCESS) status = clEnqueueNDRangeKernel(
            runtime.queue, runtime.rgb_composite_kernel, 1, nullptr,
            &global_size, local_work_size, 0, nullptr, nullptr);
        if (status == CL_SUCCESS) status = clEnqueueReadBuffer(
            runtime.queue, runtime.rgb_output, CL_TRUE, 0, output_bytes,
            output, 0, nullptr, nullptr);
        if (status != CL_SUCCESS) throw std::runtime_error(opencl_error_text(status));
        set_error("");
        return 0;
    } catch (const std::exception& error) {
        set_error(error.what());
        return 1;
    } catch (...) {
        set_error("OpenCL RGB atlas compositor failed with an unknown exception");
        return 1;
    }
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
        const bool opencl_ok = opencl_available();
        if (opencl_ok) {
            capabilities |= 4;
            // This supplementary bit lets front ends prefer a real GPU in
            // automatic mode while still exposing a CPU OpenCL ICD to users
            // who explicitly request it.
            if (opencl_runtime && opencl_runtime->device_is_gpu) capabilities |= 8;
        } else if (opencl_runtime && !opencl_runtime->error.empty()) {
            set_error(opencl_runtime->error);
            return capabilities;
        }
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
    return colourise_field_impl(
        field,
        output,
        width,
        height,
        max_iter,
        phase,
        vocal,
        instrumental,
        pitch,
        threads,
        nullptr,
        nullptr);
}

/* Optional GPU Aurora colour path. KFP and accented palettes intentionally
 * use their specialised CPU APIs instead. */
int fractal_colourise_opencl(
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
#ifdef FRACTAL_HAVE_OPENCL
    return colourise_field_opencl_impl(
        field, output, width, height, max_iter, phase, vocal,
        instrumental, pitch, threads);
#else
    (void)field;
    (void)output;
    (void)width;
    (void)height;
    (void)max_iter;
    (void)phase;
    (void)vocal;
    (void)instrumental;
    (void)pitch;
    (void)threads;
    set_error("OpenCL colourizer is not available in this build");
    return 1;
#endif
}

int fractal_atlas_colourise_opencl(
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
    double child_zoom,
    double parent_field_bias,
    double child_field_bias,
    double output_field_bias,
    int palette_max_iter,
    int feather,
    double phase,
    double vocal,
    double instrumental,
    double pitch,
    int threads,
    std::uint64_t parent_cache_token,
    std::uint64_t child_cache_token
) {
#ifdef FRACTAL_HAVE_OPENCL
    return atlas_colourise_opencl_impl(
        parent, parent_width, parent_height, parent_max_iter,
        child, child_width, child_height, child_max_iter,
        output, output_width, output_height, parent_zoom, child_fraction,
        child_zoom, parent_field_bias, child_field_bias, output_field_bias,
        palette_max_iter, feather, phase, vocal, instrumental, pitch,
        threads, parent_cache_token, child_cache_token, nullptr, nullptr);
#else
    (void)parent;
    (void)parent_width;
    (void)parent_height;
    (void)parent_max_iter;
    (void)child;
    (void)child_width;
    (void)child_height;
    (void)child_max_iter;
    (void)output;
    (void)output_width;
    (void)output_height;
    (void)parent_zoom;
    (void)child_fraction;
    (void)child_zoom;
    (void)parent_field_bias;
    (void)child_field_bias;
    (void)output_field_bias;
    (void)palette_max_iter;
    (void)feather;
    (void)phase;
    (void)vocal;
    (void)instrumental;
    (void)pitch;
    (void)threads;
    (void)parent_cache_token;
    (void)child_cache_token;
    set_error("OpenCL fused atlas colourizer is not available in this build");
    return 1;
#endif
}

int fractal_atlas_colourise_opencl_accents(
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
    double child_zoom,
    double parent_field_bias,
    double child_field_bias,
    double output_field_bias,
    int palette_max_iter,
    int feather,
    double phase,
    double vocal,
    double instrumental,
    double pitch,
    int threads,
    std::uint64_t parent_cache_token,
    std::uint64_t child_cache_token,
    const std::uint8_t* accents,
    int interior_red,
    int interior_green,
    int interior_blue
) {
#ifdef FRACTAL_HAVE_OPENCL
    const int interior_color[3] = {
        interior_red,
        interior_green,
        interior_blue,
    };
    return atlas_colourise_opencl_impl(
        parent, parent_width, parent_height, parent_max_iter,
        child, child_width, child_height, child_max_iter,
        output, output_width, output_height, parent_zoom, child_fraction,
        child_zoom, parent_field_bias, child_field_bias, output_field_bias,
        palette_max_iter, feather, phase, vocal, instrumental, pitch,
        threads, parent_cache_token, child_cache_token, accents,
        interior_color);
#else
    (void)parent;
    (void)parent_width;
    (void)parent_height;
    (void)parent_max_iter;
    (void)child;
    (void)child_width;
    (void)child_height;
    (void)child_max_iter;
    (void)output;
    (void)output_width;
    (void)output_height;
    (void)parent_zoom;
    (void)child_fraction;
    (void)child_zoom;
    (void)parent_field_bias;
    (void)child_field_bias;
    (void)output_field_bias;
    (void)palette_max_iter;
    (void)feather;
    (void)phase;
    (void)vocal;
    (void)instrumental;
    (void)pitch;
    (void)threads;
    (void)parent_cache_token;
    (void)child_cache_token;
    (void)accents;
    (void)interior_red;
    (void)interior_green;
    (void)interior_blue;
    set_error("OpenCL fused atlas colourizer is not available in this build");
    return 1;
#endif
}

int fractal_atlas_colourise_opencl_batch(
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
    const FractalAtlasColourFrame* frames,
    int frame_count,
    int threads,
    std::uint64_t parent_cache_token,
    std::uint64_t child_cache_token,
    const std::uint8_t* accents,
    int interior_red,
    int interior_green,
    int interior_blue
) {
#ifdef FRACTAL_HAVE_OPENCL
    const int interior_color[3] = {
        interior_red,
        interior_green,
        interior_blue,
    };
    return atlas_colourise_opencl_batch_impl(
        parent,
        parent_width,
        parent_height,
        parent_max_iter,
        child,
        child_width,
        child_height,
        child_max_iter,
        output,
        output_width,
        output_height,
        frames,
        frame_count,
        threads,
        parent_cache_token,
        child_cache_token,
        accents,
        interior_color);
#else
    (void)parent;
    (void)parent_width;
    (void)parent_height;
    (void)parent_max_iter;
    (void)child;
    (void)child_width;
    (void)child_height;
    (void)child_max_iter;
    (void)output;
    (void)output_width;
    (void)output_height;
    (void)frames;
    (void)frame_count;
    (void)threads;
    (void)parent_cache_token;
    (void)child_cache_token;
    (void)accents;
    (void)interior_red;
    (void)interior_green;
    (void)interior_blue;
    set_error("OpenCL fused atlas colourizer is not available in this build");
    return 1;
#endif
}

int fractal_colourise_accents(
    const float* field,
    std::uint8_t* output,
    int width,
    int height,
    int max_iter,
    double phase,
    double vocal,
    double instrumental,
    double pitch,
    const std::uint8_t* accents,
    int interior_red,
    int interior_green,
    int interior_blue,
    int threads
) {
    if (!accents) {
        set_error("ordinary palette accents are required");
        return 1;
    }
    const int interior_color[3] = {
        interior_red,
        interior_green,
        interior_blue,
    };
    return colourise_field_impl(
        field,
        output,
        width,
        height,
        max_iter,
        phase,
        vocal,
        instrumental,
        pitch,
        threads,
        interior_color,
        accents);
}

int fractal_apply_aurora_accents(
    std::uint8_t* output,
    int width,
    int height,
    const std::uint8_t* accents,
    double pitch,
    int threads
) {
    try {
        if (!output || !accents || !valid_pixel_dimensions(width, height)
            || !valid_thread_count(threads) || !std::isfinite(pitch)) {
            throw std::runtime_error("invalid Aurora accent dimensions or controls");
        }
        const double hue_angle = pitch_hue_angle(pitch);
        const bool rotate = std::abs(hue_angle) > 1.0e-15;
        const double hue_cos = rotate ? std::cos(hue_angle) : 1.0;
        const double hue_sin = rotate ? std::sin(hue_angle) : 0.0;
        const int pixel_count = width * height;
#ifdef _OPENMP
        if (threads > 0) omp_set_num_threads(threads);
#pragma omp parallel for schedule(static)
#endif
        for (int pixel = 0; pixel < pixel_count; ++pixel) {
            std::uint8_t* destination = output + static_cast<size_t>(pixel) * 3U;
            const double red_weight = std::clamp(
                static_cast<double>(destination[0]) / 140.0, 0.0, 1.0);
            const double green_weight = std::clamp(
                static_cast<double>(destination[1]) / 140.0, 0.0, 1.0);
            const double blue_weight = std::clamp(
                static_cast<double>(destination[2]) / 140.0, 0.0, 1.0);
            double red = (
                red_weight * static_cast<double>(accents[0])
                + green_weight * static_cast<double>(accents[3])
                + blue_weight * static_cast<double>(accents[6])) / 1.8;
            double green = (
                red_weight * static_cast<double>(accents[1])
                + green_weight * static_cast<double>(accents[4])
                + blue_weight * static_cast<double>(accents[7])) / 1.8;
            double blue = (
                red_weight * static_cast<double>(accents[2])
                + green_weight * static_cast<double>(accents[5])
                + blue_weight * static_cast<double>(accents[8])) / 1.8;
            if (rotate) {
                const auto rotated = rotate_hue_rgb(
                    {{red, green, blue}}, hue_cos, hue_sin);
                red = rotated[0];
                green = rotated[1];
                blue = rotated[2];
            }
            destination[0] = rounded_colour_byte(red);
            destination[1] = rounded_colour_byte(green);
            destination[2] = rounded_colour_byte(blue);
        }
        set_error("");
        return 0;
    } catch (const std::exception& error) {
        set_error(error.what());
        return 1;
    } catch (...) {
        set_error("native Aurora accent pass failed with an unknown exception");
        return 1;
    }
}

int fractal_colourise_kfp_impl(
    const float* field,
    std::uint8_t* output,
    int width,
    int height,
    int max_iter,
    double phase,
    double vocal,
    double instrumental,
    double pitch,
    const FractalKfpOptions* options,
    const FractalKfpPlanes* planes,
    const std::uint8_t* lut,
    int lut_size,
    int threads
) {
    try {
        if (!field || !output || !valid_pixel_dimensions(width, height)
            || !valid_iteration_count(max_iter) || !valid_thread_count(threads)
            || !valid_colour_controls(phase, vocal, instrumental, pitch)
            || !lut || !valid_kfp_options(options, lut_size)
            || !valid_kfp_planes(planes, width, height)) {
            throw std::runtime_error("invalid native KFP colour dimensions or controls");
        }
        const FractalKfpOptions& transfer_options = *options;
        if (transfer_options.field_bias > static_cast<double>(max_iter)) {
            throw std::runtime_error("native KFP field bias exceeds iteration cap");
        }
        const double smooth_offset = kfp_smooth_offset(transfer_options);
        const KfpSlopeDirection slope_direction = kfp_slope_direction(transfer_options);
        const KfpPixelContext pixel_context = kfp_pixel_context(
            transfer_options, smooth_offset, planes);
#ifdef _OPENMP
        if (threads > 0) {
            omp_set_dynamic(0);
            omp_set_num_threads(threads);
        }
#endif
        std::vector<double> orbit_colour_samples;
        const double* orbit_colour_samples_data = nullptr;
        std::vector<double> scalar_colour_samples;
        const double* scalar_colour_samples_data = nullptr;
        if (planes != nullptr && planes->orbit_iteration != nullptr
            && (pixel_context.needs_difference
                || pixel_context.needs_slopes
                || pixel_context.texture_active)) {
            // SetColor samples the same continuous orbit value for the
            // centre and for several stencil neighbours. Materialize that
            // value once in double precision: this preserves the exact
            // arithmetic of kfp_plane_orbit_sample while avoiding repeated
            // log/pow work and making the stencil a cache-friendly read.
            const size_t pixel_count = static_cast<size_t>(width)
                * static_cast<size_t>(height);
            orbit_colour_samples.resize(pixel_count);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
            for (int pixel = 0; pixel < width * height; ++pixel) {
                orbit_colour_samples[static_cast<size_t>(pixel)] =
                    kfp_plane_orbit_sample(
                        max_iter,
                        transfer_options,
                        planes,
                        static_cast<size_t>(pixel),
                        &pixel_context).colour;
            }
            orbit_colour_samples_data = orbit_colour_samples.data();
        }
        if (!kfp_planes_have_iteration(planes)
            && (pixel_context.needs_difference
                || pixel_context.needs_slopes
                || pixel_context.texture_active)) {
            // The scalar compatibility field is already the compact value
            // used by the visualizer. Materialize Kalles' continuous color
            // sample once so each stencil neighbour reuses the same exact
            // clamp/offset arithmetic instead of repeating it in the hot
            // pixel loop. This is especially useful for the default KFP,
            // whose difference and relief stages both read the same samples.
            const size_t pixel_count = static_cast<size_t>(width)
                * static_cast<size_t>(height);
            scalar_colour_samples.resize(pixel_count);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
            for (int pixel = 0; pixel < width * height; ++pixel) {
                scalar_colour_samples[static_cast<size_t>(pixel)] =
                    kfp_colour_sample(
                        field,
                        width,
                        height,
                        pixel % width,
                        pixel / width,
                        max_iter,
                        smooth_offset,
                        transfer_options.field_bias);
            }
            scalar_colour_samples_data = scalar_colour_samples.data();
        }
        double transfer_minimum = 0.0;
        double transfer_maximum = 1.0;
        if (transfer_options.color_method == 4) {
            const KfpTransferBounds bounds = planes != nullptr
                ? kfp_transfer_bounds_planes(
                    field,
                    width,
                    height,
                    max_iter,
                    transfer_options,
                    planes,
                    orbit_colour_samples_data)
                : kfp_transfer_bounds(
                    field,
                    width,
                    height,
                    max_iter,
                    transfer_options.field_bias,
                    smooth_offset);
            transfer_minimum = bounds.minimum;
            transfer_maximum = bounds.maximum;
        }
        // The former default branch used approximate transcendental
        // functions and was the reason the bundled palette differed from
        // Kalles even when the scalar field was identical.
        const bool fast_default = false;
        if (fast_default) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
            for (int y = 0; y < height; ++y) {
                int x = 0;
                {
                    std::uint8_t* destination = output + static_cast<size_t>(
                        y * width) * 3U;
                    write_kfp_default_fast_pixel(
                        field,
                        width,
                        height,
                        x,
                        y,
                        x,
                        y,
                        width,
                        max_iter,
                        lut,
                        lut_size,
                        destination,
                        transfer_options.field_bias);
                }
                x = 1;
#if defined(__AVX2__)
                if (y > 0 && y + 1 < height && width >= 10) {
                    const size_t row_offset = static_cast<size_t>(y)
                        * static_cast<size_t>(width) * 3U;
                    for (; x + 7 < width - 1; x += 8) {
                        write_kfp_default_fast_block8(
                        field,
                        width,
                        height,
                        x,
                        y,
                        x,
                        y,
                        width,
                            max_iter,
                            lut,
                            lut_size,
                            output + row_offset,
                            transfer_options.field_bias);
                    }
                }
#endif
                for (; x < width; ++x) {
                    std::uint8_t* destination = output + static_cast<size_t>(
                        y * width + x) * 3U;
                    write_kfp_default_fast_pixel(
                        field,
                        width,
                        height,
                        x,
                        y,
                        x,
                        y,
                        width,
                        max_iter,
                        lut,
                        lut_size,
                        destination,
                        transfer_options.field_bias);
                }
            }
        } else {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
            for (int y = 0; y < height; ++y) {
                for (int x = 0; x < width; ++x) {
                    std::uint8_t* destination = output + static_cast<size_t>(
                        y * width + x) * 3U;
                    write_kfp_pixel(
                        field,
                        width,
                        height,
                        x,
                        y,
                        x,
                        y,
                        width,
                        max_iter,
                        transfer_options,
                        pixel_context,
                        lut,
                        lut_size,
                        transfer_minimum,
                        transfer_maximum,
                        slope_direction,
                        destination,
                        planes,
                        orbit_colour_samples_data,
                        scalar_colour_samples_data);
                }
            }
        }
        set_error("");
        return 0;
    } catch (const std::exception& error) {
        set_error(error.what());
        return 1;
    } catch (...) {
        set_error("native KFP colouriser failed with an unknown exception");
        return 1;
    }
}

int fractal_colourise_kfp(
    const float* field,
    std::uint8_t* output,
    int width,
    int height,
    int max_iter,
    double phase,
    double vocal,
    double instrumental,
    double pitch,
    const FractalKfpOptions* options,
    const std::uint8_t* lut,
    int lut_size,
    int threads
) {
    return fractal_colourise_kfp_impl(
        field, output, width, height, max_iter, phase, vocal,
        instrumental, pitch, options, nullptr, lut, lut_size, threads);
}

/* Optional OpenCL scalar KFP colour pass. Plane-aware, textured, and
 * multi-colour profiles intentionally remain on the exact CPU SetColor path. */
int fractal_colourise_kfp_opencl(
    const float* field,
    std::uint8_t* output,
    int width,
    int height,
    int max_iter,
    double phase,
    double vocal,
    double instrumental,
    double pitch,
    const FractalKfpOptions* options,
    const std::uint8_t* lut,
    int lut_size,
    int threads
) {
#ifdef FRACTAL_HAVE_OPENCL
    return colourise_kfp_opencl_impl(
        field, output, width, height, max_iter, phase, vocal,
        instrumental, pitch, options, lut, lut_size, threads);
#else
    (void)field;
    (void)output;
    (void)width;
    (void)height;
    (void)max_iter;
    (void)phase;
    (void)vocal;
    (void)instrumental;
    (void)pitch;
    (void)options;
    (void)lut;
    (void)lut_size;
    (void)threads;
    set_error("OpenCL KFP colourizer is not available in this build");
    return 1;
#endif
}

/* Optional OpenCL RGB atlas compositor.  The CPU RGB ABI remains available
 * as a fallback for older drivers and for callers that do not request GPU
 * composition. */
int fractal_crop_rgb_opencl(
    const std::uint8_t* source,
    int source_width,
    int source_height,
    std::uint8_t* output,
    int output_width,
    int output_height,
    double zoom_factor,
    int threads
) {
#ifdef FRACTAL_HAVE_OPENCL
    return crop_rgb_opencl_impl(
        source, source_width, source_height, output, output_width,
        output_height, zoom_factor, threads);
#else
    (void)source;
    (void)source_width;
    (void)source_height;
    (void)output;
    (void)output_width;
    (void)output_height;
    (void)zoom_factor;
    (void)threads;
    set_error("OpenCL RGB compositor is not available in this build");
    return 1;
#endif
}

int fractal_atlas_composite_rgb_opencl(
    const std::uint8_t* parent,
    int parent_width,
    int parent_height,
    const std::uint8_t* child,
    int child_width,
    int child_height,
    std::uint8_t* output,
    int output_width,
    int output_height,
    double parent_zoom,
    double child_fraction,
    double child_zoom,
    int feather,
    int threads
) {
#ifdef FRACTAL_HAVE_OPENCL
    return atlas_composite_rgb_opencl_impl(
        parent, parent_width, parent_height, child, child_width,
        child_height, output, output_width, output_height, parent_zoom,
        child_fraction, child_zoom, feather, threads);
#else
    (void)parent;
    (void)parent_width;
    (void)parent_height;
    (void)child;
    (void)child_width;
    (void)child_height;
    (void)output;
    (void)output_width;
    (void)output_height;
    (void)parent_zoom;
    (void)child_fraction;
    (void)child_zoom;
    (void)feather;
    (void)threads;
    set_error("OpenCL RGB compositor is not available in this build");
    return 1;
#endif
}

/* Cached-input variant for the static KFP atlas. The caller promises that
 * the parent/child byte arrays remain unchanged while their tokens remain
 * unchanged; the ordinary entry point above keeps the conservative upload on
 * every call for general ABI users. */
int fractal_atlas_composite_rgb_opencl_cached(
    const std::uint8_t* parent,
    int parent_width,
    int parent_height,
    const std::uint8_t* child,
    int child_width,
    int child_height,
    std::uint8_t* output,
    int output_width,
    int output_height,
    double parent_zoom,
    double child_fraction,
    double child_zoom,
    int feather,
    int threads,
    std::uint64_t parent_cache_token,
    std::uint64_t child_cache_token
) {
#ifdef FRACTAL_HAVE_OPENCL
    return atlas_composite_rgb_opencl_impl(
        parent, parent_width, parent_height, child, child_width,
        child_height, output, output_width, output_height, parent_zoom,
        child_fraction, child_zoom, feather, threads,
        parent_cache_token, child_cache_token, true);
#else
    (void)parent;
    (void)parent_width;
    (void)parent_height;
    (void)child;
    (void)child_width;
    (void)child_height;
    (void)output;
    (void)output_width;
    (void)output_height;
    (void)parent_zoom;
    (void)child_fraction;
    (void)child_zoom;
    (void)feather;
    (void)threads;
    (void)parent_cache_token;
    (void)child_cache_token;
    set_error("OpenCL RGB compositor is not available in this build");
    return 1;
#endif
}

int fractal_colourise_kfp_planes(
    const float* field,
    std::uint8_t* output,
    int width,
    int height,
    int max_iter,
    double phase,
    double vocal,
    double instrumental,
    double pitch,
    const FractalKfpOptions* options,
    const FractalKfpPlanes* planes,
    const std::uint8_t* lut,
    int lut_size,
    int threads
) {
    return fractal_colourise_kfp_impl(
        field, output, width, height, max_iter, phase, vocal,
        instrumental, pitch, options, planes, lut, lut_size, threads);
}

int fractal_atlas_colourise_kfp(
    const float* parent,
    int parent_width,
    int parent_height,
    const float* child,
    int child_width,
    int child_height,
    std::uint8_t* output,
    int output_width,
    int output_height,
    int max_iter,
    int child_left,
    int child_top,
    int feather,
    double phase,
    double vocal,
    double instrumental,
    double pitch,
    const FractalKfpOptions* options,
    const std::uint8_t* lut,
    int lut_size,
    int threads
) {
    try {
        if (!parent || !output
            || !valid_pixel_dimensions(parent_width, parent_height)
            || !valid_pixel_dimensions(output_width, output_height)
            || parent_width != output_width
            || parent_height != output_height
            || !valid_iteration_count(max_iter)
            || !valid_thread_count(threads)
            || !valid_colour_controls(phase, vocal, instrumental, pitch)
            || !lut || !valid_kfp_options(options, lut_size)) {
            throw std::runtime_error("invalid native KFP atlas dimensions or controls");
        }

        const bool use_child = child != nullptr;
        if (use_child) {
            if (!valid_pixel_dimensions(child_width, child_height)
                || child_left < 0 || child_top < 0
                || child_width > output_width - child_left
                || child_height > output_height - child_top
                || feather < 0
                || feather > std::min(child_width, child_height)) {
                throw std::runtime_error("invalid native KFP atlas child rectangle");
            }
        } else if (child_width != 0 || child_height != 0
                   || child_left != 0 || child_top != 0 || feather != 0) {
            throw std::runtime_error("invalid native KFP atlas child absence");
        }

        if (threads > 0) {
#ifdef _OPENMP
            omp_set_num_threads(threads);
#endif
        }
        const FractalKfpOptions& transfer_options = *options;
        if (transfer_options.field_bias > static_cast<double>(max_iter)) {
            throw std::runtime_error("native KFP field bias exceeds iteration cap");
        }
        const double smooth_offset = kfp_smooth_offset(transfer_options);
        const KfpSlopeDirection slope_direction = kfp_slope_direction(transfer_options);
        const KfpPixelContext pixel_context = kfp_pixel_context(
            transfer_options, smooth_offset, nullptr);
        const bool fast_default = false;
        const KfpTransferBounds parent_bounds = transfer_options.color_method == 4
            ? kfp_transfer_bounds(
                parent,
                parent_width,
                parent_height,
                max_iter,
                transfer_options.field_bias,
                smooth_offset)
            : KfpTransferBounds{};
        const KfpTransferBounds child_bounds = use_child && transfer_options.color_method == 4
            ? kfp_transfer_bounds(
                child,
                child_width,
                child_height,
                max_iter,
                transfer_options.field_bias,
                smooth_offset)
            : KfpTransferBounds{};

        std::vector<double> parent_colour_samples;
        std::vector<double> child_colour_samples;
        const double* parent_colour_samples_data = nullptr;
        const double* child_colour_samples_data = nullptr;
        if (pixel_context.needs_difference || pixel_context.needs_slopes) {
            // Atlas tiles use the same Kalles stencil as a direct frame, but
            // each source has its own local edge. Cache the exact continuous
            // source samples once per tile so every parent/child pixel does
            // not redo the scalar clamp and smoothing offset for each
            // neighbour read.
            const size_t parent_count = static_cast<size_t>(parent_width)
                * static_cast<size_t>(parent_height);
            parent_colour_samples.resize(parent_count);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
            for (int pixel = 0; pixel < parent_width * parent_height; ++pixel) {
                parent_colour_samples[static_cast<size_t>(pixel)] =
                    kfp_colour_sample(
                        parent,
                        parent_width,
                        parent_height,
                        pixel % parent_width,
                        pixel / parent_width,
                        max_iter,
                        smooth_offset,
                        transfer_options.field_bias);
            }
            parent_colour_samples_data = parent_colour_samples.data();
            if (use_child) {
                const size_t child_count = static_cast<size_t>(child_width)
                    * static_cast<size_t>(child_height);
                child_colour_samples.resize(child_count);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
                for (int pixel = 0; pixel < child_width * child_height; ++pixel) {
                    child_colour_samples[static_cast<size_t>(pixel)] =
                        kfp_colour_sample(
                            child,
                            child_width,
                            child_height,
                            pixel % child_width,
                            pixel / child_width,
                            max_iter,
                            smooth_offset,
                            transfer_options.field_bias);
                }
                child_colour_samples_data = child_colour_samples.data();
            }
        }

        std::vector<float> child_edge_x;
        std::vector<float> child_edge_y;
        if (use_child && feather >= 2) {
            child_edge_x.resize(static_cast<size_t>(child_width));
            child_edge_y.resize(static_cast<size_t>(child_height));
            for (int x = 0; x < child_width; ++x) {
                const int edge = std::min(x, child_width - 1 - x);
                const float linear = std::min(
                    1.0F, static_cast<float>(edge) / static_cast<float>(feather));
                child_edge_x[static_cast<size_t>(x)] = linear * linear
                    * (3.0F - 2.0F * linear);
            }
            for (int y = 0; y < child_height; ++y) {
                const int edge = std::min(y, child_height - 1 - y);
                const float linear = std::min(
                    1.0F, static_cast<float>(edge) / static_cast<float>(feather));
                child_edge_y[static_cast<size_t>(y)] = linear * linear
                    * (3.0F - 2.0F * linear);
            }
        }

        if (fast_default) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
            for (int y = 0; y < output_height; ++y) {
                const size_t row_offset = static_cast<size_t>(y)
                    * static_cast<size_t>(output_width) * 3U;
                int x = 0;
                while (x < output_width) {
                    const bool in_child = use_child
                        && x >= child_left && x < child_left + child_width
                        && y >= child_top && y < child_top + child_height;
                    if (!in_child) {
                        const int segment_end = use_child && x < child_left
                            ? child_left : output_width;
                        if (y > 0 && y + 1 < output_height && x > 0
                            && x + 7 < segment_end && x + 7 < output_width - 1) {
                            write_kfp_default_fast_block8(
                                parent,
                                parent_width,
                                parent_height,
                                x,
                                y,
                                x,
                                y,
                                output_width,
                                max_iter,
                                lut,
                                lut_size,
                                output + row_offset,
                                transfer_options.field_bias);
                            x += 8;
                            continue;
                        }
                        write_kfp_default_fast_pixel(
                            parent,
                            parent_width,
                            parent_height,
                            x,
                            y,
                            x,
                            y,
                            output_width,
                            max_iter,
                            lut,
                            lut_size,
                            output + (static_cast<size_t>(y) * static_cast<size_t>(output_width)
                                + static_cast<size_t>(x)) * 3U,
                            transfer_options.field_bias);
                        ++x;
                        continue;
                    }

                    const int child_x = x - child_left;
                    const int child_y = y - child_top;
                    const float alpha = feather >= 2
                        ? std::min(
                            child_edge_x[static_cast<size_t>(child_x)],
                            child_edge_y[static_cast<size_t>(child_y)])
                        : 1.0F;
                    if (alpha >= 0.999999F) {
                        const int full_end = feather >= 2
                            ? child_left + child_width - feather
                            : child_left + child_width;
                        if (child_y > 0 && child_y + 1 < child_height
                            && child_x > 0 && child_x + 7 < child_width - 1
                            && x + 7 < full_end) {
                            write_kfp_default_fast_block8(
                                child,
                                child_width,
                                child_height,
                                child_x,
                                child_y,
                                x,
                                y,
                                output_width,
                                max_iter,
                                lut,
                                lut_size,
                                output + row_offset,
                                transfer_options.field_bias);
                            x += 8;
                            continue;
                        }
                        write_kfp_default_fast_pixel(
                            child,
                            child_width,
                            child_height,
                            child_x,
                            child_y,
                            x,
                            y,
                            output_width,
                            max_iter,
                            lut,
                            lut_size,
                            output + (static_cast<size_t>(y) * static_cast<size_t>(output_width)
                                + static_cast<size_t>(x)) * 3U,
                            transfer_options.field_bias);
                        ++x;
                        continue;
                    }

                    std::uint8_t* destination = output + (
                        static_cast<size_t>(y) * static_cast<size_t>(output_width)
                        + static_cast<size_t>(x)) * 3U;
                    if (feather < 2) {
                        const float parent_raw = parent[static_cast<size_t>(y)
                            * static_cast<size_t>(parent_width) + static_cast<size_t>(x)];
                        if (!kfp_inside_value(
                            parent_raw,
                            max_iter,
                            transfer_options.field_bias)) {
                            write_kfp_default_fast_pixel(
                                parent,
                                parent_width,
                                parent_height,
                                x,
                                y,
                                x,
                                y,
                                output_width,
                                max_iter,
                                lut,
                                lut_size,
                                destination,
                                transfer_options.field_bias);
                        } else {
                            write_kfp_default_fast_pixel(
                                child,
                                child_width,
                                child_height,
                                child_x,
                                child_y,
                                x,
                                y,
                                output_width,
                                max_iter,
                                lut,
                                lut_size,
                                destination,
                                transfer_options.field_bias);
                        }
                        ++x;
                        continue;
                    }

                    std::uint8_t parent_rgb[3];
                    std::uint8_t child_rgb[3];
                    write_kfp_default_fast_pixel(
                        parent,
                        parent_width,
                        parent_height,
                        x,
                        y,
                        x,
                        y,
                        output_width,
                        max_iter,
                        lut,
                        lut_size,
                        parent_rgb,
                        transfer_options.field_bias);
                    write_kfp_default_fast_pixel(
                        child,
                        child_width,
                        child_height,
                        child_x,
                        child_y,
                        x,
                        y,
                        output_width,
                        max_iter,
                        lut,
                        lut_size,
                        child_rgb,
                        transfer_options.field_bias);
                    destination[0] = rounded_colour_byte(
                        static_cast<double>(parent_rgb[0]) * (1.0 - alpha)
                        + static_cast<double>(child_rgb[0]) * alpha);
                    destination[1] = rounded_colour_byte(
                        static_cast<double>(parent_rgb[1]) * (1.0 - alpha)
                        + static_cast<double>(child_rgb[1]) * alpha);
                    destination[2] = rounded_colour_byte(
                        static_cast<double>(parent_rgb[2]) * (1.0 - alpha)
                        + static_cast<double>(child_rgb[2]) * alpha);
                    ++x;
                }
            }
        } else {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
            for (int y = 0; y < output_height; ++y) {
                for (int x = 0; x < output_width; ++x) {
                std::uint8_t* destination = output + static_cast<size_t>(
                    y * output_width + x) * 3U;
                const bool in_child = use_child
                    && x >= child_left && x < child_left + child_width
                    && y >= child_top && y < child_top + child_height;
                if (!in_child) {
                    write_kfp_pixel(
                        parent,
                        parent_width,
                        parent_height,
                        x,
                        y,
                        x,
                        y,
                        output_width,
                        max_iter,
                        transfer_options,
                        pixel_context,
                        lut,
                        lut_size,
                        parent_bounds.minimum,
                        parent_bounds.maximum,
                        slope_direction,
                        destination,
                        nullptr,
                        nullptr,
                        parent_colour_samples_data);
                    continue;
                }

                const int child_x = x - child_left;
                const int child_y = y - child_top;
                const float child_raw = child[static_cast<size_t>(child_y)
                    * static_cast<size_t>(child_width) + static_cast<size_t>(child_x)];
                const bool child_inside = kfp_inside_value(
                    child_raw,
                    max_iter,
                    transfer_options.field_bias);
                const float alpha = feather >= 2
                    ? std::min(
                        child_edge_x[static_cast<size_t>(child_x)],
                        child_edge_y[static_cast<size_t>(child_y)])
                    : 1.0F;

                // The deeper tile owns classification. This rule is what
                // prevents an interior child from becoming a rectangular
                // parent-coloured fill in the fully-owned region. In the
                // feather band both classifications are deliberately blended
                // so an interior cannot draw a hard rectangle.
                if (child_inside && alpha >= 0.999999F) {
                    write_kfp_pixel(
                        child,
                        child_width,
                        child_height,
                        child_x,
                        child_y,
                        x,
                        y,
                        output_width,
                        max_iter,
                        transfer_options,
                        pixel_context,
                        lut,
                        lut_size,
                        child_bounds.minimum,
                        child_bounds.maximum,
                        slope_direction,
                        destination,
                        nullptr,
                        nullptr,
                        child_colour_samples_data);
                    continue;
                }

                if (feather < 2) {
                    const float parent_raw = parent[static_cast<size_t>(y)
                        * static_cast<size_t>(parent_width) + static_cast<size_t>(x)];
                    if (!kfp_inside_value(
                        parent_raw,
                        max_iter,
                        transfer_options.field_bias)) {
                        write_kfp_pixel(
                            parent,
                            parent_width,
                            parent_height,
                            x,
                            y,
                            x,
                            y,
                            output_width,
                            max_iter,
                            transfer_options,
                            pixel_context,
                            lut,
                            lut_size,
                            parent_bounds.minimum,
                            parent_bounds.maximum,
                            slope_direction,
                            destination,
                            nullptr,
                            nullptr,
                            parent_colour_samples_data);
                    } else {
                        write_kfp_pixel(
                            child,
                            child_width,
                            child_height,
                            child_x,
                            child_y,
                            x,
                            y,
                            output_width,
                            max_iter,
                            transfer_options,
                            pixel_context,
                            lut,
                            lut_size,
                            child_bounds.minimum,
                            child_bounds.maximum,
                            slope_direction,
                            destination,
                            nullptr,
                            nullptr,
                            child_colour_samples_data);
                    }
                    continue;
                }

                if (alpha >= 0.999999F) {
                    write_kfp_pixel(
                        child,
                        child_width,
                        child_height,
                        child_x,
                        child_y,
                        x,
                        y,
                        output_width,
                        max_iter,
                        transfer_options,
                        pixel_context,
                        lut,
                        lut_size,
                        child_bounds.minimum,
                        child_bounds.maximum,
                        slope_direction,
                        destination,
                        nullptr,
                        nullptr,
                        child_colour_samples_data);
                    continue;
                }


                std::uint8_t parent_rgb[3];
                std::uint8_t child_rgb[3];
                write_kfp_pixel(
                    parent,
                    parent_width,
                    parent_height,
                    x,
                    y,
                    x,
                    y,
                    output_width,
                    max_iter,
                    transfer_options,
                    pixel_context,
                    lut,
                    lut_size,
                    parent_bounds.minimum,
                    parent_bounds.maximum,
                    slope_direction,
                    parent_rgb,
                    nullptr,
                    nullptr,
                    parent_colour_samples_data);
                write_kfp_pixel(
                    child,
                    child_width,
                    child_height,
                    child_x,
                    child_y,
                    x,
                    y,
                    output_width,
                    max_iter,
                    transfer_options,
                    pixel_context,
                    lut,
                    lut_size,
                    child_bounds.minimum,
                    child_bounds.maximum,
                    slope_direction,
                    child_rgb,
                    nullptr,
                    nullptr,
                    child_colour_samples_data);
                destination[0] = rounded_colour_byte(
                    static_cast<double>(parent_rgb[0]) * (1.0 - alpha)
                    + static_cast<double>(child_rgb[0]) * alpha);
                destination[1] = rounded_colour_byte(
                    static_cast<double>(parent_rgb[1]) * (1.0 - alpha)
                    + static_cast<double>(child_rgb[1]) * alpha);
                destination[2] = rounded_colour_byte(
                    static_cast<double>(parent_rgb[2]) * (1.0 - alpha)
                    + static_cast<double>(child_rgb[2]) * alpha);
            }
        }
        }
        set_error("");
        return 0;
    } catch (const std::exception& error) {
        set_error(error.what());
        return 1;
    } catch (...) {
        set_error("native KFP atlas colouriser failed with an unknown exception");
        return 1;
    }
}

int fractal_atlas_colourise_kfp_planes(
    const float* parent,
    int parent_width,
    int parent_height,
    int parent_max_iter,
    const FractalKfpPlanes* parent_planes,
    const float* child,
    int child_width,
    int child_height,
    int child_max_iter,
    const FractalKfpPlanes* child_planes,
    std::uint8_t* output,
    int output_width,
    int output_height,
    double parent_zoom,
    double child_fraction,
    double child_zoom,
    int max_iter,
    int centered_input,
    double phase,
    double vocal,
    double instrumental,
    double pitch,
    const FractalKfpOptions* options,
    const std::uint8_t* lut,
    int lut_size,
    int threads
) {
    try {
        if (!parent || !output || !parent_planes
            || !valid_pixel_dimensions(parent_width, parent_height)
            || !valid_pixel_dimensions(output_width, output_height)
            || !valid_iteration_count(parent_max_iter)
            || !valid_iteration_count(max_iter)
            || !valid_thread_count(threads)
            || centered_input < 0 || centered_input > 1
            || !std::isfinite(parent_zoom) || parent_zoom <= 0.0
            || !std::isfinite(child_fraction)
            || child_fraction < 0.0 || child_fraction > 1.0
            || !std::isfinite(child_zoom) || child_zoom <= 0.0
            || !valid_colour_controls(phase, vocal, instrumental, pitch)
            || !lut || !valid_kfp_options(options, lut_size)
            || !valid_kfp_planes(parent_planes, parent_width, parent_height)) {
            throw std::runtime_error(
                "invalid native plane-aware KFP atlas dimensions or controls");
        }

        const bool use_child = child != nullptr && child_fraction > 0.0;
        if (use_child) {
            if (!valid_pixel_dimensions(child_width, child_height)
                || !valid_iteration_count(child_max_iter)
                || !child_planes
                || !valid_kfp_planes(child_planes, child_width, child_height)) {
                throw std::runtime_error(
                    "invalid native plane-aware KFP atlas child tile");
            }
        } else if (child != nullptr || child_planes != nullptr
                   || child_width != 0 || child_height != 0
                   || child_max_iter != 0) {
            throw std::runtime_error(
                "invalid native plane-aware KFP atlas child absence");
        }

        const FractalKfpOptions& transfer_options = *options;
        if (transfer_options.field_bias > static_cast<double>(max_iter)) {
            throw std::runtime_error("native KFP field bias exceeds iteration cap");
        }
        parent_zoom = std::max(parent_zoom, 1.0);
        child_zoom = std::max(child_zoom, 1.0);

        const bool full_child = use_child && child_fraction >= 0.999999;
        const float* primary_field = full_child ? child : parent;
        const int primary_width = full_child ? child_width : parent_width;
        const int primary_height = full_child ? child_height : parent_height;
        const int primary_max_iter = full_child ? child_max_iter : parent_max_iter;
        const double primary_zoom = full_child ? child_zoom : parent_zoom;
        const FractalKfpPlanes* primary_planes = full_child
            ? child_planes : parent_planes;
        const float* secondary_field = !full_child && use_child ? child : nullptr;
        const int secondary_width = !full_child && use_child ? child_width : 0;
        const int secondary_height = !full_child && use_child ? child_height : 0;
        const int secondary_max_iter = !full_child && use_child
            ? child_max_iter : 0;
        const double secondary_zoom = child_zoom;
        const FractalKfpPlanes* secondary_planes = !full_child && use_child
            ? child_planes : nullptr;
        const int target_max_iter = full_child ? child_max_iter : max_iter;
        const double primary_field_bias = centered_input
            ? static_cast<double>(primary_max_iter) : 0.0;
        const double secondary_field_bias = centered_input
            ? static_cast<double>(secondary_max_iter) : 0.0;

        const int visible_child_width = secondary_field != nullptr
            ? std::max(1, static_cast<int>(std::lround(
                static_cast<double>(output_width) * child_fraction))) : 0;
        const int visible_child_height = secondary_field != nullptr
            ? std::max(1, static_cast<int>(std::lround(
                static_cast<double>(output_height) * child_fraction))) : 0;
        const int child_left = secondary_field != nullptr
            ? (output_width - visible_child_width) / 2 : 0;
        const int child_top = secondary_field != nullptr
            ? (output_height - visible_child_height) / 2 : 0;
        const int seam_feather = secondary_field != nullptr
            ? kfp_atlas_seam_feather(visible_child_width, visible_child_height)
            : 0;

        BilinearWorkspace& workspace = bilinear_workspace;
        BilinearAxis& primary_x_axis = workspace.parent_x_axis;
        BilinearAxis& primary_y_axis = workspace.parent_y_axis;
        fill_bilinear_axis(
            primary_x_axis, primary_width, output_width, primary_zoom);
        fill_bilinear_axis(
            primary_y_axis, primary_height, output_height, primary_zoom);
        BilinearAxis& secondary_x_axis = workspace.child_x_axis;
        BilinearAxis& secondary_y_axis = workspace.child_y_axis;
        if (secondary_field != nullptr) {
            fill_bilinear_axis(
                secondary_x_axis, secondary_width, visible_child_width,
                secondary_zoom);
            fill_bilinear_axis(
                secondary_y_axis, secondary_height, visible_child_height,
                secondary_zoom);
        }

        std::vector<float>& output_field = workspace.kfp_parent_field;
        const size_t pixel_count = static_cast<size_t>(output_width)
            * static_cast<size_t>(output_height);
        output_field.resize(pixel_count);
        FractalKfpPlanes output_planes{};
        output_planes.struct_size = sizeof(FractalKfpPlanes);
        output_planes.version = FRACTAL_KFP_PLANES_VERSION;

        // Bind only the planes that actually exist in either source. This is
        // important for live mode: a profile can omit phase/DE data and must
        // not pay for several extra full-resolution double buffers.
        auto bind_int_plane = [&](
            const std::int64_t* FractalKfpPlanes::* member,
            std::vector<std::int64_t>& storage
        ) {
            const std::int64_t* primary = primary_planes->*member;
            const std::int64_t* secondary = secondary_planes != nullptr
                ? secondary_planes->*member : nullptr;
            if (!primary && !secondary) return;
            storage.resize(pixel_count);
            if (!primary) std::fill(storage.begin(), storage.end(), 0);
            output_planes.*member = storage.data();
        };
        auto bind_double_plane = [&](
            const double* FractalKfpPlanes::* member,
            std::vector<double>& storage
        ) {
            const double* primary = primary_planes->*member;
            const double* secondary = secondary_planes != nullptr
                ? secondary_planes->*member : nullptr;
            if (!primary && !secondary) return;
            storage.resize(pixel_count);
            if (!primary) std::fill(storage.begin(), storage.end(), 0.0);
            output_planes.*member = storage.data();
        };
        bind_int_plane(
            &FractalKfpPlanes::orbit_iteration,
            workspace.kfp_atlas_orbit_iteration);
        bind_int_plane(
            &FractalKfpPlanes::iteration,
            workspace.kfp_atlas_iteration);
        bind_double_plane(
            &FractalKfpPlanes::bailout,
            workspace.kfp_atlas_bailout);
        bind_double_plane(
            &FractalKfpPlanes::transition,
            workspace.kfp_atlas_transition);
        bind_double_plane(
            &FractalKfpPlanes::phase,
            workspace.kfp_atlas_phase);
        bind_double_plane(
            &FractalKfpPlanes::de_x,
            workspace.kfp_atlas_de_x);
        bind_double_plane(
            &FractalKfpPlanes::de_y,
            workspace.kfp_atlas_de_y);
        bind_double_plane(
            &FractalKfpPlanes::test1,
            workspace.kfp_atlas_test1);
        bind_double_plane(
            &FractalKfpPlanes::test2,
            workspace.kfp_atlas_test2);

        // Texture coordinates are evaluated in final screen space. Keep one
        // source image attached to the output plane bundle rather than
        // attempting to crop or blend it with the atlas tiles.
        const FractalKfpPlanes* texture_planes = primary_planes;
        if (texture_planes->texture_rgb == nullptr
            && secondary_planes != nullptr
            && secondary_planes->texture_rgb != nullptr) {
            texture_planes = secondary_planes;
        }
        output_planes.texture_rgb = texture_planes->texture_rgb;
        output_planes.texture_width = texture_planes->texture_width;
        output_planes.texture_height = texture_planes->texture_height;
        output_planes.texture_stride = texture_planes->texture_stride;

        auto primary_scalar = [&](int x, int y, bool& inside) {
            const float smooth = sample_bilinear_mapped_preserving_interior(
                primary_field,
                primary_width,
                primary_x_axis,
                primary_y_axis,
                x,
                y,
                primary_max_iter,
                inside,
                primary_field_bias);
            return inside
                ? encode_render_iteration(target_max_iter, transfer_options.field_bias)
                : encode_render_value(
                    static_cast<long double>(smooth),
                    transfer_options.field_bias);
        };
        auto secondary_scalar = [&](int x, int y, bool& inside) {
            const float smooth = sample_bilinear_mapped_preserving_interior(
                secondary_field,
                secondary_width,
                secondary_x_axis,
                secondary_y_axis,
                x,
                y,
                secondary_max_iter,
                inside,
                secondary_field_bias);
            return inside
                ? encode_render_iteration(target_max_iter, transfer_options.field_bias)
                : encode_render_value(
                    static_cast<long double>(smooth),
                    transfer_options.field_bias);
        };

        auto write_primary_int_plane = [&](
            const std::int64_t* source,
            std::int64_t* destination,
            int x,
            int y,
            bool inside
        ) {
            if (!source || !destination) return;
            destination[static_cast<size_t>(y) * static_cast<size_t>(output_width)
                + static_cast<size_t>(x)] = inside
                ? static_cast<std::int64_t>(target_max_iter)
                : sample_nearest_mapped_int64(
                    source, primary_width, primary_x_axis,
                    primary_y_axis, x, y);
        };
        auto write_primary_double_plane = [&](
            const double* source,
            double* destination,
            bool preserve_negative,
            bool reset_inside,
            int x,
            int y,
            bool inside
        ) {
            if (!source || !destination) return;
            double value = sample_bilinear_mapped_double(
                source, primary_width, primary_x_axis, primary_y_axis,
                x, y, preserve_negative);
            if (preserve_negative) {
                const double nearest = sample_nearest_mapped_double(
                    source, primary_width, primary_x_axis, primary_y_axis,
                    x, y);
                if (nearest < 0.0) value = nearest;
            }
            destination[static_cast<size_t>(y) * static_cast<size_t>(output_width)
                + static_cast<size_t>(x)] = inside && reset_inside ? 0.0 : value;
        };

        const std::int64_t* parent_orbit = primary_planes->orbit_iteration;
        const std::int64_t* parent_iteration = primary_planes->iteration;
        const double* parent_bailout = primary_planes->bailout;
        const double* parent_transition = primary_planes->transition;
        const double* parent_phase = primary_planes->phase;
        const double* parent_de_x = primary_planes->de_x;
        const double* parent_de_y = primary_planes->de_y;
        const double* parent_test1 = primary_planes->test1;
        const double* parent_test2 = primary_planes->test2;

        std::int64_t* output_orbit = const_cast<std::int64_t*>(
            output_planes.orbit_iteration);
        std::int64_t* output_iteration = const_cast<std::int64_t*>(
            output_planes.iteration);
        double* output_bailout = const_cast<double*>(output_planes.bailout);
        double* output_transition = const_cast<double*>(output_planes.transition);
        double* output_phase = const_cast<double*>(output_planes.phase);
        double* output_de_x = const_cast<double*>(output_planes.de_x);
        double* output_de_y = const_cast<double*>(output_planes.de_y);
        double* output_test1 = const_cast<double*>(output_planes.test1);
        double* output_test2 = const_cast<double*>(output_planes.test2);

#ifdef _OPENMP
        if (threads > 0) {
            omp_set_dynamic(0);
            omp_set_num_threads(threads);
        }
#pragma omp parallel for schedule(static)
#endif
        for (int y = 0; y < output_height; ++y) {
            for (int x = 0; x < output_width; ++x) {
                const size_t index = static_cast<size_t>(y)
                    * static_cast<size_t>(output_width)
                    + static_cast<size_t>(x);
                bool primary_inside = false;
                output_field[index] = primary_scalar(x, y, primary_inside);
                write_primary_int_plane(
                    parent_orbit, output_orbit, x, y, primary_inside);
                write_primary_int_plane(
                    parent_iteration, output_iteration, x, y, primary_inside);
                write_primary_double_plane(
                    parent_bailout, output_bailout, false, false,
                    x, y, primary_inside);
                write_primary_double_plane(
                    parent_transition, output_transition, true, true,
                    x, y, primary_inside);
                write_primary_double_plane(
                    parent_phase, output_phase, false, true,
                    x, y, primary_inside);
                write_primary_double_plane(
                    parent_de_x, output_de_x, false, true,
                    x, y, primary_inside);
                write_primary_double_plane(
                    parent_de_y, output_de_y, false, true,
                    x, y, primary_inside);
                write_primary_double_plane(
                    parent_test1, output_test1, false, true,
                    x, y, primary_inside);
                write_primary_double_plane(
                    parent_test2, output_test2, false, true,
                    x, y, primary_inside);
            }
        }

        if (secondary_field != nullptr) {
            const std::int64_t* child_orbit = secondary_planes->orbit_iteration;
            const std::int64_t* child_iteration = secondary_planes->iteration;
            const double* child_bailout = secondary_planes->bailout;
            const double* child_transition = secondary_planes->transition;
            const double* child_phase = secondary_planes->phase;
            const double* child_de_x = secondary_planes->de_x;
            const double* child_de_y = secondary_planes->de_y;
            const double* child_test1 = secondary_planes->test1;
            const double* child_test2 = secondary_planes->test2;
            const bool parent_has_bailout = parent_bailout != nullptr;
            const bool parent_has_transition = parent_transition != nullptr;
            const bool parent_has_phase = parent_phase != nullptr;
            const bool parent_has_de_x = parent_de_x != nullptr;
            const bool parent_has_de_y = parent_de_y != nullptr;
            const bool parent_has_test1 = parent_test1 != nullptr;
            const bool parent_has_test2 = parent_test2 != nullptr;
            const auto child_alpha = [&](int x, int y) {
                if (seam_feather < 2) return 1.0F;
                const int local_x = x - child_left;
                const int local_y = y - child_top;
                const int edge_distance = std::min(
                    std::min(local_x, visible_child_width - 1 - local_x),
                    std::min(local_y, visible_child_height - 1 - local_y));
                const float linear = std::clamp(
                    static_cast<float>(edge_distance)
                        / static_cast<float>(seam_feather),
                    0.0F,
                    1.0F);
                return linear * linear * (3.0F - 2.0F * linear);
            };
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
            for (int y = child_top; y < child_top + visible_child_height; ++y) {
                for (int x = child_left;
                     x < child_left + visible_child_width; ++x) {
                    const size_t index = static_cast<size_t>(y)
                        * static_cast<size_t>(output_width)
                        + static_cast<size_t>(x);
                    const int child_x = x - child_left;
                    const int child_y = y - child_top;
                    bool child_inside = false;
                    const float child_value = secondary_scalar(
                        child_x, child_y, child_inside);
                    const float alpha = child_alpha(x, y);
                    output_field[index] = output_field[index]
                        * (1.0F - alpha) + child_value * alpha;

                    auto merge_int_plane = [&](
                        const std::int64_t* child_source,
                        std::int64_t* destination
                    ) {
                        if (!child_source || !destination) return;
                        destination[index] = child_inside
                            ? static_cast<std::int64_t>(target_max_iter)
                            : sample_nearest_mapped_int64(
                                child_source, secondary_width,
                                secondary_x_axis, secondary_y_axis,
                                child_x, child_y);
                    };
                    auto merge_double_plane = [&] (
                        const double* child_source,
                        double* destination,
                        bool preserve_negative,
                        bool reset_inside,
                        bool parent_exists
                    ) {
                        if (!child_source || !destination) return;
                        double child_value_sample = sample_bilinear_mapped_double(
                            child_source, secondary_width,
                            secondary_x_axis, secondary_y_axis,
                            child_x, child_y, preserve_negative);
                        if (preserve_negative) {
                            const double nearest = sample_nearest_mapped_double(
                                child_source, secondary_width,
                                secondary_x_axis, secondary_y_axis,
                                child_x, child_y);
                            if (nearest < 0.0) child_value_sample = nearest;
                        }
                        if (child_inside && reset_inside) {
                            child_value_sample = 0.0;
                        }
                        if (!parent_exists) {
                            destination[index] = child_value_sample;
                            return;
                        }
                        const double parent_value = destination[index];
                        const double blended = parent_value * (1.0 - alpha)
                            + child_value_sample * alpha;
                        if (preserve_negative) {
                            const double selected = alpha >= 0.5F
                                ? child_value_sample : parent_value;
                            destination[index] = selected < 0.0
                                ? selected : blended;
                        } else {
                            destination[index] = blended;
                        }
                    };

                    merge_int_plane(child_orbit, output_orbit);
                    merge_int_plane(child_iteration, output_iteration);
                    merge_double_plane(
                        child_bailout, output_bailout, false, false,
                        parent_has_bailout);
                    merge_double_plane(
                        child_transition, output_transition, true, true,
                        parent_has_transition);
                    merge_double_plane(
                        child_phase, output_phase, false, true,
                        parent_has_phase);
                    merge_double_plane(
                        child_de_x, output_de_x, false, true,
                        parent_has_de_x);
                    merge_double_plane(
                        child_de_y, output_de_y, false, true,
                        parent_has_de_y);
                    merge_double_plane(
                        child_test1, output_test1, false, true,
                        parent_has_test1);
                    merge_double_plane(
                        child_test2, output_test2, false, true,
                        parent_has_test2);
                }
            }
        }

        const int colour_max_iter = target_max_iter;
        return fractal_colourise_kfp_impl(
            output_field.data(), output, output_width, output_height,
            colour_max_iter, phase, vocal, instrumental, pitch,
            &transfer_options, &output_planes, lut, lut_size, threads);
    } catch (const std::exception& error) {
        set_error(error.what());
        return 1;
    } catch (...) {
        set_error(
            "native plane-aware KFP atlas colourizer failed with an unknown exception");
        return 1;
    }
}

int fractal_crop_colourise_kfp(
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
    const FractalKfpOptions* options,
    const std::uint8_t* lut,
    int lut_size,
    int threads
) {
    try {
        if (!source || !output
            || !valid_pixel_dimensions(source_width, source_height)
            || !valid_pixel_dimensions(output_width, output_height)
            || !valid_iteration_count(max_iter)
            || !valid_thread_count(threads)
            || !std::isfinite(zoom_factor) || zoom_factor <= 0.0
            || !valid_colour_controls(phase, vocal, instrumental, pitch)
            || !lut || !valid_kfp_options(options, lut_size)) {
            throw std::runtime_error("invalid native KFP crop dimensions or controls");
        }
        zoom_factor = std::max(zoom_factor, 1.0);
        BilinearWorkspace& workspace = bilinear_workspace;
        BilinearAxis& x_axis = workspace.parent_x_axis;
        BilinearAxis& y_axis = workspace.parent_y_axis;
        fill_bilinear_axis(x_axis, source_width, output_width, zoom_factor);
        fill_bilinear_axis(y_axis, source_height, output_height, zoom_factor);

        // KFP's neighbourhood stencil must see a lossless interior marker.
        // Resampling the cap as an ordinary float creates a fake escape band;
        // first build the interior-aware scalar view, then colourise it in the
        // same native pass used by direct KFP frames.
        std::vector<float>& cropped = workspace.kfp_parent_field;
        cropped.resize(static_cast<size_t>(output_width)
            * static_cast<size_t>(output_height));
        const FractalKfpOptions& transfer_options = *options;
        if (transfer_options.field_bias > static_cast<double>(max_iter)) {
            throw std::runtime_error("native KFP field bias exceeds iteration cap");
        }
        const double smooth_offset = kfp_smooth_offset(transfer_options);
#ifdef _OPENMP
        if (threads > 0) {
            omp_set_dynamic(0);
            omp_set_num_threads(threads);
        }
#pragma omp parallel for schedule(static)
#endif
        for (int y = 0; y < output_height; ++y) {
            for (int x = 0; x < output_width; ++x) {
                bool inside = false;
                const float smooth = sample_bilinear_mapped_preserving_interior(
                    source,
                    source_width,
                    x_axis,
                    y_axis,
                    x,
                    y,
                    max_iter,
                    inside,
                    transfer_options.field_bias);
                cropped[static_cast<size_t>(y) * static_cast<size_t>(output_width)
                    + static_cast<size_t>(x)] = inside
                    ? encode_render_iteration(max_iter, transfer_options.field_bias)
                    : encode_render_value(
                        static_cast<long double>(smooth),
                        transfer_options.field_bias);
            }
        }

        const KfpSlopeDirection slope_direction = kfp_slope_direction(transfer_options);
        const KfpPixelContext pixel_context = kfp_pixel_context(
            transfer_options, smooth_offset, nullptr);
        const KfpTransferBounds bounds = transfer_options.color_method == 4
            ? kfp_transfer_bounds(
                cropped.data(),
                output_width,
                output_height,
                max_iter,
                transfer_options.field_bias,
                smooth_offset)
            : KfpTransferBounds{};
        std::vector<double> colour_samples;
        const double* colour_samples_data = nullptr;
        if (pixel_context.needs_difference || pixel_context.needs_slopes) {
            const size_t pixel_count = static_cast<size_t>(output_width)
                * static_cast<size_t>(output_height);
            colour_samples.resize(pixel_count);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
            for (int pixel = 0; pixel < output_width * output_height; ++pixel) {
                colour_samples[static_cast<size_t>(pixel)] =
                    kfp_colour_sample(
                        cropped.data(),
                        output_width,
                        output_height,
                        pixel % output_width,
                        pixel / output_width,
                        max_iter,
                        smooth_offset,
                        transfer_options.field_bias);
            }
            colour_samples_data = colour_samples.data();
        }
        const bool fast_default = false;
        if (fast_default) {
            colourise_kfp_default_fast_field(
                cropped.data(),
                output_width,
                output_height,
                max_iter,
                lut,
                lut_size,
                output,
                threads,
                transfer_options.field_bias);
        } else {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (int y = 0; y < output_height; ++y) {
            for (int x = 0; x < output_width; ++x) {
                write_kfp_pixel(
                    cropped.data(),
                    output_width,
                    output_height,
                    x,
                    y,
                    x,
                    y,
                    output_width,
                    max_iter,
                    transfer_options,
                    pixel_context,
                    lut,
                    lut_size,
                    bounds.minimum,
                    bounds.maximum,
                    slope_direction,
                    output + (static_cast<size_t>(y) * static_cast<size_t>(output_width)
                        + static_cast<size_t>(x)) * 3U,
                    nullptr,
                    nullptr,
                    colour_samples_data);
            }
        }
        }
        set_error("");
        return 0;
    } catch (const std::exception& error) {
        set_error(error.what());
        return 1;
    } catch (...) {
        set_error("native KFP crop colourizer failed with an unknown exception");
        return 1;
    }
}

int fractal_atlas_colourise_kfp_raw(
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
    double child_zoom,
    int max_iter,
    double phase,
    double vocal,
    double instrumental,
    double pitch,
    const FractalKfpOptions* options,
    const std::uint8_t* lut,
    int lut_size,
    int threads
) {
    try {
        if (!parent || !output
            || !valid_pixel_dimensions(parent_width, parent_height)
            || !valid_pixel_dimensions(output_width, output_height)
            || !valid_iteration_count(parent_max_iter)
            || !valid_iteration_count(max_iter)
            || !valid_thread_count(threads)
            || !std::isfinite(parent_zoom) || parent_zoom <= 0.0
            || !std::isfinite(child_fraction)
            || child_fraction < 0.0 || child_fraction > 1.0
            || !std::isfinite(child_zoom) || child_zoom <= 0.0
            || !valid_colour_controls(phase, vocal, instrumental, pitch)
            || !lut || !valid_kfp_options(options, lut_size)) {
            throw std::runtime_error("invalid native raw KFP atlas dimensions or controls");
        }
        parent_zoom = std::max(parent_zoom, 1.0);
        child_zoom = std::max(child_zoom, 1.0);
        const bool use_child = child != nullptr && child_fraction > 0.0;
        if (use_child && (!valid_pixel_dimensions(child_width, child_height)
                          || !valid_iteration_count(child_max_iter))) {
            throw std::runtime_error("invalid native raw KFP atlas child tile");
        }
        if (!use_child && child != nullptr) {
            throw std::runtime_error("raw KFP atlas child needs a positive visible fraction");
        }
        if (!use_child && (child_width != 0 || child_height != 0 || child_max_iter != 0)) {
            throw std::runtime_error("invalid native raw KFP atlas child absence");
        }
        const FractalKfpOptions& transfer_options = *options;
        if (transfer_options.field_bias > static_cast<double>(max_iter)) {
            throw std::runtime_error("native KFP field bias exceeds iteration cap");
        }
        // A non-zero output bias denotes the centered atlas encoding used by
        // the production renderer. Each source tile was encoded relative to
        // its own cap, so decode parent/child samples with their individual
        // caps before converting both into the target frame representation.
        const bool centered_input = transfer_options.field_bias > 0.0;
        const double parent_field_bias = centered_input
            ? static_cast<double>(parent_max_iter) : 0.0;
        const double child_field_bias = centered_input
            ? static_cast<double>(child_max_iter) : 0.0;

        const int visible_child_width = use_child
            ? std::max(1, static_cast<int>(std::lround(
                static_cast<double>(output_width) * child_fraction)))
            : 0;
        const int visible_child_height = use_child
            ? std::max(1, static_cast<int>(std::lround(
                static_cast<double>(output_height) * child_fraction)))
            : 0;
        const int visible_child_left = use_child
            ? (output_width - visible_child_width) / 2
            : 0;
        const int visible_child_top = use_child
            ? (output_height - visible_child_height) / 2
            : 0;
        const int halo = use_child
            ? std::min(
                2,
                std::min(
                    std::min(visible_child_left, output_width
                        - visible_child_left - visible_child_width),
                    std::min(visible_child_top, output_height
                        - visible_child_top - visible_child_height)))
            : 0;
        const int child_left = use_child ? visible_child_left - halo : 0;
        const int child_top = use_child ? visible_child_top - halo : 0;
        const int sampled_child_width = use_child
            ? visible_child_width + 2 * halo
            : 0;
        const int sampled_child_height = use_child
            ? visible_child_height + 2 * halo
            : 0;
        // The halo is sampling support, not visible atlas content. Only the
        // nominal child rectangle may replace the parent. Keep a tiny scalar
        // transition inside that rectangle: it hides the unavoidable
        // screen-space stencil seam without copying a wide halo or creating
        // the rectangular interior fill reported for Burning Ship and
        // Tricorn.
        const int ownership_left = visible_child_left;
        const int ownership_top = visible_child_top;
        const int seam_feather = use_child
            ? kfp_atlas_seam_feather(visible_child_width, visible_child_height)
            : 0;
        BilinearWorkspace& workspace = bilinear_workspace;
        BilinearAxis& parent_x_axis = workspace.parent_x_axis;
        BilinearAxis& parent_y_axis = workspace.parent_y_axis;
        fill_bilinear_axis(parent_x_axis, parent_width, output_width, parent_zoom);
        fill_bilinear_axis(parent_y_axis, parent_height, output_height, parent_zoom);
        std::vector<float>& parent_view = workspace.kfp_parent_field;
        parent_view.resize(static_cast<size_t>(output_width)
            * static_cast<size_t>(output_height));

#ifdef _OPENMP
        if (threads > 0) {
            omp_set_dynamic(0);
            omp_set_num_threads(threads);
        }
#pragma omp parallel for schedule(static)
#endif
        for (int y = 0; y < output_height; ++y) {
            for (int x = 0; x < output_width; ++x) {
                bool inside = false;
                const float smooth = sample_bilinear_mapped_preserving_interior(
                    parent,
                    parent_width,
                    parent_x_axis,
                    parent_y_axis,
                    x,
                    y,
                    parent_max_iter,
                    inside,
                    parent_field_bias);
                parent_view[static_cast<size_t>(y) * static_cast<size_t>(output_width)
                    + static_cast<size_t>(x)] = inside
                    ? encode_render_iteration(max_iter, transfer_options.field_bias)
                    : encode_render_value(
                        static_cast<long double>(smooth),
                        transfer_options.field_bias);
            }
        }

        std::vector<float>& child_view = workspace.kfp_child_field;
        if (use_child) {
            BilinearAxis& child_x_axis = workspace.child_x_axis;
            BilinearAxis& child_y_axis = workspace.child_y_axis;
            fill_bilinear_axis_window(
                child_x_axis,
                child_width,
                sampled_child_width,
                -static_cast<double>(halo),
                static_cast<double>(visible_child_width),
                child_zoom);
            fill_bilinear_axis_window(
                child_y_axis,
                child_height,
                sampled_child_height,
                -static_cast<double>(halo),
                static_cast<double>(visible_child_height),
                child_zoom);
            child_view.resize(static_cast<size_t>(sampled_child_width)
                * static_cast<size_t>(sampled_child_height));
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
            for (int y = 0; y < sampled_child_height; ++y) {
                for (int x = 0; x < sampled_child_width; ++x) {
                    bool inside = false;
                    const float smooth = sample_bilinear_mapped_preserving_interior(
                        child,
                        child_width,
                        child_x_axis,
                        child_y_axis,
                        x,
                        y,
                        child_max_iter,
                        inside,
                        child_field_bias);
                    child_view[static_cast<size_t>(y)
                        * static_cast<size_t>(sampled_child_width)
                        + static_cast<size_t>(x)] = inside
                        ? encode_render_iteration(max_iter, transfer_options.field_bias)
                        : encode_render_value(
                            static_cast<long double>(smooth),
                            transfer_options.field_bias);
                }
            }
        } else {
            child_view.clear();
        }

        if (use_child) {
            // KFP's differences and slope passes are screen-space
            // operations. Colourising the parent and child independently
            // gives each tile a different finite-difference scale, so the
            // child can become a large, perfectly rectangular colour patch
            // when it is inserted into a zoomed parent. Reproject both
            // tiles into one scalar surface first and run one KFP pass over
            // that surface. This keeps the neighbourhood stencil continuous
            // across the atlas transition and also avoids a second full RGB
            // colourisation.
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
            for (int y = ownership_top;
                 y < ownership_top + visible_child_height;
                 ++y) {
                for (int x = ownership_left;
                     x < ownership_left + visible_child_width;
                     ++x) {
                    const int child_x = x - child_left;
                    const int child_y = y - child_top;
                    const size_t output_index = static_cast<size_t>(y)
                        * static_cast<size_t>(output_width)
                        + static_cast<size_t>(x);
                    const float child_value = child_view[
                        static_cast<size_t>(child_y)
                        * static_cast<size_t>(sampled_child_width)
                        + static_cast<size_t>(child_x)];
                    // The child owns the visible region. Its source halo is
                    // used only to make edge samples well-defined; the
                    // resolution-scaled scalar blend keeps an interior
                    // sentinel from becoming a hard rectangular RGB patch.
                    float alpha = 1.0F;
                    if (seam_feather >= 2) {
                        const int visible_x = x - ownership_left;
                        const int visible_y = y - ownership_top;
                        const int edge_distance = std::min(
                            std::min(visible_x, visible_child_width - 1 - visible_x),
                            std::min(visible_y, visible_child_height - 1 - visible_y));
                        const float linear = std::clamp(
                            static_cast<float>(edge_distance)
                                / static_cast<float>(seam_feather),
                            0.0F,
                            1.0F);
                        alpha = linear * linear * (3.0F - 2.0F * linear);
                    }
                    if (alpha >= 0.999999F) {
                        parent_view[output_index] = child_value;
                    } else {
                        const float parent_value = parent_view[output_index];
                        parent_view[output_index] = parent_value * (1.0F - alpha)
                            + child_value * alpha;
                    }
                }
            }
        }

        // The complete frame now has one coordinate system and one
        // neighbourhood stencil. In particular, do not call the atlas RGB
        // compositor here: it would recreate the independent-tile seam this
        // raw path exists to eliminate.
        return fractal_colourise_kfp(
            parent_view.data(),
            output,
            output_width,
            output_height,
            max_iter,
            phase,
            vocal,
            instrumental,
            pitch,
            &transfer_options,
            lut,
            lut_size,
            threads);
    } catch (const std::exception& error) {
        set_error(error.what());
        return 1;
    } catch (...) {
        set_error("native raw KFP atlas colourizer failed with an unknown exception");
        return 1;
    }
}

// Additive compatibility wrappers for callers that still request the precise
// symbol. The production entry point currently uses the same source-faithful
// native transfer path; retaining this symbol avoids an ABI break.
int fractal_colourise_kfp_precise(
    const float* field,
    std::uint8_t* output,
    int width,
    int height,
    int max_iter,
    double phase,
    double vocal,
    double instrumental,
    double pitch,
    const FractalKfpOptions* options,
    const std::uint8_t* lut,
    int lut_size,
    int threads
) {
    KfpPreciseGuard guard;
    return fractal_colourise_kfp(
        field, output, width, height, max_iter, phase, vocal,
        instrumental, pitch, options, lut, lut_size, threads);
}

int fractal_crop_colourise_kfp_precise(
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
    const FractalKfpOptions* options,
    const std::uint8_t* lut,
    int lut_size,
    int threads
) {
    KfpPreciseGuard guard;
    return fractal_crop_colourise_kfp(
        source, source_width, source_height, output, output_width,
        output_height, zoom_factor, max_iter, phase, vocal,
        instrumental, pitch, options, lut, lut_size, threads);
}

int fractal_atlas_colourise_kfp_precise(
    const float* parent,
    int parent_width,
    int parent_height,
    const float* child,
    int child_width,
    int child_height,
    std::uint8_t* output,
    int output_width,
    int output_height,
    int max_iter,
    int child_left,
    int child_top,
    int feather,
    double phase,
    double vocal,
    double instrumental,
    double pitch,
    const FractalKfpOptions* options,
    const std::uint8_t* lut,
    int lut_size,
    int threads
) {
    KfpPreciseGuard guard;
    return fractal_atlas_colourise_kfp(
        parent, parent_width, parent_height, child, child_width,
        child_height, output, output_width, output_height, max_iter,
        child_left, child_top, feather, phase, vocal, instrumental,
        pitch, options, lut, lut_size, threads);
}

int fractal_atlas_colourise_kfp_planes_precise(
    const float* parent,
    int parent_width,
    int parent_height,
    int parent_max_iter,
    const FractalKfpPlanes* parent_planes,
    const float* child,
    int child_width,
    int child_height,
    int child_max_iter,
    const FractalKfpPlanes* child_planes,
    std::uint8_t* output,
    int output_width,
    int output_height,
    double parent_zoom,
    double child_fraction,
    double child_zoom,
    int max_iter,
    int centered_input,
    double phase,
    double vocal,
    double instrumental,
    double pitch,
    const FractalKfpOptions* options,
    const std::uint8_t* lut,
    int lut_size,
    int threads
) {
    KfpPreciseGuard guard;
    return fractal_atlas_colourise_kfp_planes(
        parent, parent_width, parent_height, parent_max_iter, parent_planes,
        child, child_width, child_height, child_max_iter, child_planes,
        output, output_width, output_height, parent_zoom, child_fraction,
        child_zoom, max_iter, centered_input, phase, vocal, instrumental,
        pitch, options, lut, lut_size, threads);
}

int fractal_atlas_colourise_kfp_raw_precise(
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
    double child_zoom,
    int max_iter,
    double phase,
    double vocal,
    double instrumental,
    double pitch,
    const FractalKfpOptions* options,
    const std::uint8_t* lut,
    int lut_size,
    int threads
) {
    KfpPreciseGuard guard;
    return fractal_atlas_colourise_kfp_raw(
        parent, parent_width, parent_height, parent_max_iter, child,
        child_width, child_height, child_max_iter, output, output_width,
        output_height, parent_zoom, child_fraction, child_zoom, max_iter, phase,
        vocal, instrumental, pitch, options, lut, lut_size, threads);
}

int fractal_crop_rgb(
    const std::uint8_t* source,
    int source_width,
    int source_height,
    std::uint8_t* output,
    int output_width,
    int output_height,
    double zoom_factor,
    int threads
) {
    try {
        if (!source || !output
            || !valid_pixel_dimensions(source_width, source_height)
            || !valid_pixel_dimensions(output_width, output_height)
            || !valid_thread_count(threads)
            || !std::isfinite(zoom_factor) || zoom_factor <= 0.0) {
            throw std::runtime_error("invalid native RGB crop dimensions");
        }
        zoom_factor = std::max(zoom_factor, 1.0);
        if (source_width == output_width
            && source_height == output_height
            && zoom_factor == 1.0) {
            std::memcpy(
                output,
                source,
                static_cast<size_t>(output_width)
                    * static_cast<size_t>(output_height) * 3U);
            set_error("");
            return 0;
        }
        BilinearWorkspace& workspace = bilinear_workspace;
        BilinearAxis& x_axis = workspace.parent_x_axis;
        BilinearAxis& y_axis = workspace.parent_y_axis;
        fill_bilinear_axis(x_axis, source_width, output_width, zoom_factor);
        fill_bilinear_axis(y_axis, source_height, output_height, zoom_factor);
#ifdef _OPENMP
        if (threads > 0) {
            omp_set_dynamic(0);
            omp_set_num_threads(threads);
        }
#pragma omp parallel for schedule(static)
#endif
        for (int output_y = 0; output_y < output_height; ++output_y) {
            for (int output_x = 0; output_x < output_width; ++output_x) {
                sample_rgb_bilinear_mapped(
                    source,
                    source_width,
                    x_axis,
                    y_axis,
                    output_x,
                    output_y,
                    output + (
                        static_cast<size_t>(output_y)
                        * static_cast<size_t>(output_width)
                        + static_cast<size_t>(output_x)) * 3U);
            }
        }
        set_error("");
        return 0;
    } catch (const std::exception& error) {
        set_error(error.what());
        return 1;
    } catch (...) {
        set_error("native RGB crop failed with an unknown exception");
        return 1;
    }
}

int fractal_atlas_composite_rgb(
    const std::uint8_t* parent,
    int parent_width,
    int parent_height,
    const std::uint8_t* child,
    int child_width,
    int child_height,
    std::uint8_t* output,
    int output_width,
    int output_height,
    double parent_zoom,
    double child_fraction,
    double child_zoom,
    int feather,
    int threads
) {
    try {
        if (!parent || !output
            || !valid_pixel_dimensions(parent_width, parent_height)
            || !valid_pixel_dimensions(output_width, output_height)
            || !valid_thread_count(threads)
            || !std::isfinite(parent_zoom) || parent_zoom <= 0.0
            || !std::isfinite(child_fraction)
            || child_fraction < 0.0 || child_fraction > 1.0
            || !std::isfinite(child_zoom) || child_zoom <= 0.0
            || feather < 0) {
            throw std::runtime_error("invalid native RGB atlas dimensions or controls");
        }
        parent_zoom = std::max(parent_zoom, 1.0);
        child_zoom = std::max(child_zoom, 1.0);
        const bool use_child = child != nullptr && child_fraction > 0.0;
        if (!use_child && child_fraction > 0.0) {
            throw std::runtime_error("RGB atlas child data is missing for a positive fraction");
        }
        if (!use_child && child != nullptr) {
            throw std::runtime_error("RGB atlas child needs a positive visible fraction");
        }
        if (!use_child && (child_width != 0 || child_height != 0)) {
            throw std::runtime_error("invalid native RGB atlas child absence");
        }
        if (use_child
            && (!valid_pixel_dimensions(child_width, child_height)
                || feather > std::min(
                    output_width,
                    output_height))) {
            throw std::runtime_error("invalid native RGB atlas child tile");
        }

        const int visible_child_width = use_child
            ? std::max(1, static_cast<int>(std::lround(
                static_cast<double>(output_width) * child_fraction)))
            : 0;
        const int visible_child_height = use_child
            ? std::max(1, static_cast<int>(std::lround(
                static_cast<double>(output_height) * child_fraction)))
            : 0;
        const int child_left = use_child
            ? (output_width - visible_child_width) / 2
            : 0;
        const int child_top = use_child
            ? (output_height - visible_child_height) / 2
            : 0;
        const int effective_feather = use_child
            ? std::min(
                feather,
                std::min(visible_child_width, visible_child_height))
            : 0;

        BilinearWorkspace& workspace = bilinear_workspace;
        BilinearAxis& parent_x_axis = workspace.parent_x_axis;
        BilinearAxis& parent_y_axis = workspace.parent_y_axis;
        fill_bilinear_axis(parent_x_axis, parent_width, output_width, parent_zoom);
        fill_bilinear_axis(parent_y_axis, parent_height, output_height, parent_zoom);
        BilinearAxis& child_x_axis = workspace.child_x_axis;
        BilinearAxis& child_y_axis = workspace.child_y_axis;
        if (use_child) {
            // A child tile is deliberately stored with the same fixed
            // overscan as its parent. Apply that crop during partial atlas
            // transitions too; otherwise the child is stretched across a
            // rectangle that is 1.2x wider than its actual camera viewport.
            fill_bilinear_axis(
                child_x_axis,
                child_width,
                visible_child_width,
                child_zoom);
            fill_bilinear_axis(
                child_y_axis,
                child_height,
                visible_child_height,
                child_zoom);
        }

#ifdef _OPENMP
        if (threads > 0) {
            omp_set_dynamic(0);
            omp_set_num_threads(threads);
        }
#pragma omp parallel for schedule(static)
#endif
        for (int output_y = 0; output_y < output_height; ++output_y) {
            for (int output_x = 0; output_x < output_width; ++output_x) {
                std::uint8_t* destination = output + (
                    static_cast<size_t>(output_y)
                    * static_cast<size_t>(output_width)
                    + static_cast<size_t>(output_x)) * 3U;
                sample_rgb_bilinear_mapped(
                    parent,
                    parent_width,
                    parent_x_axis,
                    parent_y_axis,
                    output_x,
                    output_y,
                    destination);
                if (!use_child
                    || output_x < child_left
                    || output_x >= child_left + visible_child_width
                    || output_y < child_top
                    || output_y >= child_top + visible_child_height) {
                    continue;
                }
                std::uint8_t child_rgb[3];
                sample_rgb_bilinear_mapped(
                    child,
                    child_width,
                    child_x_axis,
                    child_y_axis,
                    output_x - child_left,
                    output_y - child_top,
                    child_rgb);
                double alpha = 1.0;
                if (effective_feather >= 2) {
                    const int edge_distance = std::min(
                        std::min(output_x - child_left,
                            visible_child_width - 1 - (output_x - child_left)),
                        std::min(output_y - child_top,
                            visible_child_height - 1 - (output_y - child_top)));
                    const double linear = std::clamp(
                        static_cast<double>(edge_distance)
                            / static_cast<double>(effective_feather),
                        0.0,
                        1.0);
                    alpha = linear * linear * (3.0 - 2.0 * linear);
                }
                for (int channel = 0; channel < 3; ++channel) {
                    destination[channel] = rounded_colour_byte(
                        static_cast<double>(destination[channel]) * (1.0 - alpha)
                        + static_cast<double>(child_rgb[channel]) * alpha);
                }
            }
        }
        set_error("");
        return 0;
    } catch (const std::exception& error) {
        set_error(error.what());
        return 1;
    } catch (...) {
        set_error("native RGB atlas compositor failed with an unknown exception");
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

/* Reproject a parent/child scalar atlas in one native pass.  The Python
 * compositor used to do this with two Pillow resizes plus a NumPy feather for
 * every video frame before handing the result to the OpenCL palette lookup.
 * Keep the same interior-aware sampling rules as the native RGB compositor,
 * but expose the scalar surface so the device can colour it without another
 * host-side crop/composite round trip. */
int fractal_atlas_field_ex(
    const float* parent,
    int parent_width,
    int parent_height,
    int parent_max_iter,
    const float* child,
    int child_width,
    int child_height,
    int child_max_iter,
    float* output,
    int output_width,
    int output_height,
    double parent_zoom,
    double child_fraction,
    double child_zoom,
    double parent_field_bias,
    double child_field_bias,
    double output_field_bias,
    int palette_max_iter,
    int feather,
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
            || !std::isfinite(child_zoom) || child_zoom <= 0.0
            || !std::isfinite(parent_field_bias)
            || parent_field_bias < 0.0
            || parent_field_bias > static_cast<double>(parent_max_iter)
            || !std::isfinite(child_field_bias) || child_field_bias < 0.0
            || !std::isfinite(output_field_bias)
            || output_field_bias < 0.0
            || feather < 0
            || !valid_iteration_count(palette_max_iter)) {
            throw std::runtime_error("invalid native scalar atlas dimensions");
        }
        const bool use_child = child != nullptr && child_fraction > 0.0;
        if (use_child && (!valid_pixel_dimensions(child_width, child_height)
                          || !valid_iteration_count(child_max_iter)
                          || child_field_bias > static_cast<double>(child_max_iter))) {
            throw std::runtime_error("invalid native scalar atlas child tile");
        }
        const int effective_iter = std::max(
            palette_max_iter,
            std::max(parent_max_iter, use_child ? child_max_iter : 0));
        if (!valid_iteration_count(effective_iter)
            || output_field_bias > static_cast<double>(effective_iter)) {
            throw std::runtime_error("invalid native scalar atlas iteration cap");
        }
        const bool full_child = use_child && child_fraction >= 0.999999;
        const int visible_child_width = use_child
            ? std::max(1, static_cast<int>(std::lround(
                static_cast<double>(output_width) * child_fraction)))
            : 0;
        const int visible_child_height = use_child
            ? std::max(1, static_cast<int>(std::lround(
                static_cast<double>(output_height) * child_fraction)))
            : 0;
        const int child_left = use_child
            ? (output_width - visible_child_width) / 2
            : 0;
        const int child_top = use_child
            ? (output_height - visible_child_height) / 2
            : 0;
        const int seam_feather = use_child && !full_child
            ? std::min(feather, std::min(visible_child_width / 2,
                                          visible_child_height / 2))
            : 0;

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
            fill_bilinear_axis(
                child_x_axis,
                child_width,
                full_child ? output_width : visible_child_width,
                std::max(child_zoom, 1.0));
            fill_bilinear_axis(
                child_y_axis,
                child_height,
                full_child ? output_height : visible_child_height,
                std::max(child_zoom, 1.0));
        }
        std::vector<float>& child_edge_x = workspace.child_edge_x;
        std::vector<float>& child_edge_y = workspace.child_edge_y;
        if (use_child && !full_child && seam_feather >= 2) {
            child_edge_x.resize(static_cast<size_t>(visible_child_width));
            child_edge_y.resize(static_cast<size_t>(visible_child_height));
            for (int x = 0; x < visible_child_width; ++x) {
                const int edge = std::min(x, visible_child_width - 1 - x);
                const float linear = std::min(
                    1.0F, static_cast<float>(edge) / static_cast<float>(seam_feather));
                child_edge_x[static_cast<size_t>(x)] = linear * linear
                    * (3.0F - 2.0F * linear);
            }
            for (int y = 0; y < visible_child_height; ++y) {
                const int edge = std::min(y, visible_child_height - 1 - y);
                const float linear = std::min(
                    1.0F, static_cast<float>(edge) / static_cast<float>(seam_feather));
                child_edge_y[static_cast<size_t>(y)] = linear * linear
                    * (3.0F - 2.0F * linear);
            }
        }

#ifdef _OPENMP
        if (threads > 0) omp_set_num_threads(threads);
#pragma omp parallel for schedule(static)
#endif
        for (int output_y = 0; output_y < output_height; ++output_y) {
            for (int output_x = 0; output_x < output_width; ++output_x) {
                float value = 0.0F;
                if (full_child) {
                    bool child_inside = false;
                    const float child_smooth =
                        sample_bilinear_mapped_preserving_interior(
                            child, child_width, child_x_axis, child_y_axis,
                            output_x,
                            output_y,
                            child_max_iter,
                            child_inside,
                            child_field_bias);
                    value = child_inside
                        ? static_cast<float>(effective_iter - output_field_bias)
                        : child_smooth - static_cast<float>(output_field_bias);
                } else {
                    bool parent_inside = false;
                    const float parent_smooth =
                        sample_bilinear_mapped_preserving_interior(
                            parent, parent_width, parent_x_axis, parent_y_axis,
                            output_x,
                            output_y,
                            parent_max_iter,
                            parent_inside,
                            parent_field_bias);
                    value = parent_inside
                        ? static_cast<float>(effective_iter - output_field_bias)
                        : parent_smooth - static_cast<float>(output_field_bias);
                    if (use_child
                        && output_x >= child_left
                        && output_x < child_left + visible_child_width
                        && output_y >= child_top
                        && output_y < child_top + visible_child_height) {
                        const int child_x = output_x - child_left;
                        const int child_y = output_y - child_top;
                        bool child_inside = false;
                        const float child_smooth =
                            sample_bilinear_mapped_preserving_interior(
                                child, child_width, child_x_axis, child_y_axis,
                                child_x,
                                child_y,
                                child_max_iter,
                                child_inside,
                                child_field_bias);
                        const float child_value = child_inside
                            ? static_cast<float>(effective_iter - output_field_bias)
                            : child_smooth - static_cast<float>(output_field_bias);
                        float alpha = 1.0F;
                        if (seam_feather >= 2) {
                            alpha = std::min(
                                child_edge_x[static_cast<size_t>(child_x)],
                                child_edge_y[static_cast<size_t>(child_y)]);
                        }
                        value = value * (1.0F - alpha) + child_value * alpha;
                    }
                }
                output[static_cast<size_t>(output_y) * output_width + output_x] = value;
            }
        }
        set_error("");
        return 0;
    } catch (const std::exception& error) {
        set_error(error.what());
        return 1;
    } catch (...) {
        set_error("native scalar atlas compositor failed with an unknown exception");
        return 1;
    }
}

/* Keep the original ABI available for older callers. */
int fractal_atlas_field(
    const float* parent,
    int parent_width,
    int parent_height,
    int parent_max_iter,
    const float* child,
    int child_width,
    int child_height,
    int child_max_iter,
    float* output,
    int output_width,
    int output_height,
    double parent_zoom,
    double child_fraction,
    int palette_max_iter,
    int threads
) {
    return fractal_atlas_field_ex(
        parent,
        parent_width,
        parent_height,
        parent_max_iter,
        child,
        child_width,
        child_height,
        child_max_iter,
        output,
        output_width,
        output_height,
        parent_zoom,
        child_fraction,
        1.0,
        0.0,
        0.0,
        0.0,
        palette_max_iter,
        48,
        threads);
}

int crop_colourise_impl(
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
    int threads,
    const int* interior_color,
    const std::uint8_t* accents
) {
    try {
        if (!source || !output || !valid_pixel_dimensions(source_width, source_height)
            || !valid_pixel_dimensions(output_width, output_height)
            || !valid_iteration_count(max_iter) || !valid_thread_count(threads)
            || !std::isfinite(zoom_factor) || zoom_factor <= 0.0
            || !valid_colour_controls(phase, vocal, instrumental, pitch)
            || (interior_color != nullptr
                && (interior_color[0] < 0 || interior_color[0] > 255
                    || interior_color[1] < 0 || interior_color[1] > 255
                    || interior_color[2] < 0 || interior_color[2] > 255))) {
            throw std::runtime_error("invalid native crop/colour dimensions or palette");
        }
        zoom_factor = std::max(zoom_factor, 1.0);
        const AuroraPalette& palette = aurora_palette_for(
            max_iter, phase, vocal, instrumental, pitch, accents);
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
                    rgb[0] = interior_color
                        ? static_cast<std::uint8_t>(interior_color[0]) : 0;
                    rgb[1] = interior_color
                        ? static_cast<std::uint8_t>(interior_color[1]) : 0;
                    rgb[2] = interior_color
                        ? static_cast<std::uint8_t>(interior_color[2]) : 0;
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
    return crop_colourise_impl(
        source,
        source_width,
        source_height,
        output,
        output_width,
        output_height,
        zoom_factor,
        max_iter,
        phase,
        vocal,
        instrumental,
        pitch,
        threads,
        nullptr,
        nullptr);
}

int fractal_crop_colourise_interior(
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
    int interior_red,
    int interior_green,
    int interior_blue,
    int threads
) {
    const int interior_color[3] = {
        interior_red,
        interior_green,
        interior_blue,
    };
    return crop_colourise_impl(
        source,
        source_width,
        source_height,
        output,
        output_width,
        output_height,
        zoom_factor,
        max_iter,
        phase,
        vocal,
        instrumental,
        pitch,
        threads,
        interior_color,
        nullptr);
}

int fractal_crop_colourise_accents(
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
    const std::uint8_t* accents,
    int interior_red,
    int interior_green,
    int interior_blue,
    int threads
) {
    const int interior_color[3] = {
        interior_red,
        interior_green,
        interior_blue,
    };
    return crop_colourise_impl(
        source,
        source_width,
        source_height,
        output,
        output_width,
        output_height,
        zoom_factor,
        max_iter,
        phase,
        vocal,
        instrumental,
        pitch,
        threads,
        &interior_color[0],
        accents);
}

int atlas_colourise_impl(
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
    int threads,
    const int* interior_color,
    const std::uint8_t* accents
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
            || !valid_colour_controls(phase, vocal, instrumental, pitch)
            || (interior_color != nullptr
                && (interior_color[0] < 0 || interior_color[0] > 255
                    || interior_color[1] < 0 || interior_color[1] > 255
                    || interior_color[2] < 0 || interior_color[2] > 255))) {
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
            effective_palette_iter, phase, vocal, instrumental, pitch, accents);
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
        // A narrow linear strip still reads as a rectangle at 1080p. Keep the
        // native ordinary compositor in lockstep with the portable/KFP path;
        // the cap prevents a small live preview from becoming over-soft.
        const int feather = use_child
            ? std::min(48, std::min(visible_child_width / 8, visible_child_height / 8))
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
                const float linear = std::min(
                    1.0F, static_cast<float>(edge) / static_cast<float>(feather));
                child_edge_x[static_cast<size_t>(x)] = linear * linear
                    * (3.0F - 2.0F * linear);
            }
            for (int y = 0; y < visible_child_height; ++y) {
                const int edge = std::min(y, visible_child_height - 1 - y);
                const float linear = std::min(
                    1.0F, static_cast<float>(edge) / static_cast<float>(feather));
                child_edge_y[static_cast<size_t>(y)] = linear * linear
                    * (3.0F - 2.0F * linear);
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
                    destination,
                    interior_color);
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
                        destination,
                        interior_color);
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
                        destination,
                        interior_color);
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
                // The feather band is a real compositing operation, including
                // when either source classifies its sample as interior. A
                // hard interior shortcut here makes a valid black fill inherit
                // the child rectangle's four edges. Blend already-colourised
                // samples only in this narrow band; outside it the deeper tile
                // remains authoritative and the hot path still performs one
                // scalar colour lookup.
                std::uint8_t parent_rgb[3];
                std::uint8_t child_rgb[3];
                write_colour_pixel(
                    parent_inside
                        ? static_cast<float>(effective_palette_iter)
                        : parent_smooth,
                    effective_palette_iter,
                    palette,
                    palette_index_scale,
                    parent_rgb,
                    interior_color);
                write_colour_pixel(
                    child_inside
                        ? static_cast<float>(effective_palette_iter)
                        : child_smooth,
                    effective_palette_iter,
                    palette,
                    palette_index_scale,
                    child_rgb,
                    interior_color);
                destination[0] = rounded_colour_byte(
                    static_cast<double>(parent_rgb[0]) * (1.0 - alpha)
                    + static_cast<double>(child_rgb[0]) * alpha);
                destination[1] = rounded_colour_byte(
                    static_cast<double>(parent_rgb[1]) * (1.0 - alpha)
                    + static_cast<double>(child_rgb[1]) * alpha);
                destination[2] = rounded_colour_byte(
                    static_cast<double>(parent_rgb[2]) * (1.0 - alpha)
                    + static_cast<double>(child_rgb[2]) * alpha);
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
    return atlas_colourise_impl(
        parent,
        parent_width,
        parent_height,
        parent_max_iter,
        child,
        child_width,
        child_height,
        child_max_iter,
        output,
        output_width,
        output_height,
        parent_zoom,
        child_fraction,
        palette_max_iter,
        phase,
        vocal,
        instrumental,
        pitch,
        threads,
        nullptr,
        nullptr);
}

int fractal_atlas_colourise_interior(
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
    int interior_red,
    int interior_green,
    int interior_blue,
    int threads
) {
    const int interior_color[3] = {
        interior_red,
        interior_green,
        interior_blue,
    };
    return atlas_colourise_impl(
        parent,
        parent_width,
        parent_height,
        parent_max_iter,
        child,
        child_width,
        child_height,
        child_max_iter,
        output,
        output_width,
        output_height,
        parent_zoom,
        child_fraction,
        palette_max_iter,
        phase,
        vocal,
        instrumental,
        pitch,
        threads,
        interior_color,
        nullptr);
}

int fractal_atlas_colourise_accents(
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
    const std::uint8_t* accents,
    int interior_red,
    int interior_green,
    int interior_blue,
    int threads
) {
    if (!accents) {
        set_error("ordinary palette accents are required");
        return 1;
    }
    const int interior_color[3] = {
        interior_red,
        interior_green,
        interior_blue,
    };
    return atlas_colourise_impl(
        parent,
        parent_width,
        parent_height,
        parent_max_iter,
        child,
        child_width,
        child_height,
        child_max_iter,
        output,
        output_width,
        output_height,
        parent_zoom,
        child_fraction,
        palette_max_iter,
        phase,
        vocal,
        instrumental,
        pitch,
        threads,
        &interior_color[0],
        accents);
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

void* fractal_create_reference_reusable_options(
    const char* x_center,
    const char* y_center,
    const char* viewport_zoom,
    int max_iter,
    int precision_bits,
    int series_order,
    const FractalRenderOptions* supplied_options
) {
    try {
        if (!x_center || !y_center || !viewport_zoom
            || !valid_iteration_count(max_iter)
            || !valid_precision_bits(precision_bits)
            || !valid_series_parameters(series_order, 2)) {
            throw std::runtime_error("invalid reusable reference configuration");
        }
        const FractalRenderOptions options = checked_render_options(supplied_options);
        auto context = create_reference_context(
            x_center, y_center, viewport_zoom, max_iter, precision_bits,
            series_order, true, FRACTAL_FORMULA_MANDELBROT,
            "0", "0", options.escape_radius_mode, options.coordinate_mode);
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

void* fractal_create_reference_ex(
    const char* x_center,
    const char* y_center,
    const char* viewport_zoom,
    int max_iter,
    int precision_bits,
    int series_order,
    int formula,
    const char* julia_real,
    const char* julia_imag
) {
    try {
        if (!x_center || !y_center || !viewport_zoom
            || !julia_real || !julia_imag
            || !valid_iteration_count(max_iter)
            || !valid_precision_bits(precision_bits)
            || !valid_series_parameters(series_order, 2)
            || !valid_formula(formula)) {
            throw std::runtime_error("invalid formula-aware reference configuration");
        }
        auto context = create_reference_context(
            x_center, y_center, viewport_zoom, max_iter, precision_bits,
            series_order, formula != FRACTAL_FORMULA_MANDELBROT,
            formula, julia_real, julia_imag);
        set_error("");
        return register_reference(std::move(context));
    } catch (const std::exception& error) {
        set_error(error.what());
        return nullptr;
    } catch (...) {
        set_error("formula-aware reference creation failed with an unknown exception");
        return nullptr;
    }
}

void* fractal_create_reference_ex_options(
    const char* x_center,
    const char* y_center,
    const char* viewport_zoom,
    int max_iter,
    int precision_bits,
    int series_order,
    int formula,
    const char* julia_real,
    const char* julia_imag,
    const FractalRenderOptions* supplied_options
) {
    try {
        if (!x_center || !y_center || !viewport_zoom
            || !julia_real || !julia_imag
            || !valid_iteration_count(max_iter)
            || !valid_precision_bits(precision_bits)
            || !valid_series_parameters(series_order, 2)
            || !valid_formula(formula)) {
            throw std::runtime_error("invalid formula-aware reference configuration");
        }
        const FractalRenderOptions options = checked_render_options(supplied_options);
        auto context = create_reference_context(
            x_center, y_center, viewport_zoom, max_iter, precision_bits,
            series_order, formula != FRACTAL_FORMULA_MANDELBROT,
            formula,
            julia_real,
            julia_imag,
            options.escape_radius_mode,
            options.coordinate_mode);
        set_error("");
        return register_reference(std::move(context));
    } catch (const std::exception& error) {
        set_error(error.what());
        return nullptr;
    } catch (...) {
        set_error("formula-aware reference creation failed with an unknown exception");
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

int fractal_render_reference_ex_planes(
    float* output,
    int width,
    int height,
    const char* zoom_text,
    void* handle,
    int max_iter,
    int threads,
    int series_order,
    int series_block,
    const FractalRenderOptions* supplied_options,
    FractalRenderPlanes* planes
) {
    try {
        if (!output || !zoom_text || !handle || !valid_pixel_dimensions(width, height)
            || !valid_iteration_count(max_iter) || !valid_thread_count(threads)
            || !valid_series_parameters(series_order, series_block)
            || (planes != nullptr && !valid_render_planes(planes, width, height))) {
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
        if (planes != nullptr
            && options.backend != 0
            && !(options.backend == 2
                 && context->formula == FRACTAL_FORMULA_MANDELBROT)) {
            throw std::runtime_error(
                "render metadata requires the native scalar backend for this formula");
        }
        if (context->escape_radius_mode != options.escape_radius_mode) {
            throw std::runtime_error(
                "reference escape-radius mode does not match render options");
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
        if (zoom_log10 >= 6.0L && context->formula != FRACTAL_FORMULA_MANDELBROT) {
#ifdef FRACTAL_HAVE_MPFR
            if (options.backend == 2) {
#ifdef FRACTAL_HAVE_OPENCL
                render_deep_perturbation_opencl(
                    output, width, height, zoom_text, *context, max_iter, options, nullptr);
#else
                throw std::runtime_error("OpenCL backend is not available in this build");
#endif
            } else {
            RenderStats stats;
            if (render_stats_enabled.load(std::memory_order_relaxed)) {
                render_alternate_reference_impl(
                    output, width, height, zoom_text, *context, max_iter,
                    threads, options, &stats, nullptr, nullptr, planes);
                publish_render_stats(stats);
            } else {
                render_alternate_reference_impl(
                    output, width, height, zoom_text, *context, max_iter,
                    threads, options, nullptr, nullptr, nullptr, planes);
            }
            }
#else
            throw std::runtime_error("deep rendering requires MPFR/GMP; rebuild with make");
#endif
        } else if (zoom_log10 >= 6.0L) {
#ifdef FRACTAL_HAVE_MPFR
            if (options.backend == 2 && zoom_log10 >= 12.0L) {
#ifdef FRACTAL_HAVE_OPENCL
                render_deep_perturbation_opencl(
                    output, width, height, zoom_text, *context, max_iter, options, planes);
#else
                throw std::runtime_error("OpenCL backend is not available in this build");
#endif
            } else {
                RenderStats stats;
                if (render_stats_enabled.load(std::memory_order_relaxed)) {
                    render_bla_dispatch<true>(output, width, height, zoom_text, *context, max_iter,
                                              threads, series_order, series_block, options, &stats,
                                              nullptr, nullptr, planes);
                    publish_render_stats(stats);
                } else {
                    render_bla_dispatch<false>(output, width, height, zoom_text, *context, max_iter,
                                               threads, series_order, series_block, options, nullptr,
                                               nullptr, nullptr, planes);
                }
            }
#else
            (void)zoom;
            throw std::runtime_error("deep rendering requires MPFR/GMP; rebuild with make");
#endif
        } else {
#ifdef FRACTAL_HAVE_MPFR
            const long double zoom = parse_zoom(zoom_text);
#endif
            if (planes != nullptr) {
                render_direct_with_planes(
                    output,
                    width,
                    height,
                    zoom,
                    context->x_center,
                    context->y_center,
                    max_iter,
                    threads,
                    context->formula,
                    context->julia_real,
                    context->julia_imag,
                    options.output_bias,
                    options.escape_radius_mode,
                    options.coordinate_mode,
                    static_cast<std::uint32_t>(options.reserved[0]),
                    planes);
            } else {
                render_direct(
                    output,
                    width,
                    height,
                    zoom,
                    context->x_center,
                    context->y_center,
                    max_iter,
                    threads,
                    options.backend,
                    context->formula,
                    context->julia_real,
                    context->julia_imag,
                    options.output_bias,
                    options.escape_radius_mode,
                    options.coordinate_mode);
            }
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

int fractal_render_reference_ex(
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
    return fractal_render_reference_ex_planes(
        output,
        width,
        height,
        zoom_text,
        handle,
        max_iter,
        threads,
        series_order,
        series_block,
        supplied_options,
        nullptr);
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
    return fractal_render_reference_ex(
        output, width, height, zoom_text, handle, max_iter, threads,
        series_order, series_block, supplied_options);
}

// Render an arbitrary list of scaled perturbations.  The Python exp-map
// layer uses this to sample (log radius, angle) coordinates directly; the
// numerical core and its validated series/BLA machinery remain shared with
// rectangular atlas tiles.
int fractal_render_points_impl(
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
    const FractalRenderOptions* supplied_options,
    FractalRenderPlanes* planes
) {
    try {
        if (!output || point_count <= 0 || point_count > MAX_NATIVE_POINTS || !zoom_text
            || !real_mantissa || !imag_mantissa || !exponents || !handle
            || (planes != nullptr && !valid_render_planes(planes, point_count, 1))) {
            throw std::runtime_error("invalid native point-render arguments");
        }
        if (!valid_iteration_count(max_iter) || !valid_thread_count(threads)
            || !valid_series_parameters(series_order, series_block)) {
            throw std::runtime_error("invalid native point-render limits");
        }
        const FractalRenderOptions options = checked_render_options(supplied_options);
        if (options.backend != 0 && options.backend != 1 && options.backend != 2) {
            throw std::runtime_error("unknown native render backend");
        }
        if (planes != nullptr && options.backend != 0) {
            throw std::runtime_error(
                "point metadata requires the native scalar backend");
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
        if (options.backend == 2) {
#ifdef FRACTAL_HAVE_OPENCL
            if (options.coordinate_mode != COORDINATE_MODE_PROJECT) {
                throw std::runtime_error(
                    "OpenCL point rendering requires project coordinates");
            }
            render_deep_perturbation_opencl(
                output,
                point_count,
                1,
                zoom_text,
                *context,
                max_iter,
                options,
                nullptr,
                &points);
#else
            throw std::runtime_error("OpenCL backend is not available in this build");
#endif
        } else if (context->formula != FRACTAL_FORMULA_MANDELBROT) {
            if (render_stats_enabled.load(std::memory_order_relaxed)) {
                render_alternate_reference_impl(
                    output,
                    point_count,
                    1,
                    zoom_text,
                    *context,
                    max_iter,
                    threads,
                    options,
                    &stats,
                    &points,
                    &point_radius,
                    planes);
                publish_render_stats(stats);
            } else {
                render_alternate_reference_impl(
                    output,
                    point_count,
                    1,
                    zoom_text,
                    *context,
                    max_iter,
                    threads,
                    options,
                    nullptr,
                    &points,
                    &point_radius,
                    planes);
            }
        } else {
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
                    &point_radius,
                    planes);
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
                    &point_radius,
                    planes);
            }
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
    return fractal_render_points_impl(
        output,
        point_count,
        zoom_text,
        real_mantissa,
        imag_mantissa,
        exponents,
        handle,
        max_iter,
        threads,
        series_order,
        series_block,
        supplied_options,
        nullptr);
}

int fractal_render_points_ex_planes(
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
    const FractalRenderOptions* supplied_options,
    FractalRenderPlanes* planes
) {
    return fractal_render_points_impl(
        output,
        point_count,
        zoom_text,
        real_mantissa,
        imag_mantissa,
        exponents,
        handle,
        max_iter,
        threads,
        series_order,
        series_block,
        supplied_options,
        planes);
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

// Versioned one-shot entry point for all supported formulas. The direct and
// deep paths share the same scalar field contract; the formula-aware deep
// reference keeps the alternate recurrence native instead of falling back to
// Python at the precision boundary.
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
                julia_imag,
                options.output_bias,
                options.escape_radius_mode,
                options.coordinate_mode);
        } else {
            char julia_real_text[64];
            char julia_imag_text[64];
            if (std::snprintf(
                    julia_real_text, sizeof(julia_real_text), "%.17g", julia_real)
                < 0
                || std::snprintf(
                    julia_imag_text, sizeof(julia_imag_text), "%.17g", julia_imag)
                    < 0) {
                throw std::runtime_error("failed to format the Julia constant");
            }
            void* context_handle = fractal_create_reference_ex_options(
                x_center, y_center, zoom_text, max_iter, precision_bits, 8,
                formula, julia_real_text, julia_imag_text, &options);
            if (!context_handle) throw std::runtime_error(last_error);
            const int status = fractal_render_reference_ex(
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

int render_fractal_ex_planes(
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
    const FractalRenderOptions* supplied_options,
    FractalRenderPlanes* planes
) {
    try {
        if (!output || !zoom_text || !x_center || !y_center
            || !valid_pixel_dimensions(width, height)
            || !valid_render_planes(planes, width, height)
            || !valid_iteration_count(max_iter)
            || !valid_precision_bits(precision_bits)
            || use_perturbation < 0 || use_perturbation > 1
            || !valid_thread_count(threads)
            || !valid_formula(formula)
            || !std::isfinite(julia_real) || !std::isfinite(julia_imag)) {
            throw std::runtime_error(
                "render metadata requires valid native render arguments");
        }
        const FractalRenderOptions options = checked_render_options(supplied_options);
        if (options.backend != 0
            && !(options.backend == 2
                 && formula == FRACTAL_FORMULA_MANDELBROT)) {
            throw std::runtime_error(
                "render metadata requires the native scalar backend for this formula");
        }
        if (use_perturbation == 0) {
            const long double zoom = parse_zoom(zoom_text);
            render_direct_with_planes(
                output,
                width,
                height,
                zoom,
                parse_coordinate(x_center, "real"),
                parse_coordinate(y_center, "imaginary"),
                max_iter,
                threads,
                formula,
                julia_real,
                julia_imag,
                options.output_bias,
                options.escape_radius_mode,
                options.coordinate_mode,
                static_cast<std::uint32_t>(options.reserved[0]),
                planes);
        } else {
            char julia_real_text[64];
            char julia_imag_text[64];
            if (std::snprintf(
                    julia_real_text, sizeof(julia_real_text), "%.17g", julia_real)
                < 0
                || std::snprintf(
                    julia_imag_text, sizeof(julia_imag_text), "%.17g", julia_imag)
                    < 0) {
                throw std::runtime_error("failed to format the Julia constant");
            }
            void* context_handle = fractal_create_reference_ex_options(
                x_center,
                y_center,
                zoom_text,
                max_iter,
                precision_bits,
                8,
                formula,
                julia_real_text,
                julia_imag_text,
                &options);
            if (!context_handle) throw std::runtime_error(last_error);
            const int status = fractal_render_reference_ex_planes(
                output,
                width,
                height,
                zoom_text,
                context_handle,
                max_iter,
                threads,
                8,
                32,
                &options,
                planes);
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
        set_error("native metadata render failed with an unknown exception");
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
