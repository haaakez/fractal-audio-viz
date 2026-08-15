// Mandelbrot renderer with reusable MPFR reference orbits, scaled
// mantissa/exponent perturbation, and hierarchical BLA maps. The exported
// functions intentionally use a C ABI so Python can drive the native core
// without knowing its C++ types.

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef FRACTAL_HAVE_MPFR
#include <mpfr.h>
#endif

namespace {

constexpr int ABI_VERSION = 6;
constexpr long double ESCAPE_RADIUS_SQUARED = 4.0L;
constexpr long double LOG_TWO = 0.693147180559945309417232121458176568L;

std::mutex error_mutex;
std::string last_error;

struct PaletteBasis {
    int max_iter = -1;
    std::vector<float> cosine;
    std::vector<float> sine;
};

// The colourizer is called from Python's calling thread, while OpenMP only
// parallelises its pixel loop.  Thread-local storage therefore gives each
// caller a reusable LUT without locks or cross-renderer interference.
thread_local PaletteBasis palette_basis;

void set_error(const std::string& message) {
    std::lock_guard<std::mutex> lock(error_mutex);
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
        const double angle = 0.19 * static_cast<double>(max_iter)
            * static_cast<double>(index) / denominator;
        palette_basis.cosine[static_cast<size_t>(index)] = static_cast<float>(std::cos(angle));
        palette_basis.sine[static_cast<size_t>(index)] = static_cast<float>(std::sin(angle));
    }
    return palette_basis;
}

inline std::uint8_t colour_byte(double value) {
    return static_cast<std::uint8_t>(std::clamp(value, 0.0, 255.0));
}

long double parse_zoom(const char* text) {
    if (!text) throw std::runtime_error("null Mandelbrot zoom");
    errno = 0;
    char* end = nullptr;
    const long double zoom = std::strtold(text, &end);
    if (end == text || *end != '\0' || errno == ERANGE
        || !std::isfinite(zoom) || zoom <= 0.0L) {
        throw std::runtime_error("invalid Mandelbrot zoom");
    }
    return zoom;
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
        return {normalized, exponent + shift};
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
    const int difference = larger->exponent - smaller->exponent;
    if (difference > 60) return *larger;
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
        return {mantissa * 0.5, larger->exponent + 1};
    }
    if (magnitude >= 0.5) {
        return {mantissa, larger->exponent};
    }
    int shift = 0;
    return {std::frexp(mantissa, &shift), larger->exponent + shift};
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
    if (magnitude >= 1.0) {
        return {mantissa * 0.5, a.exponent + b.exponent + 1};
    }
    if (magnitude < 0.5) {
        return {mantissa * 2.0, a.exponent + b.exponent - 1};
    }
    return {mantissa, a.exponent + b.exponent};
}

inline FloatExp fe_mul(const FloatExp& a, double b) {
    return fe_mul(a, FloatExp::from_parts(static_cast<long double>(b), 0));
}

inline FloatExp fe_div(const FloatExp& a, const FloatExp& b) {
    if (a.zero()) return {0.0, 0};
    const double mantissa = a.mantissa / b.mantissa;
    if (!std::isfinite(mantissa)) return FloatExp::from_parts(mantissa, 0);
    const double magnitude = std::abs(mantissa);
    if (magnitude >= 1.0) {
        return {mantissa * 0.5, a.exponent - b.exponent + 1};
    }
    if (magnitude < 0.5) {
        return {mantissa * 2.0, a.exponent - b.exponent - 1};
    }
    return {mantissa, a.exponent - b.exponent};
}

inline FloatExp fe_sqr(const FloatExp& value) {
    return fe_mul(value, value);
}

inline FloatExp fe_sqrt(const FloatExp& value) {
    if (value.zero()) return value;
    int exponent = value.exponent;
    double mantissa = value.mantissa;
    if (mantissa < 0.0 || !std::isfinite(mantissa)) {
        return FloatExp::from_parts(std::sqrt(mantissa), 0);
    }
    if (exponent & 1) {
        mantissa *= 2.0;
        --exponent;
        // sqrt(2 * normalized_mantissa) is in [1, sqrt(2)); normalize the
        // result directly rather than sending it through frexp.
        return {std::sqrt(mantissa) * 0.5, exponent / 2 + 1};
    }
    return {std::sqrt(mantissa), exponent / 2};
}

inline int fe_compare(const FloatExp& a, const FloatExp& b) {
    if (a.mantissa == 0.0 && b.mantissa == 0.0) return 0;
    if (a.mantissa < 0.0 && b.mantissa >= 0.0) return -1;
    if (a.mantissa >= 0.0 && b.mantissa < 0.0) return 1;
    const bool negative = a.mantissa < 0.0;
    if (a.exponent != b.exponent) {
        const int result = a.exponent < b.exponent ? -1 : 1;
        return negative ? -result : result;
    }
    const int result = (a.mantissa > b.mantissa) - (a.mantissa < b.mantissa);
    return negative ? -result : result;
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

struct BlaLevels {
    std::vector<std::vector<BlaStep>> levels;
    FloatExp input_radius;

    const BlaStep* lookup(int start, const FloatExp& delta_norm_squared) const noexcept {
        if (start <= 0) return nullptr;
        const int base_count = levels.empty() ? 0 : static_cast<int>(levels[0].size());
        if (start > base_count) return nullptr;
        const BlaStep* selected = nullptr;
        for (size_t level = 0; level < levels.size(); ++level) {
            const int span = 1 << std::min<size_t>(level, 30);
            if ((start - 1) % span != 0) break;
            const int index = (start - 1) / span;
            if (index >= static_cast<int>(levels[level].size())) break;
            const BlaStep& candidate = levels[level][static_cast<size_t>(index)];
            if (fe_compare(delta_norm_squared, candidate.radius_squared) >= 0) break;
            selected = &candidate;
        }
        return selected;
    }

};

struct ReferenceContext {
    std::vector<FloatExpComplex> fast_orbit;
    BlaLevels bla;
    long double x_center = 0.0L;
    long double y_center = 0.0L;
#ifdef FRACTAL_HAVE_MPFR
    mpfr_prec_t precision_bits = 0;
#endif
};

void render_direct(
    float* output,
    int width,
    int height,
    long double zoom,
    long double x_center,
    long double y_center,
    int max_iter,
    int threads
) {
    // This path is deliberately ordinary double precision.  The Python
    // layer routes only shallow views here; using long double for every
    // pixel made the inexpensive part of a zoom sequence disproportionately
    // slow on low-power CPUs.
    const double height_span = 2.8 / static_cast<double>(zoom);
    const double width_span = height_span * static_cast<double>(width) / static_cast<double>(height);
    const double center_real = static_cast<double>(x_center);
    const double center_imag = static_cast<double>(y_center);
#ifdef _OPENMP
    if (threads > 0) omp_set_num_threads(threads);
#pragma omp parallel for schedule(static)
#endif
    for (int py = 0; py < height; ++py) {
        const double y_offset =
            (static_cast<double>(height - 1) / 2.0 - static_cast<double>(py))
            * height_span / static_cast<double>(height);
        for (int px = 0; px < width; ++px) {
            const double x_offset =
                (static_cast<double>(px) - static_cast<double>(width - 1) / 2.0)
                * width_span / static_cast<double>(width);
            const double cx = center_real + x_offset;
            const double cy = center_imag + y_offset;
            const double q = (cx - 0.25) * (cx - 0.25) + cy * cy;
            const bool in_cardioid = q * (q + cx - 0.25) <= 0.25 * cy * cy;
            const bool in_bulb = (cx + 1.0) * (cx + 1.0) + cy * cy <= 0.0625;
            const int index = py * width + px;
            if (in_cardioid || in_bulb) {
                output[index] = static_cast<float>(max_iter);
                continue;
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

#ifdef FRACTAL_HAVE_MPFR

void make_reference_orbit(
    ReferenceContext& context,
    const char* x_text,
    const char* y_text,
    const char* viewport_zoom_text,
    int max_iter,
    int precision_bits
) {
    precision_bits = std::max(128, precision_bits);
    mpfr_t cx, cy, viewport_zoom, viewport_radius, zr, zi, next_real, next_imag, temporary;
    mpfr_init2(cx, precision_bits); mpfr_init2(cy, precision_bits);
    mpfr_init2(viewport_zoom, precision_bits); mpfr_init2(viewport_radius, precision_bits);
    mpfr_init2(zr, precision_bits); mpfr_init2(zi, precision_bits);
    mpfr_init2(next_real, precision_bits); mpfr_init2(next_imag, precision_bits);
    mpfr_init2(temporary, precision_bits);
    if (mpfr_set_str(cx, x_text, 10, MPFR_RNDN) != 0 || mpfr_set_str(cy, y_text, 10, MPFR_RNDN) != 0) {
        mpfr_clears(cx, cy, viewport_zoom, viewport_radius, zr, zi, next_real, next_imag, temporary, nullptr);
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
    context.fast_orbit.assign(static_cast<size_t>(max_iter) + 1U, {});
    for (int i = 0; i <= max_iter; ++i) {
        context.fast_orbit[static_cast<size_t>(i)] = {
            FloatExp::from_mpfr(zr), FloatExp::from_mpfr(zi)
        };
        mpfr_mul(next_real, zr, zr, MPFR_RNDN);
        mpfr_mul(temporary, zi, zi, MPFR_RNDN);
        mpfr_sub(next_real, next_real, temporary, MPFR_RNDN);
        mpfr_add(next_real, next_real, cx, MPFR_RNDN);
        mpfr_mul(next_imag, zr, zi, MPFR_RNDN);
        mpfr_mul_ui(next_imag, next_imag, 2, MPFR_RNDN);
        mpfr_add(next_imag, next_imag, cy, MPFR_RNDN);
        mpfr_set(zr, next_real, MPFR_RNDN); mpfr_set(zi, next_imag, MPFR_RNDN);
    }
    mpfr_clears(cx, cy, viewport_zoom, viewport_radius, zr, zi, next_real, next_imag, temporary, nullptr);
}

#else

void make_reference_orbit(ReferenceContext&, const char*, const char*, const char*, int, int) {
    throw std::runtime_error("deep rendering requires MPFR/GMP; rebuild with make");
}

#endif

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

void build_bla(ReferenceContext& context) {
    const int max_iter = static_cast<int>(context.fast_orbit.size()) - 1;
    const int base_count = std::max(0, max_iter - 1);
    context.bla.levels.clear();
    if (base_count == 0) return;

    context.bla.levels.emplace_back(static_cast<size_t>(base_count));
    // Keep the fast map below a conservative relative error budget.  The
    // visualizer uses the iteration value for smooth colouring, so a map
    // that is merely visually plausible is not enough at a keyframe seam.
    // The endpoint guard below replays maps that approach escape, so this
    // radius may be looser than machine epsilon without changing the smooth
    // colouring boundary.  1e-14 enables useful map reuse before the zoom
    // reaches the extreme e20+ range while retaining a large safety margin.
    const FloatExp tolerance = FloatExp::from_long_double(1.0e-14L);
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
        context.bla.levels[0][static_cast<size_t>(start - 1)] = step;
    }

    for (size_t level = 1; ; ++level) {
        const size_t previous_size = context.bla.levels[level - 1].size();
        const size_t current_size = previous_size / 2;
        if (current_size == 0) break;
        context.bla.levels.emplace_back(current_size);
        for (size_t index = 0; index < current_size; ++index) {
            context.bla.levels[level][index] = merge_bla(
                context.bla.levels[level - 1][index * 2 + 1],
                context.bla.levels[level - 1][index * 2],
                context.bla.input_radius);
        }
    }
}

#ifdef FRACTAL_HAVE_MPFR

FloatExp parse_zoom_float_exp(const char* text, mpfr_prec_t precision_bits) {
    mpfr_t value;
    mpfr_init2(value, precision_bits);
    const int status = mpfr_set_str(value, text, 10, MPFR_RNDN);
    if (status != 0 || mpfr_sgn(value) <= 0 || !mpfr_number_p(value)) {
        mpfr_clear(value);
        throw std::runtime_error("invalid deep Mandelbrot zoom");
    }
    const FloatExp result = FloatExp::from_mpfr(value);
    mpfr_clear(value);
    return result;
}

float smooth_escape_float_exp(int iteration, const FloatExp& magnitude_squared) {
    const long double log_magnitude = 0.5L * fe_log(magnitude_squared);
    if (!(log_magnitude > 0.0L) || !std::isfinite(log_magnitude)) {
        return static_cast<float>(iteration);
    }
    return static_cast<float>(static_cast<long double>(iteration)
        - std::log(log_magnitude) / LOG_TWO);
}

void render_double_tail(
    float& output,
    const FloatExpComplex& dc,
    const ReferenceContext& context,
    int max_iter,
    int& iteration,
    int& reference_index,
    FloatExpComplex delta
) {
    // Once the perturbation is comfortably above the double ulp of an
    // O(1) reference orbit, carrying a separate exponent is unnecessary.
    // Finish the escape test in ordinary doubles; this is the hot path for
    // deep boundary pixels and avoids paying frexp/exponent costs for every
    // remaining iteration.
    double dc_real = static_cast<double>(dc.real.as_long_double());
    double dc_imag = static_cast<double>(dc.imag.as_long_double());
    double delta_real = static_cast<double>(delta.real.as_long_double());
    double delta_imag = static_cast<double>(delta.imag.as_long_double());
    output = static_cast<float>(max_iter);
    while (iteration < max_iter) {
        const FloatExpComplex& reference =
            context.fast_orbit[static_cast<size_t>(reference_index)];
        const double reference_real = static_cast<double>(reference.real.as_long_double());
        const double reference_imag = static_cast<double>(reference.imag.as_long_double());
        const double linear_real = 2.0 * (reference_real * delta_real - reference_imag * delta_imag);
        const double linear_imag = 2.0 * (reference_real * delta_imag + reference_imag * delta_real);
        const double square_real = delta_real * delta_real - delta_imag * delta_imag;
        const double square_imag = 2.0 * delta_real * delta_imag;
        delta_real = linear_real + square_real + dc_real;
        delta_imag = linear_imag + square_imag + dc_imag;
        ++reference_index;
        ++iteration;
        const double total_real = static_cast<double>(context.fast_orbit[static_cast<size_t>(reference_index)].real.as_long_double()) + delta_real;
        const double total_imag = static_cast<double>(context.fast_orbit[static_cast<size_t>(reference_index)].imag.as_long_double()) + delta_imag;
        const double magnitude_squared = total_real * total_real + total_imag * total_imag;
        if (magnitude_squared > 4.0) {
            const double magnitude = std::sqrt(std::max(magnitude_squared, 4.0000001));
            output = static_cast<float>(static_cast<double>(iteration)
                - std::log(std::log(magnitude)) / static_cast<double>(LOG_TWO));
            return;
        }
        if (total_real * total_real + total_imag * total_imag
            < delta_real * delta_real + delta_imag * delta_imag) {
            delta_real = total_real;
            delta_imag = total_imag;
            reference_index = 0;
        }
    }
}

void render_bla(
    float* output,
    int width,
    int height,
    const char* zoom_text,
    const ReferenceContext& context,
    int max_iter,
    int threads,
    int series_order,
    int series_block
) {
    const FloatExp zoom = parse_zoom_float_exp(zoom_text, context.precision_bits);
    const FloatExp inverse_zoom = fe_div(FloatExp::from_parts(1.0, 0), zoom);
    const FloatExp view_height = fe_mul(inverse_zoom, 2.8);
    const FloatExp view_width = fe_mul(
        view_height, static_cast<double>(width) / static_cast<double>(height));
    const FloatExp escape_squared = FloatExp::from_parts(4.0, 0);
    const FloatExp conservative_escape_squared = FloatExp::from_parts(3.0, 0);
    const FloatExp current_input_radius = fe_mul(inverse_zoom, 2.8);
    const bool bla_radius_covers_view =
        fe_compare(current_input_radius, context.bla.input_radius) <= 0;
    const bool disable_bla = !bla_radius_covers_view
        || std::getenv("FRACTAL_DISABLE_BLA") != nullptr;
    // The existing BLA hierarchy is a real polynomial approximation, not a
    // compatibility label.  Use its lower-precision tail only after the
    // perturbation has grown large enough for ordinary doubles; keeping BLA
    // active through e100 is what turns reference reuse into iteration reuse.
    // The cubic terms suppress the accumulated error that limited the old
    // quadratic map to short blocks.
    const int max_bla_length = std::clamp(series_block, 2, 4096);
    const int approximation_order = std::clamp(series_order, 1, 3);

#ifdef _OPENMP
    if (threads > 0) omp_set_num_threads(threads);
#pragma omp parallel for schedule(static)
#endif
    for (int py = 0; py < height; ++py) {
        const double y_fraction =
            (static_cast<double>(height - 1) / 2.0 - static_cast<double>(py))
            / static_cast<double>(height);
        const FloatExp y_offset = fe_mul(view_height, y_fraction);
        for (int px = 0; px < width; ++px) {
            const double x_fraction =
                (static_cast<double>(px) - static_cast<double>(width - 1) / 2.0)
                / static_cast<double>(width);
            const FloatExp x_offset = fe_mul(view_width, x_fraction);
            const FloatExpComplex dc{x_offset, y_offset};
            const FloatExpComplex dc_squared = fec_mul(dc, dc);
            const FloatExpComplex dc_cubed = fec_mul(dc_squared, dc);
            const int index = py * width + px;
            FloatExpComplex delta = dc;
            int reference_index = 1;
            int iteration = 1;
            bool escaped = false;

            while (iteration < max_iter) {
                if (reference_index < 0 || reference_index >= static_cast<int>(context.fast_orbit.size())) {
                    output[index] = static_cast<float>(max_iter);
                    escaped = true;
                    break;
                }
                const FloatExpComplex total = fec_add(
                    context.fast_orbit[static_cast<size_t>(reference_index)], delta);
                const FloatExp total_norm_squared = fec_norm_squared(total);
                if (fe_compare(total_norm_squared, escape_squared) > 0) {
                    output[index] = smooth_escape_float_exp(iteration, total_norm_squared);
                    escaped = true;
                    break;
                }

                // Rebase to the beginning of the same reference orbit when
                // the perturbation is larger than the reference state.  This
                // is the cheap glitch-avoidance step used by deep zoomers.
                const FloatExp delta_norm_squared = fec_norm_squared(delta);
                if (fe_compare(total_norm_squared, delta_norm_squared) < 0) {
                    delta = total;
                    reference_index = 0;
                    continue;
                }

                // A normal double keeps roughly 53 significant bits.  Once
                // |delta| is above about 2^-45, its evolution has ample
                // headroom above double rounding noise, so the hot tail can
                // use ordinary complex doubles without discarding visible
                // pixel differences.  The previous 2^-22 threshold left
                // many e12--e14 frames in the much slower FloatExp loop.
                if (delta_norm_squared.exponent > -90) {
                    render_double_tail(output[index], dc, context, max_iter,
                                       iteration, reference_index, delta);
                    escaped = true;
                    break;
                }

                const BlaStep* step = disable_bla
                    ? nullptr : context.bla.lookup(reference_index, delta_norm_squared);
                if (step && step->length > max_bla_length) step = nullptr;
                if (step && reference_index + step->length <= max_iter) {
                    const FloatExpComplex previous_delta = delta;
                    const int previous_reference_index = reference_index;
                    const int previous_iteration = iteration;
                    const FloatExpComplex delta_squared = fec_mul(delta, delta);
                    const FloatExpComplex delta_cubed = fec_mul(delta_squared, delta);
                    const FloatExpComplex delta_dc = fec_mul(delta, dc);
                    const FloatExpComplex delta_squared_dc = fec_mul(delta_squared, dc);
                    const FloatExpComplex delta_dc_squared = fec_mul(delta, dc_squared);
                    delta = fec_add(fec_mul(step->A, delta), fec_mul(step->B, dc));
                    if (approximation_order >= 2) {
                        delta = fec_add(
                            delta,
                            fec_add(
                                fec_add(fec_mul(step->C, delta_squared), fec_mul(step->D, delta_dc)),
                                fec_mul(step->E, dc_squared)));
                    }
                    if (approximation_order >= 3) {
                        delta = fec_add(
                            delta,
                            fec_add(
                                fec_add(fec_mul(step->F, delta_cubed),
                                        fec_mul(step->G, delta_squared_dc)),
                                fec_add(fec_mul(step->H, delta_dc_squared),
                                        fec_mul(step->I, dc_cubed))));
                    }
                    reference_index += step->length;
                    iteration += step->length;
                    const FloatExpComplex endpoint = fec_add(
                        context.fast_orbit[static_cast<size_t>(reference_index)], delta);
                    const FloatExp endpoint_norm_squared = fec_norm_squared(endpoint);
                    // A block that approaches the escape boundary is replayed
                    // one iteration at a time so smooth colouring does not
                    // acquire broad BLA-sized bands.
                    if (fe_compare(endpoint_norm_squared, conservative_escape_squared) >= 0) {
                        delta = previous_delta;
                        reference_index = previous_reference_index;
                        iteration = previous_iteration;
                        for (int replay = 0; replay < step->length && iteration < max_iter; ++replay) {
                            const FloatExpComplex reference =
                                context.fast_orbit[static_cast<size_t>(reference_index)];
                            delta = fec_add(
                                fec_add(fec_mul(reference, delta), fec_mul(delta, reference)),
                                fec_add(fec_mul(delta, delta), dc));
                            ++reference_index;
                            ++iteration;
                            const FloatExpComplex replay_total = fec_add(
                                context.fast_orbit[static_cast<size_t>(reference_index)], delta);
                            const FloatExp replay_norm_squared = fec_norm_squared(replay_total);
                            if (fe_compare(replay_norm_squared, escape_squared) > 0) {
                                output[index] = smooth_escape_float_exp(iteration, replay_norm_squared);
                                escaped = true;
                                break;
                            }
                        }
                        if (escaped) break;
                    }
                    continue;
                }

                // One exact perturbation step.  The expression is written as
                // 2*Z*delta + delta^2 + delta_c, but the symmetric product
                // keeps the same operation count as a complex multiply.
                const FloatExpComplex reference =
                    context.fast_orbit[static_cast<size_t>(reference_index)];
                delta = fec_add(
                    fec_add(fec_mul(reference, delta), fec_mul(delta, reference)),
                    fec_add(fec_mul(delta, delta), dc));
                ++reference_index;
                ++iteration;
            }
            if (!escaped) output[index] = static_cast<float>(max_iter);
        }
    }
}

#endif

} // namespace

extern "C" {

int fractal_abi_version() { return ABI_VERSION; }

const char* fractal_last_error() {
    std::lock_guard<std::mutex> lock(error_mutex);
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
    int threads
) {
    try {
        if (!field || !output || width <= 0 || height <= 0 || max_iter <= 0) {
            throw std::runtime_error("invalid native colour dimensions or palette");
        }
        const PaletteBasis& basis = colour_basis_for(max_iter);
        const int palette_size = static_cast<int>(basis.cosine.size());
        // Match the float32 indexing used by the Python fallback.  A double
        // product can occasionally select the adjacent 65k-entry palette
        // bin, which is visible as a several-level RGB difference.
        const float index_scale = static_cast<float>(palette_size - 1)
            / static_cast<float>(max_iter);
        const double split = 0.7 + 3.0 * vocal * vocal;
        const double brightness = 0.65 + 0.35 * instrumental;
        const double red_cos = std::cos(phase);
        const double red_sin = std::sin(phase);
        const double green_cos = std::cos(phase + split * 0.35);
        const double green_sin = std::sin(phase + split * 0.35);
        const double blue_cos = std::cos(phase + split);
        const double blue_sin = std::sin(phase + split);
        const double red_gain = (150.0 + 80.0 * vocal) * brightness;
        const double green_gain = 180.0 * brightness;
        const double blue_gain = 210.0 * brightness;
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
            const int palette_index = std::clamp(
                static_cast<int>(smooth * index_scale),
                0,
                palette_size - 1);
            const double cosine = basis.cosine[static_cast<size_t>(palette_index)];
            const double sine = basis.sine[static_cast<size_t>(palette_index)];
            const double red_wave = cosine * red_cos + sine * red_sin;
            const double green_wave = cosine * green_cos + sine * green_sin;
            const double blue_wave = cosine * blue_cos + sine * blue_sin;
            rgb[0] = colour_byte((0.5 - 0.5 * red_wave) * red_gain);
            rgb[1] = colour_byte((0.5 - 0.5 * green_wave) * green_gain);
            rgb[2] = colour_byte((0.5 - 0.5 * blue_wave) * blue_gain);
        }
        set_error("");
        return 0;
    } catch (const std::exception& error) {
        set_error(error.what());
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
    int threads
) {
    try {
        if (!source || !output || source_width <= 0 || source_height <= 0
            || output_width <= 0 || output_height <= 0 || max_iter <= 0
            || !std::isfinite(zoom_factor) || zoom_factor <= 0.0) {
            throw std::runtime_error("invalid native crop/colour dimensions or palette");
        }
        zoom_factor = std::max(zoom_factor, 1.0);
        const double inverse_zoom = 1.0 / zoom_factor;
        const double crop_width = static_cast<double>(source_width) * inverse_zoom;
        const double crop_height = static_cast<double>(source_height) * inverse_zoom;
        const double left = (static_cast<double>(source_width) - crop_width) * 0.5;
        const double top = (static_cast<double>(source_height) - crop_height) * 0.5;
        const PaletteBasis& basis = colour_basis_for(max_iter);
        const int palette_size = static_cast<int>(basis.cosine.size());
        const float index_scale = static_cast<float>(palette_size - 1)
            / static_cast<float>(max_iter);
        const double split = 0.7 + 3.0 * vocal * vocal;
        const double brightness = 0.65 + 0.35 * instrumental;
        const double red_cos = std::cos(phase);
        const double red_sin = std::sin(phase);
        const double green_cos = std::cos(phase + split * 0.35);
        const double green_sin = std::sin(phase + split * 0.35);
        const double blue_cos = std::cos(phase + split);
        const double blue_sin = std::sin(phase + split);
        const double red_gain = (150.0 + 80.0 * vocal) * brightness;
        const double green_gain = 180.0 * brightness;
        const double blue_gain = 210.0 * brightness;

#ifdef _OPENMP
        if (threads > 0) omp_set_num_threads(threads);
#pragma omp parallel for schedule(static)
#endif
        for (int output_y = 0; output_y < output_height; ++output_y) {
            double source_y = top
                + (static_cast<double>(output_y) + 0.5) * crop_height
                    / static_cast<double>(output_height)
                - 0.5;
            source_y = std::clamp(source_y, 0.0, static_cast<double>(source_height - 1));
            const int y0 = static_cast<int>(std::floor(source_y));
            const int y1 = std::min(y0 + 1, source_height - 1);
            const double y_fraction = source_y - static_cast<double>(y0);
            for (int output_x = 0; output_x < output_width; ++output_x) {
                double source_x = left
                    + (static_cast<double>(output_x) + 0.5) * crop_width
                        / static_cast<double>(output_width)
                    - 0.5;
                source_x = std::clamp(source_x, 0.0, static_cast<double>(source_width - 1));
                const int x0 = static_cast<int>(std::floor(source_x));
                const int x1 = std::min(x0 + 1, source_width - 1);
                const double x_fraction = source_x - static_cast<double>(x0);
                const double top_value =
                    static_cast<double>(source[y0 * source_width + x0]) * (1.0 - x_fraction)
                    + static_cast<double>(source[y0 * source_width + x1]) * x_fraction;
                const double bottom_value =
                    static_cast<double>(source[y1 * source_width + x0]) * (1.0 - x_fraction)
                    + static_cast<double>(source[y1 * source_width + x1]) * x_fraction;
                const float smooth = static_cast<float>(
                    top_value * (1.0 - y_fraction) + bottom_value * y_fraction);
                std::uint8_t* rgb = output + static_cast<size_t>(
                    output_y * output_width + output_x) * 3U;
                if (!std::isfinite(smooth)
                    || smooth >= static_cast<float>(max_iter) - 0.5F) {
                    rgb[0] = 0;
                    rgb[1] = 0;
                    rgb[2] = 0;
                    continue;
                }
                const int palette_index = std::clamp(
                    static_cast<int>(smooth * index_scale), 0, palette_size - 1);
                const double cosine = basis.cosine[static_cast<size_t>(palette_index)];
                const double sine = basis.sine[static_cast<size_t>(palette_index)];
                const double red_wave = cosine * red_cos + sine * red_sin;
                const double green_wave = cosine * green_cos + sine * green_sin;
                const double blue_wave = cosine * blue_cos + sine * blue_sin;
                rgb[0] = colour_byte((0.5 - 0.5 * red_wave) * red_gain);
                rgb[1] = colour_byte((0.5 - 0.5 * green_wave) * green_gain);
                rgb[2] = colour_byte((0.5 - 0.5 * blue_wave) * blue_gain);
            }
        }
        set_error("");
        return 0;
    } catch (const std::exception& error) {
        set_error(error.what());
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
        if (!x_center || !y_center || !viewport_zoom || max_iter <= 0
            || series_order < 1 || series_order > 32) {
            throw std::runtime_error("invalid reference configuration");
        }
        auto context = std::make_unique<ReferenceContext>();
        context->x_center = std::strtold(x_center, nullptr);
        context->y_center = std::strtold(y_center, nullptr);
        make_reference_orbit(*context, x_center, y_center, viewport_zoom, max_iter, precision_bits);
        build_bla(*context);
        set_error("");
        return context.release();
    } catch (const std::exception& error) {
        set_error(error.what());
        return nullptr;
    }
}

void fractal_destroy_reference(void* handle) {
    delete static_cast<ReferenceContext*>(handle);
}

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
    try {
        if (!output || !zoom_text || !handle || width <= 0 || height <= 0) {
            throw std::runtime_error("invalid native render dimensions or handle");
        }
        auto* context = static_cast<ReferenceContext*>(handle);
        if (max_iter <= 0 || max_iter >= static_cast<int>(context->fast_orbit.size())) {
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
            render_bla(output, width, height, zoom_text, *context, max_iter, threads,
                       series_order, series_block);
#else
            (void)zoom;
            throw std::runtime_error("deep rendering requires MPFR/GMP; rebuild with make");
#endif
        } else {
#ifdef FRACTAL_HAVE_MPFR
            const long double zoom = parse_zoom(zoom_text);
#endif
            render_direct(output, width, height, zoom, context->x_center, context->y_center, max_iter, threads);
        }
        set_error("");
        return 0;
    } catch (const std::exception& error) {
        set_error(error.what());
        return 1;
    }
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
    try {
        if (!output || !zoom_text || !x_center || !y_center
            || width <= 0 || height <= 0 || max_iter <= 0) {
            throw std::runtime_error("invalid native render dimensions or argument");
        }
        if (!use_perturbation) {
            const long double zoom = parse_zoom(zoom_text);
            render_direct(output, width, height, zoom, std::strtold(x_center, nullptr),
                          std::strtold(y_center, nullptr), max_iter, threads);
        } else {
            std::unique_ptr<ReferenceContext> context(static_cast<ReferenceContext*>(
                fractal_create_reference(x_center, y_center, zoom_text, max_iter, precision_bits, 8)));
            if (!context) throw std::runtime_error(last_error);
            const int status = render_mandelbrot_reference(
                output, width, height, zoom_text, context.get(), max_iter, threads, 8, 32);
            if (status != 0) throw std::runtime_error(last_error);
        }
        return 0;
    } catch (const std::exception& error) {
        set_error(error.what());
        return 1;
    }
}

} // extern "C"
