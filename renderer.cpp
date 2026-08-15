// Mandelbrot renderer with reusable MPFR reference orbits, scaled
// mantissa/exponent perturbation, and hierarchical BLA maps. The exported
// functions intentionally use a C ABI so Python can drive the native core
// without knowing its C++ types.

#include <algorithm>
#include <array>
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

constexpr int ABI_VERSION = 8;
// Degree-three maps have been independently checked through this block size.
// Larger maps are still accepted by the ABI but deliberately fall back to
// this validated limit until higher-order composition is implemented.
constexpr int MAX_SAFE_BLA_LENGTH = 256;
constexpr long double ESCAPE_RADIUS_SQUARED = 4.0L;
constexpr long double LOG_TWO = 0.693147180559945309417232121458176568L;

std::mutex error_mutex;
std::string last_error;

struct PaletteBasis {
    int max_iter = -1;
    std::vector<float> cosine;
    std::vector<float> sine;
};

struct AuroraPalette {
    std::vector<std::array<std::uint8_t, 3>> rgb;
};

// The colourizer is called from Python's calling thread, while OpenMP only
// parallelises its pixel loop.  Thread-local storage therefore gives each
// caller a reusable LUT without locks or cross-renderer interference.
thread_local PaletteBasis palette_basis;
thread_local AuroraPalette aurora_palette;

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
    const double split = 0.7 + 3.0 * vocal_mix * vocal_mix;
    const double brightness = 0.65 + 0.35 * instrumental_mix;
    const double red_cos = std::cos(phase);
    const double red_sin = std::sin(phase);
    const double green_cos = std::cos(phase + split * 0.35);
    const double green_sin = std::sin(phase + split * 0.35);
    const double blue_cos = std::cos(phase + split);
    const double blue_sin = std::sin(phase + split);
    const double red_gain = (150.0 + 80.0 * vocal_mix) * brightness;
    const double green_gain = 180.0 * brightness;
    const double blue_gain = 210.0 * brightness;
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

BilinearAxis make_bilinear_axis(
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

    BilinearAxis axis;
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
    return axis;
}

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
    const int palette_index = std::clamp(
        static_cast<int>(smooth * palette_index_scale),
        0,
        palette_size - 1);
    const auto& colour = palette.rgb[static_cast<size_t>(palette_index)];
    destination[0] = colour[0];
    destination[1] = colour[1];
    destination[2] = colour[2];
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
            ++exponent;
            return;
        }
        if (magnitude >= 0.5) return;
        if (magnitude >= 0.25) {
            real *= 2.0;
            imag *= 2.0;
            --exponent;
            return;
        }
        int shift = 0;
        (void)std::frexp(magnitude, &shift);
        real = std::ldexp(real, -shift);
        imag = std::ldexp(imag, -shift);
        exponent += shift;
    }

    static ScaledComplex from_float_exp(const FloatExp& real_part, const FloatExp& imag_part) {
        if (real_part.zero() && imag_part.zero()) return {};
        const int common_exponent = std::max(real_part.exponent, imag_part.exponent);
        ScaledComplex result{
            real_part.exponent - common_exponent < -1074
                ? 0.0 : std::ldexp(real_part.mantissa, real_part.exponent - common_exponent),
            imag_part.exponent - common_exponent < -1074
                ? 0.0 : std::ldexp(imag_part.mantissa, imag_part.exponent - common_exponent),
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
    const int difference = larger->exponent - smaller->exponent;
    if (difference > 60) return *larger;
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
        a.exponent + b.exponent,
    };
    result.normalize();
    return result;
}

inline ScaledComplex sc_double(const ScaledComplex& value) {
    if (value.real == 0.0 && value.imag == 0.0) return {};
    // Multiplication by two is exact in this representation and does not
    // need another mantissa normalization.
    return {value.real, value.imag, value.exponent + 1};
}

struct ScaledNorm {
    double mantissa = 0.0;
    int exponent = 0;
};

inline ScaledNorm sc_norm_squared(const ScaledComplex& value) {
    const double norm = value.real * value.real + value.imag * value.imag;
    if (norm == 0.0) return {};
    if (norm >= 1.0) {
        return {norm * 0.5, value.exponent * 2 + 1};
    }
    if (norm < 0.5) {
        return {norm * 2.0, value.exponent * 2 - 1};
    }
    return {norm, value.exponent * 2};
}

inline int sc_compare_norm(const ScaledNorm& a, const ScaledNorm& b) {
    if (a.mantissa == 0.0 && b.mantissa == 0.0) return 0;
    if (a.mantissa == 0.0) return -1;
    if (b.mantissa == 0.0) return 1;
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

struct FastBlaStep {
    std::array<ScaledComplex, 9> coefficients{};
    FloatExp radius_squared;
    int length = 1;
};

FastBlaStep compact_bla_step(const BlaStep& step) {
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
        step.radius_squared,
        step.length,
    };
    return result;
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
    FloatExp input_radius;

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
        if (start <= 0 || max_length <= 0) return nullptr;
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
        for (int level = highest_level; level >= 0; --level) {
            const int span = 1 << level;
            if ((start - 1) % span != 0) continue;
            const int index = (start - 1) / span;
            if (index >= static_cast<int>(levels[level].size())) continue;
            const FastBlaStep& candidate = levels[level][static_cast<size_t>(index)];
            if (fe_compare(delta_norm_squared, candidate.radius_squared) < 0) {
                return &candidate;
            }
        }
        return nullptr;
    }

};

struct ReferenceContext {
    std::vector<FloatExpComplex> fast_orbit;
    std::vector<ScaledComplex> fast_orbit_scaled;
    std::vector<double> orbit_real_double;
    std::vector<double> orbit_imag_double;
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
#pragma omp parallel for schedule(dynamic, 1)
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
    context.fast_orbit_scaled.resize(context.fast_orbit.size());
    for (size_t index = 0; index < context.fast_orbit.size(); ++index) {
        context.fast_orbit_scaled[index] = ScaledComplex::from_float_exp(
            context.fast_orbit[index].real,
            context.fast_orbit[index].imag);
    }
    context.orbit_real_double.resize(context.fast_orbit_scaled.size());
    context.orbit_imag_double.resize(context.fast_orbit_scaled.size());
    for (size_t index = 0; index < context.fast_orbit_scaled.size(); ++index) {
        const ScaledComplex& value = context.fast_orbit_scaled[index];
        context.orbit_real_double[index] = std::ldexp(value.real, value.exponent);
        context.orbit_imag_double[index] = std::ldexp(value.imag, value.exponent);
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

    std::vector<std::vector<BlaStep>> builder_levels;
    builder_levels.emplace_back(static_cast<size_t>(base_count));
    // Keep the fast map below a conservative relative error budget.  The
    // visualizer uses the iteration value for smooth colouring, so a map
    // that is merely visually plausible is not enough at a keyframe seam.
    // The endpoint guard below replays maps that approach escape, so this
    // radius may be looser than machine epsilon without changing the smooth
    // colouring boundary.  A 1e-8 relative bound gives deep frames useful
    // blocks while leaving a large margin for smooth-colour stability.
    const FloatExp tolerance = FloatExp::from_long_double(1.0e-8L);
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
        // Compose the orbit map independently of the current frame's
        // parameter radius.  The old code passed the shallowest-frame radius
        // here (typically ~1e-6), which made the conservative
        // `y.radius - |x.B| * input_radius` bound negative for practically
        // every multi-iteration block.  That collapsed the hierarchy to
        // length-one maps and turned deep keyframes back into an expensive
        // perturbation loop.  The per-frame viewport check in render_bla()
        // still prevents using the map outside its intended zoom range;
        // deep frames have a tiny enough dc that the zero-radius composition
        // bound is the useful reusable limit.
        const FloatExp composition_input_radius{0.0, 0};
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
    context.fast_orbit.clear();
    context.fast_orbit.shrink_to_fit();
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

void render_scaled_double_tail(
    float& output,
    const ScaledComplex& dc,
    const ReferenceContext& context,
    int max_iter,
    int& iteration,
    int& reference_index,
    ScaledComplex delta
) {
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
    bool cycle_ready = false;
    while (iteration < max_iter) {
        const double reference_real =
            context.orbit_real_double[static_cast<size_t>(reference_index)];
        const double reference_imag =
            context.orbit_imag_double[static_cast<size_t>(reference_index)];
        const double linear_real = 2.0 * (reference_real * delta_real - reference_imag * delta_imag);
        const double linear_imag = 2.0 * (reference_real * delta_imag + reference_imag * delta_real);
        const double square_real = delta_real * delta_real - delta_imag * delta_imag;
        const double square_imag = 2.0 * delta_real * delta_imag;
        delta_real = linear_real + square_real + dc_real;
        delta_imag = linear_imag + square_imag + dc_imag;
        ++reference_index;
        ++iteration;
        const double total_real =
            context.orbit_real_double[static_cast<size_t>(reference_index)] + delta_real;
        const double total_imag =
            context.orbit_imag_double[static_cast<size_t>(reference_index)] + delta_imag;
        const double magnitude_squared = total_real * total_real + total_imag * total_imag;
        if (magnitude_squared > 4.0) {
            const double magnitude = std::sqrt(std::max(magnitude_squared, 4.0000001));
            output = static_cast<float>(static_cast<double>(iteration)
                - std::log(std::log(magnitude)) / static_cast<double>(LOG_TWO));
            return;
        }
        const double delta_magnitude_squared =
            delta_real * delta_real + delta_imag * delta_imag;

        ++tail_steps;
        if (tail_steps >= 64) {
            // Brent-style cycle detection is deliberately conservative: it
            // only runs after a long bounded tail and requires a near-exact
            // recurrence well inside the escape circle. This lets attracting
            // interior pixels finish early without turning a transient
            // boundary revisit into a false black classification.
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
                if (iteration >= 512
                    && magnitude_squared < 3.0
                    && cycle_distance_squared
                        <= 1.0e-24 * std::max(1.0, magnitude_squared)) {
                    return;
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
        if (magnitude_squared < delta_magnitude_squared) {
            delta_real = total_real;
            delta_imag = total_imag;
            reference_index = 0;
            tail_steps = 0;
            cycle_ready = false;
        }
    }
}

void render_bla(
    float* __restrict output,
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
    const int max_bla_length = std::clamp(series_block, 2, MAX_SAFE_BLA_LENGTH);
    const int approximation_order = std::clamp(series_order, 1, 3);

    // These offsets are shared by every pixel in a row/column.  Computing
    // the FloatExp multiplication once here removes two viewport-scale
    // operations from the innermost perturbation loop without changing the
    // exact pixel-centre mapping.
    std::vector<FloatExp> x_offsets(static_cast<size_t>(width));
    std::vector<FloatExp> y_offsets(static_cast<size_t>(height));
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

#ifdef _OPENMP
    if (threads > 0) omp_set_num_threads(threads);
#pragma omp parallel for schedule(dynamic, 1)
#endif
    for (int py = 0; py < height; ++py) {
        const FloatExp& y_offset = y_offsets[static_cast<size_t>(py)];
        for (int px = 0; px < width; ++px) {
            const FloatExp& x_offset = x_offsets[static_cast<size_t>(px)];
            const ScaledComplex dc = ScaledComplex::from_float_exp(x_offset, y_offset);
            const int index = py * width + px;
            ScaledComplex delta = dc;
            int reference_index = 1;
            int iteration = 1;
            bool escaped = false;
            bool have_total = false;
            ScaledComplex total{};
            ScaledNorm total_norm{};
            ScaledComplex cycle_tortoise{};
            int cycle_power = 1;
            int cycle_length = 0;
            bool cycle_ready = false;

            while (iteration < max_iter) {
                if (reference_index < 0
                    || reference_index >= static_cast<int>(context.fast_orbit_scaled.size())) {
                    output[index] = static_cast<float>(max_iter);
                    escaped = true;
                    break;
                }
                if (!have_total) {
                    total = sc_add(
                        context.fast_orbit_scaled[static_cast<size_t>(reference_index)], delta);
                    total_norm = sc_norm_squared(total);
                }
                have_total = false;
                if (sc_outside_escape(total_norm)) {
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
                if (iteration >= 512
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
                            || cycle_distance.exponent <= scale_exponent - 78) {
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

                // Rebase to the beginning of the same reference orbit when
                // the perturbation is larger than the reference state.  This
                // is the cheap glitch-avoidance step used by deep zoomers.
                const ScaledNorm delta_norm = sc_norm_squared(delta);
                if (sc_compare_norm(total_norm, delta_norm) < 0) {
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
                if (delta_norm.exponent > -90) {
                    render_scaled_double_tail(output[index], dc, context, max_iter,
                                       iteration, reference_index, delta);
                    escaped = true;
                    break;
                }

                const FloatExp delta_norm_float{delta_norm.mantissa, delta_norm.exponent};
                const int remaining_iterations = max_iter - iteration;
                const FastBlaStep* step = disable_bla
                    ? nullptr
                    : context.bla.lookup(
                        reference_index,
                        delta_norm_float,
                        std::min(max_bla_length, remaining_iterations));
                if (step && reference_index + step->length <= max_iter) {
                    const ScaledComplex previous_delta = delta;
                    const int previous_reference_index = reference_index;
                    const int previous_iteration = iteration;
                    const ScaledComplex input_delta = delta;
                    // Far below the BLA radius, the quadratic and cubic
                    // terms are many orders smaller than the requested
                    // float output. Keep the full series for the numerically
                    // sensitive approach to escape, but use the cheaper
                    // linear Horner branch while the perturbation is tiny.
                    // The thresholds are on |delta|^2 in binary exponent
                    // form; -160 still leaves roughly 80 bits of margin
                    // before the nonlinear terms can affect a float pixel.
                    const int effective_order =
                        approximation_order >= 3 && delta_norm.exponent < -115
                            ? (delta_norm.exponent < -160 ? 1 : 2)
                            : approximation_order;
                    delta = apply_bla_series(
                        *step,
                        input_delta,
                        dc,
                        effective_order);
                    reference_index += step->length;
                    iteration += step->length;
                    const ScaledComplex endpoint = sc_add(
                        context.fast_orbit_scaled[static_cast<size_t>(reference_index)], delta);
                    const ScaledNorm endpoint_norm = sc_norm_squared(endpoint);
                    // A block that approaches the escape boundary is replayed
                    // one iteration at a time so smooth colouring does not
                    // acquire broad BLA-sized bands.
                    if (sc_compare_norm(endpoint_norm, ScaledNorm{0.75, 2}) >= 0) {
                        delta = previous_delta;
                        reference_index = previous_reference_index;
                        iteration = previous_iteration;
                        ScaledComplex replay_total{};
                        ScaledNorm replay_norm{};
                        for (int replay = 0; replay < step->length && iteration < max_iter; ++replay) {
                            const ScaledComplex reference =
                                context.fast_orbit_scaled[static_cast<size_t>(reference_index)];
                            delta = sc_add(
                                sc_double(sc_mul(reference, delta)),
                                sc_add(sc_mul(delta, delta), dc));
                            ++reference_index;
                            ++iteration;
                            replay_total = sc_add(
                                context.fast_orbit_scaled[static_cast<size_t>(reference_index)], delta);
                            replay_norm = sc_norm_squared(replay_total);
                            if (sc_outside_escape(replay_norm)) {
                                output[index] = smooth_escape_scaled(iteration, replay_norm);
                                escaped = true;
                                break;
                            }
                        }
                        if (escaped) break;
                        total = replay_total;
                        total_norm = replay_norm;
                        have_total = true;
                    } else {
                        total = endpoint;
                        total_norm = endpoint_norm;
                        have_total = true;
                    }
                    continue;
                }

                // One exact perturbation step.  The expression is written as
                // 2*Z*delta + delta^2 + delta_c, but the symmetric product
                // keeps the same operation count as a complex multiply.
                const ScaledComplex reference =
                    context.fast_orbit_scaled[static_cast<size_t>(reference_index)];
                delta = sc_add(
                    sc_double(sc_mul(reference, delta)),
                    sc_add(sc_mul(delta, delta), dc));
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
    double pitch,
    int threads
) {
    try {
        if (!field || !output || width <= 0 || height <= 0 || max_iter <= 0) {
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
            const int palette_index = std::clamp(
                static_cast<int>(smooth * index_scale),
                0,
                palette_size - 1);
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
        const AuroraPalette& palette = aurora_palette_for(
            max_iter, phase, vocal, instrumental, pitch);
        const int palette_size = static_cast<int>(palette.rgb.size());
        const float index_scale = static_cast<float>(palette_size - 1)
            / static_cast<float>(max_iter);
        // The horizontal crop mapping is identical for every output row.
        // Compute the floor/clamp work once instead of repeating it for every
        // pixel in the inner loop.
        std::vector<int> x0_map(static_cast<size_t>(output_width));
        std::vector<int> x1_map(static_cast<size_t>(output_width));
        std::vector<double> x_fraction_map(static_cast<size_t>(output_width));
        for (int output_x = 0; output_x < output_width; ++output_x) {
            double source_x = left
                + (static_cast<double>(output_x) + 0.5) * crop_width
                    / static_cast<double>(output_width)
                - 0.5;
            source_x = std::clamp(source_x, 0.0, static_cast<double>(source_width - 1));
            const int x0 = static_cast<int>(std::floor(source_x));
            x0_map[static_cast<size_t>(output_x)] = x0;
            x1_map[static_cast<size_t>(output_x)] = std::min(x0 + 1, source_width - 1);
            x_fraction_map[static_cast<size_t>(output_x)] = source_x - static_cast<double>(x0);
        }

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
            const float* top_row = source + static_cast<size_t>(y0) * source_width;
            const float* bottom_row = source + static_cast<size_t>(y1) * source_width;
            for (int output_x = 0; output_x < output_width; ++output_x) {
                const size_t x_index = static_cast<size_t>(output_x);
                const int x0 = x0_map[x_index];
                const int x1 = x1_map[x_index];
                const double x_fraction = x_fraction_map[x_index];
                const double top_value =
                    static_cast<double>(top_row[x0]) * (1.0 - x_fraction)
                    + static_cast<double>(top_row[x1]) * x_fraction;
                const double bottom_value =
                    static_cast<double>(bottom_row[x0]) * (1.0 - x_fraction)
                    + static_cast<double>(bottom_row[x1]) * x_fraction;
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
            || parent_width <= 0 || parent_height <= 0
            || output_width <= 0 || output_height <= 0
            || parent_max_iter <= 0
            || !std::isfinite(parent_zoom) || parent_zoom <= 0.0
            || !std::isfinite(child_fraction)
            || child_fraction < 0.0 || child_fraction > 1.0) {
            throw std::runtime_error("invalid native atlas dimensions or controls");
        }
        const bool use_child = child != nullptr && child_fraction > 0.0;
        if (use_child && (child_width <= 0 || child_height <= 0 || child_max_iter <= 0)) {
            throw std::runtime_error("invalid native atlas child tile");
        }
        const int effective_palette_iter = std::max(
            1,
            std::max(palette_max_iter, std::max(parent_max_iter, child_max_iter)));
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
        BilinearAxis parent_x_axis;
        BilinearAxis parent_y_axis;
        if (!full_child) {
            parent_x_axis = make_bilinear_axis(parent_width, output_width, parent_zoom);
            parent_y_axis = make_bilinear_axis(parent_height, output_height, parent_zoom);
        }
        BilinearAxis child_x_axis;
        BilinearAxis child_y_axis;
        if (use_child) {
            const int child_destination_width = full_child ? output_width : visible_child_width;
            const int child_destination_height = full_child ? output_height : visible_child_height;
            child_x_axis = make_bilinear_axis(child_width, child_destination_width, 1.0);
            child_y_axis = make_bilinear_axis(child_height, child_destination_height, 1.0);
        }
        std::vector<float> child_edge_x;
        std::vector<float> child_edge_y;
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
                const float smooth = sample_bilinear_mapped(
                    parent,
                    parent_width,
                    parent_x_axis,
                    parent_y_axis,
                    output_x,
                    output_y);
                std::uint8_t* destination = output + static_cast<size_t>(
                    output_y * output_width + output_x) * 3U;
                write_colour_pixel(
                    smooth,
                    parent_max_iter,
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
                    const float smooth = sample_bilinear_mapped(
                        child,
                        child_width,
                        child_x_axis,
                        child_y_axis,
                        output_x,
                        output_y);
                    std::uint8_t* destination = output + static_cast<size_t>(
                        output_y * output_width + output_x) * 3U;
                    write_colour_pixel(
                        smooth,
                        child_max_iter,
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
                const float parent_smooth = sample_bilinear_mapped(
                    parent,
                    parent_width,
                    parent_x_axis,
                    parent_y_axis,
                    output_x,
                    output_y);
                const float child_smooth = sample_bilinear_mapped(
                    child,
                    child_width,
                    child_x_axis,
                    child_y_axis,
                    child_x,
                    child_y);
                std::uint8_t parent_rgb[3];
                std::uint8_t child_rgb[3];
                write_colour_pixel(
                    parent_smooth,
                    parent_max_iter,
                    palette,
                    palette_index_scale,
                    parent_rgb);
                write_colour_pixel(
                    child_smooth,
                    child_max_iter,
                    palette,
                    palette_index_scale,
                    child_rgb);
                const float alpha = feather >= 2
                    ? std::min(
                        child_edge_x[static_cast<size_t>(child_x)],
                        child_edge_y[static_cast<size_t>(child_y)])
                    : 1.0F;
                std::uint8_t* destination = output + static_cast<size_t>(
                    output_y * output_width + output_x) * 3U;
                const double inverse_alpha = 1.0 - static_cast<double>(alpha);
                for (int channel = 0; channel < 3; ++channel) {
                    destination[channel] = colour_byte(
                        static_cast<double>(parent_rgb[channel]) * inverse_alpha
                        + static_cast<double>(child_rgb[channel]) * static_cast<double>(alpha));
                }
            }
            render_parent_span(output_y, child_right, output_width);
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
        if (max_iter <= 0
            || max_iter >= static_cast<int>(context->fast_orbit_scaled.size())) {
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
