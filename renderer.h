#ifndef FRACTAL_VIZ_RENDERER_H
#define FRACTAL_VIZ_RENDERER_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define FRACTAL_ABI_VERSION 10
#define FRACTAL_RENDER_OPTIONS_VERSION 1
#define FRACTAL_KFP_OPTIONS_VERSION 1
#define FRACTAL_KFP_MAX_MULTI_COLORS 256

/* Formula ids used by render_fractal_ex. */
#define FRACTAL_FORMULA_MANDELBROT 0
#define FRACTAL_FORMULA_JULIA 1
#define FRACTAL_FORMULA_BURNING_SHIP 2
#define FRACTAL_FORMULA_TRICORN 3

/*
 * Per-call controls for the native renderer.  Callers should initialize this
 * with fractal_render_options_default() and keep struct_size/version intact.
 * No render decision depends on process-global environment variables.  In a
 * strict deep render, a float NaN in the output means that the compact
 * reference detected a perturbation glitch and the caller must refine that
 * pixel/region with another reference; it is never an interior colour.
 */
typedef struct FractalRenderOptions {
    uint32_t struct_size;
    uint32_t version;
    int32_t strict;
    int32_t allow_recovery;
    int32_t time_budget_ms;
    int32_t disable_bla;
    int32_t disable_cycle;
    int32_t strict_cycle;
    int32_t series_min_terms;
    int32_t series_max_terms;
    int32_t max_bla_length;
    int32_t max_linear_bla_length;
    int32_t backend;
    int32_t reserved[3];
} FractalRenderOptions;

/*
 * Portable Kalles Fraktaler colour-transfer controls.  The scalar field and
 * LUT stay owned by the caller; the native colouriser only reads them.  The
 * fixed-size multi-colour arrays keep this extension ABI-safe and avoid
 * handing C++ containers through ctypes.
 */
typedef struct FractalKfpOptions {
    uint32_t struct_size;
    uint32_t version;
    double iter_div;
    double color_offset;
    double ratio;
    int32_t color_method;
    int32_t smooth_method;
    int32_t smooth;
    int32_t flat;
    int32_t inverse_transition;
    double phase_color_strength;
    int32_t multi_color;
    int32_t blend_multi_color;
    uint32_t multi_color_count;
    double multi_color_period[FRACTAL_KFP_MAX_MULTI_COLORS];
    int32_t multi_color_start[FRACTAL_KFP_MAX_MULTI_COLORS];
    int32_t multi_color_type[FRACTAL_KFP_MAX_MULTI_COLORS];
    double power;
    int32_t slopes;
    double slope_power;
    double slope_ratio;
    double slope_angle;
    int32_t differences;
    int32_t interior_color[3];
} FractalKfpOptions;

int fractal_abi_version(void);
int fractal_render_options_version(void);
void fractal_render_options_default(FractalRenderOptions *options);
int fractal_backend_capabilities(void);
const char *fractal_last_error(void);

void fractal_set_stats_enabled(int enabled);
int fractal_get_last_stats(uint64_t *values, int capacity);
int fractal_get_last_stats_ex(uint64_t *values, int capacity);

void *fractal_create_reference(
    const char *x_center,
    const char *y_center,
    const char *viewport_zoom,
    int max_iter,
    int precision_bits,
    int series_order
);
/* Retains the compact builder orbit so radius-specific tiers can clone it. */
void *fractal_create_reference_reusable(
    const char *x_center,
    const char *y_center,
    const char *viewport_zoom,
    int max_iter,
    int precision_bits,
    int series_order
);
/* Rebuild only radius-dependent series/BLA tables around a shared orbit. */
void *fractal_clone_reference(void *source_handle, const char *viewport_zoom);
void fractal_destroy_reference(void *handle);
int fractal_get_reference_stats(void *handle, uint64_t *values, int capacity);

int fractal_render_mandelbrot_reference_ex(
    float *output,
    int width,
    int height,
    const char *zoom_text,
    void *handle,
    int max_iter,
    int threads,
    int series_order,
    int series_block,
    const FractalRenderOptions *options
);

int fractal_render_points(
    float *output,
    int point_count,
    const char *zoom_text,
    const double *real_mantissa,
    const double *imag_mantissa,
    const int32_t *exponents,
    void *handle,
    int max_iter,
    int threads,
    int series_order,
    int series_block,
    const FractalRenderOptions *options
);

int render_mandelbrot_ex(
    float *output,
    int width,
    int height,
    const char *zoom_text,
    const char *x_center,
    const char *y_center,
    int max_iter,
    int precision_bits,
    int use_perturbation,
    int threads,
    const FractalRenderOptions *options
);

/* Direct renderer for the Mandelbrot family.  Alternate formulas are
 * intentionally direct/shallow for now; the validated MPFR+BLA deep path
 * remains attached to the Mandelbrot parameter plane. */
int render_fractal_ex(
    float *output,
    int width,
    int height,
    const char *zoom_text,
    const char *x_center,
    const char *y_center,
    int max_iter,
    int precision_bits,
    int use_perturbation,
    int threads,
    int formula,
    double julia_real,
    double julia_imag,
    const FractalRenderOptions *options
);

/* Compatibility entry points retained for older callers. */
int render_mandelbrot_reference(
    float *output,
    int width,
    int height,
    const char *zoom_text,
    void *handle,
    int max_iter,
    int threads,
    int series_order,
    int series_block
);
int render_mandelbrot(
    float *output,
    int width,
    int height,
    const char *zoom_text,
    const char *x_center,
    const char *y_center,
    int max_iter,
    int precision_bits,
    int use_perturbation,
    int threads
);

int fractal_colourise(
    const float *field,
    uint8_t *output,
    int width,
    int height,
    int max_iter,
    double phase,
    double vocal,
    double instrumental,
    double pitch,
    int threads
);
int fractal_apply_aurora_accents(
    uint8_t *output,
    int width,
    int height,
    const uint8_t *accents,
    double pitch,
    int threads
);
int fractal_colourise_kfp(
    const float *field,
    uint8_t *output,
    int width,
    int height,
    int max_iter,
    double phase,
    double vocal,
    double instrumental,
    double pitch,
    const FractalKfpOptions *options,
    const uint8_t *lut,
    int lut_size,
    int threads
);
int fractal_atlas_colourise_kfp(
    const float *parent,
    int parent_width,
    int parent_height,
    const float *child,
    int child_width,
    int child_height,
    uint8_t *output,
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
    const FractalKfpOptions *options,
    const uint8_t *lut,
    int lut_size,
    int threads
);
int fractal_crop_field(
    const float *source,
    int source_width,
    int source_height,
    float *output,
    int output_width,
    int output_height,
    double zoom_factor,
    int threads
);
int fractal_crop_colourise(
    const float *source,
    int source_width,
    int source_height,
    uint8_t *output,
    int output_width,
    int output_height,
    double zoom_factor,
    int max_iter,
    double phase,
    double vocal,
    double instrumental,
    double pitch,
    int threads
);
int fractal_atlas_colourise(
    const float *parent,
    int parent_width,
    int parent_height,
    int parent_max_iter,
    const float *child,
    int child_width,
    int child_height,
    int child_max_iter,
    uint8_t *output,
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
);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* FRACTAL_VIZ_RENDERER_H */
