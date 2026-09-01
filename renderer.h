#ifndef FRACTAL_VIZ_RENDERER_H
#define FRACTAL_VIZ_RENDERER_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define FRACTAL_ABI_VERSION 10
#define FRACTAL_RENDER_OPTIONS_VERSION 1

/*
 * Per-call controls for the native renderer.  Callers should initialize this
 * with fractal_render_options_default() and keep struct_size/version intact.
 * No render decision depends on process-global environment variables.
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
