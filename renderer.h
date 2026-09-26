#ifndef FRACTAL_VIZ_RENDERER_H
#define FRACTAL_VIZ_RENDERER_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define FRACTAL_ABI_VERSION 10
#define FRACTAL_RENDER_OPTIONS_VERSION 2
#define FRACTAL_KFP_OPTIONS_VERSION 3
#define FRACTAL_KFP_PLANES_VERSION 1
#define FRACTAL_RENDER_PLANES_VERSION 1
#define FRACTAL_KFP_MAX_MULTI_COLORS 256

/* Formula ids used by render_fractal_ex. */
#define FRACTAL_FORMULA_MANDELBROT 0
#define FRACTAL_FORMULA_JULIA 1
#define FRACTAL_FORMULA_BURNING_SHIP 2
#define FRACTAL_FORMULA_TRICORN 3

/* Pixel-grid conventions used by the native field renderer. */
#define FRACTAL_COORDINATE_MODE_PROJECT 0
#define FRACTAL_COORDINATE_MODE_KALLES 1

/* Optimization hints for render_fractal_ex_planes.  These occupy the
 * existing ``reserved[0]`` options word, so the public struct layout and
 * version remain unchanged.  Zero means the legacy behavior: calculate all
 * optional metadata.  A KFP caller may set either bit when its profile does
 * not consume that plane. */
#define FRACTAL_RENDER_HINT_SKIP_PHASE (1u << 0)
#define FRACTAL_RENDER_HINT_SKIP_ANALYTIC_DE (1u << 1)

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
    /* 0 = the normal radius-2 escape test; 1 = Kalles' high (10000)
     * bailout radius. */
    int32_t escape_radius_mode;
    /* Kalles uses integer width/2 and height/2 as the image origin. */
    int32_t coordinate_mode;
    int32_t reserved[1]; /* FRACTAL_RENDER_HINT_* for plane renders */
    /* Optional per-pixel offset used to preserve fractional escape precision
     * when a high-iteration field is stored as float32.  A zero value keeps
     * the historical absolute encoding. */
    double output_bias;
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
    /* The scalar field may be stored relative to this iteration offset. */
    double field_bias;
    /* Modern Kalles smoothing/bailout settings.  The renderer accepts the
     * complete serialized transfer block even when a scalar field cannot
     * carry Kalles' optional derivative/phase planes. */
    int32_t bailout_radius_preset;
    double bailout_radius_custom;
    int32_t bailout_norm_preset;
    double bailout_norm_custom;
    int32_t texture_enabled;
    double texture_merge;
    double texture_power;
    double texture_ratio;
    int32_t texture_resize;
    int32_t use_opengl;
    int32_t use_srgb;
    /* Optional tail field. Version 3 remains valid when this field is
     * absent; missing data means Kalles' default of showing glitches. */
    int32_t show_glitches;
} FractalKfpOptions;

/*
 * Optional per-pixel data produced by Kalles' orbit stage.  The historical
 * KFP entry points accept only the scalar field, so they cannot reproduce
 * OutputIterationData/SetColor features which depend on test1/test2, phase,
 * derivatives, or a texture.  This additive ABI lets a caller pass those
 * planes without changing the old function signatures.
 *
 * All non-texture planes are row-major arrays with width*height entries.
 * ``orbit_iteration`` is the raw ``antal`` counter before
 * OutputIterationData smoothing.  ``iteration`` and ``transition`` are the
 * already materialized nPixels/nTrans values handed to SetColor.  If both
 * forms are supplied, the raw orbit form wins and is smoothed using the test
 * planes.  Missing optional values use the scalar-field compatibility path.
 */
typedef struct FractalKfpPlanes {
    uint32_t struct_size;
    uint32_t version;
    const int64_t *orbit_iteration;
    const double *bailout;
    const int64_t *iteration;
    const double *transition;
    const double *phase;
    const double *de_x;
    const double *de_y;
    const double *test1;
    const double *test2;
    /* Optional top-down RGB texture, row-major with ``texture_stride``
     * bytes per row.  SetTexture samples it bottom-up, like Kalles' DIB. */
    const uint8_t *texture_rgb;
    int32_t texture_width;
    int32_t texture_height;
    int32_t texture_stride;
} FractalKfpPlanes;

/*
 * Optional per-pixel data emitted by a shallow native orbit render.  These
 * planes are intentionally separate from FractalKfpPlanes: the renderer
 * owns and writes them, while the colouriser only reads the same layout.
 * Arrays are row-major with width*height entries.
 */
typedef struct FractalRenderPlanes {
    uint32_t struct_size;
    uint32_t version;
    int64_t *orbit_iteration;
    double *phase;
    double *de_x;
    double *de_y;
    double *test1;
    double *test2;
} FractalRenderPlanes;

int fractal_abi_version(void);
int fractal_render_options_version(void);
void fractal_render_options_default(FractalRenderOptions *options);
/* Bitmask: scalar CPU=1, AVX2 CPU=2, double-precision OpenCL=4,
 * double-precision OpenCL GPU=8. */
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
/* Option-aware variants used by the KFP path.  The legacy creators retain
 * the historical radius-2 behaviour for older callers. */
void *fractal_create_reference_reusable_options(
    const char *x_center,
    const char *y_center,
    const char *viewport_zoom,
    int max_iter,
    int precision_bits,
    int series_order,
    const FractalRenderOptions *options
);
/* Formula-aware reference creation.  The legacy creators remain Mandelbrot
 * wrappers; alternate formulas use this entry point so their exact Julia
 * constant and native deep renderer travel with the reference handle. */
void *fractal_create_reference_ex(
    const char *x_center,
    const char *y_center,
    const char *viewport_zoom,
    int max_iter,
    int precision_bits,
    int series_order,
    int formula,
    const char *julia_real,
    const char *julia_imag
);
void *fractal_create_reference_ex_options(
    const char *x_center,
    const char *y_center,
    const char *viewport_zoom,
    int max_iter,
    int precision_bits,
    int series_order,
    int formula,
    const char *julia_real,
    const char *julia_imag,
    const FractalRenderOptions *options
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

/* Formula-aware version of the reusable-reference renderer. */
int fractal_render_reference_ex(
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

/* Reference-render variant carrying the same orbit metadata as the direct
 * plane ABI.  This is the entry point used by atlas/live renders when a
 * reusable deep reference is already prepared. */
int fractal_render_reference_ex_planes(
    float *output,
    int width,
    int height,
    const char *zoom_text,
    void *handle,
    int max_iter,
    int threads,
    int series_order,
    int series_block,
    const FractalRenderOptions *options,
    FractalRenderPlanes *planes
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

/* Point-render variant carrying the same orbit metadata as the rectangular
 * reference renderer.  The point list remains row-major, so the supplied
 * plane arrays must contain ``point_count`` entries.  Ordinary project-
 * coordinate point repairs also accept backend=2 and use the deep OpenCL
 * recurrence; metadata-bearing/Kalles point renders remain scalar-only. */
int fractal_render_points_ex_planes(
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
    const FractalRenderOptions *options,
    FractalRenderPlanes *planes
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

/* Direct/deep renderer for the complete supported Mandelbrot-family set. */
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

/* Direct/native render with the optional orbit metadata required by the
 * complete Kalles OutputIterationData -> SetColor palette path. */
int render_fractal_ex_planes(
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
    const FractalRenderOptions *options,
    FractalRenderPlanes *planes
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
/* Optional double-precision OpenCL Aurora palette lookup. */
int fractal_colourise_opencl(
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
/* One-pass Aurora-wave colourisation through three ordinary-palette accents. */
int fractal_colourise_accents(
    const float *field,
    uint8_t *output,
    int width,
    int height,
    int max_iter,
    double phase,
    double vocal,
    double instrumental,
    double pitch,
    const uint8_t *accents,
    int interior_red,
    int interior_green,
    int interior_blue,
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
/* Optional OpenCL scalar KFP colour pass. Profiles that require orbit
 * planes, textures, or multi-colour waves use the exact CPU entry point. */
int fractal_colourise_kfp_opencl(
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
/* Full Kalles SetColor transfer when optional per-pixel orbit planes exist. */
int fractal_colourise_kfp_planes(
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
    const FractalKfpPlanes *planes,
    const uint8_t *lut,
    int lut_size,
    int threads
);
/* Compatibility entry point for callers that explicitly request the
 * reference symbol. Both KFP entry points use the source-faithful native
 * scalar transfer path; the symbol is retained so older callers keep working. */
int fractal_colourise_kfp_precise(
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
/* Interior-aware centred crop followed by the native KFP colour pass. */
int fractal_crop_colourise_kfp(
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
    const FractalKfpOptions *options,
    const uint8_t *lut,
    int lut_size,
    int threads
);
int fractal_crop_colourise_kfp_precise(
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
int fractal_atlas_colourise_kfp_precise(
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
/* Reproject parent/child scalar fields and their optional Kalles orbit
 * planes directly in native code, then run one screen-space SetColor pass.
 * ``centered_input`` tells the compositor that each source scalar/plane
 * field is stored relative to its own iteration cap. */
int fractal_atlas_colourise_kfp_planes(
    const float *parent,
    int parent_width,
    int parent_height,
    int parent_max_iter,
    const FractalKfpPlanes *parent_planes,
    const float *child,
    int child_width,
    int child_height,
    int child_max_iter,
    const FractalKfpPlanes *child_planes,
    uint8_t *output,
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
    const FractalKfpOptions *options,
    const uint8_t *lut,
    int lut_size,
    int threads
);
int fractal_atlas_colourise_kfp_planes_precise(
    const float *parent,
    int parent_width,
    int parent_height,
    int parent_max_iter,
    const FractalKfpPlanes *parent_planes,
    const float *child,
    int child_width,
    int child_height,
    int child_max_iter,
    const FractalKfpPlanes *child_planes,
    uint8_t *output,
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
    const FractalKfpOptions *options,
    const uint8_t *lut,
    int lut_size,
    int threads
);
/* Crop raw parent/child scalar tiles into one shared surface, then run one
 * native KFP pass so screen-space differences and slopes remain continuous
 * across the atlas handoff. */
int fractal_atlas_colourise_kfp_raw(
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
    double child_zoom,
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
int fractal_atlas_colourise_kfp_raw_precise(
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
    double child_zoom,
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
/* Crop and composite already-colourised RGB atlas tiles.  Keeping this
 * operation separate from the scalar KFP pass lets video frames reuse the
 * expensive Kalles stencil once per tile instead of once per frame. */
int fractal_crop_rgb(
    const uint8_t *source,
    int source_width,
    int source_height,
    uint8_t *output,
    int output_width,
    int output_height,
    double zoom_factor,
    int threads
);
int fractal_atlas_composite_rgb(
    const uint8_t *parent,
    int parent_width,
    int parent_height,
    const uint8_t *child,
    int child_width,
    int child_height,
    uint8_t *output,
    int output_width,
    int output_height,
    double parent_zoom,
    double child_fraction,
    double child_zoom,
    int feather,
    int threads
);
int fractal_crop_rgb_opencl(
    const uint8_t *source,
    int source_width,
    int source_height,
    uint8_t *output,
    int output_width,
    int output_height,
    double zoom_factor,
    int threads
);
int fractal_atlas_composite_rgb_opencl(
    const uint8_t *parent,
    int parent_width,
    int parent_height,
    const uint8_t *child,
    int child_width,
    int child_height,
    uint8_t *output,
    int output_width,
    int output_height,
    double parent_zoom,
    double child_fraction,
    double child_zoom,
    int feather,
    int threads
);
int fractal_atlas_composite_rgb_opencl_cached(
    const uint8_t *parent,
    int parent_width,
    int parent_height,
    const uint8_t *child,
    int child_width,
    int child_height,
    uint8_t *output,
    int output_width,
    int output_height,
    double parent_zoom,
    double child_fraction,
    double child_zoom,
    int feather,
    int threads,
    uint64_t parent_cache_token,
    uint64_t child_cache_token
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
/* Interior-aware scalar parent/child atlas composition for GPU colour paths. */
int fractal_atlas_field(
    const float *parent,
    int parent_width,
    int parent_height,
    int parent_max_iter,
    const float *child,
    int child_width,
    int child_height,
    int child_max_iter,
    float *output,
    int output_width,
    int output_height,
    double parent_zoom,
    double child_fraction,
    int palette_max_iter,
    int threads
);
/* Extended scalar atlas ABI with source/output bias and child crop control. */
int fractal_atlas_field_ex(
    const float *parent,
    int parent_width,
    int parent_height,
    int parent_max_iter,
    const float *child,
    int child_width,
    int child_height,
    int child_max_iter,
    float *output,
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
);
/* OpenCL ordinary Aurora atlas compositor.  The device keeps immutable
 * parent/child scalar tiles resident when the cache tokens are non-zero and
 * performs bilinear reprojection, the narrow atlas seam, and palette lookup
 * without downloading a composed float frame to the host. */
int fractal_atlas_colourise_opencl(
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
    uint64_t parent_cache_token,
    uint64_t child_cache_token
);
/* OpenCL ordinary atlas compositor with optional three-wave RGB accents and
 * an explicit interior colour for the built-in/custom non-KFP palettes. */
int fractal_atlas_colourise_opencl_accents(
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
    uint64_t parent_cache_token,
    uint64_t child_cache_token,
    const uint8_t *accents,
    int interior_red,
    int interior_green,
    int interior_blue
);
/* Controls for one frame in the batched OpenCL atlas compositor.  Several
 * consecutive frames may share the same scalar parent/child tiles; the
 * native implementation queues their kernels together and performs one
 * device-to-host readback. */
typedef struct FractalAtlasColourFrame {
    double parent_zoom;
    double child_fraction;
    double child_zoom;
    double parent_field_bias;
    double child_field_bias;
    double output_field_bias;
    int32_t palette_max_iter;
    int32_t feather;
    double phase;
    double vocal;
    double instrumental;
    double pitch;
} FractalAtlasColourFrame;
int fractal_atlas_colourise_opencl_batch(
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
    const FractalAtlasColourFrame *frames,
    int frame_count,
    int threads,
    uint64_t parent_cache_token,
    uint64_t child_cache_token,
    const uint8_t *accents,
    int interior_red,
    int interior_green,
    int interior_blue
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
/* Same crop/compositor path with an explicit ordinary-palette interior colour. */
int fractal_crop_colourise_interior(
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
    int interior_red,
    int interior_green,
    int interior_blue,
    int threads
);
/* Same crop path with three Aurora-compatible RGB accents and an interior. */
int fractal_crop_colourise_accents(
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
    const uint8_t *accents,
    int interior_red,
    int interior_green,
    int interior_blue,
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
/* Same atlas path with an explicit ordinary-palette interior colour. */
int fractal_atlas_colourise_interior(
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
    int interior_red,
    int interior_green,
    int interior_blue,
    int threads
);
/* Same atlas path with three Aurora-compatible RGB accents and an interior. */
int fractal_atlas_colourise_accents(
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
    const uint8_t *accents,
    int interior_red,
    int interior_green,
    int interior_blue,
    int threads
);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* FRACTAL_VIZ_RENDERER_H */
