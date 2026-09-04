# fractal audio viz

[![render preview](images/preview.GIF)](images/preview.GIF)


fractal audio viz renders audio-reactive zooms through the Mandelbrot family:
mandelbrot, Julia, Burning Ship, and Tricorn. It reads one local
song, turns the audio into animation controls, renders reusable scalar
keyframes, and sends the final RGB frames to FFmpeg. The Mandelbrot path is
the production path for e150+ deep zooms; the other formulas are first-class
high-precision modes with formula-specific deep boundary targets.

## how it works

the render is split into five stages.

1. **audio analysis.** the input is loaded as mono audio and resampled to
   `--sample-rate`. A frame-aligned loudness envelope drives the camera. YIN
   pitch tracking controls the colour movement. With `--separation auto`, the
   program tries Demucs first; if no usable vocal/instrumental split is
   available, the whole song drives both controls. `--separation spectral`
   enables frequency-band proxies instead. An onset-strength curve is also
   cached; enable its contribution with `--beat-strength`.

2. **zoom planning.** the camera moves in logarithmic zoom space from
   `--base-zoom` to `--max-zoom`. Instrumental energy changes the speed. The
   default `--zoom-speed -0.04` allows small pullbacks during quiet passages,
   while the final frame is still placed exactly at the requested maximum.

3. **source images.** the default `atlas` mode builds a fixed ladder of nested
   keyframes. A keyframe covers one `--keyframe-factor` interval. Frames between
   keyframes are centred crops of the source images, so the expensive fractal
   calculation is not repeated for every video frame. The older full-field
   chunk renderer remains available as `--keyframe-mode legacy`.

4. **fractal rendering.** shallow formula views use the direct native renderer.
   At Mandelbrot e150+ depths, the C++ backend builds a high-precision MPFR
   reference orbit and renders pixel offsets with perturbation arithmetic.
   Scaled mantissa/exponent values preserve tiny offsets, while BLA maps, an
   adaptive image-wide series, OpenMP, and depth-specific reference tiers
   reduce the per-pixel cost. Julia, Burning Ship, and Tricorn use
   formula-aware direct iteration and a high-precision Python perturbation
   fallback for deep boundary targets.

5. **colour and encoding.** keyframes store scalar iteration data. the native
   colour path or Pillow turns that data into RGB after cropping. ordinary
   palettes reuse Aurora's detail-preserving waves with their own RGB accents;
   `.kfp` files keep Kalles' cyclic transfer, distance, multi-colour, and slope
   settings, including the bundled default key colours. KFP crop, atlas, and
   direct colourisation stay in the native path when it is available. atlas levels are
   prepared before they become visible and use an interior-aware feathered
   handoff, so a freshly resolved child tile does not become a rectangle or
   ghost over the parent. optional glow and motion blur are applied at this
   stage, and
   FFmpeg writes the video with the original audio. `--codec auto` probes
   available hardware encoders and falls back to `libx264`.

the KFP path follows Kalles Fraktaler's normal CPU colour pipeline: it builds
the 1024-entry cyclic palette with sine interpolation, keeps the exact
iteration cap as the interior marker, reconstructs interior neighbours as
`max_iter + 1` for distance/slope stencils, applies the selected transfer and
glossy slope pass, and uses Kalles' deterministic 8-bit dither. The lookup
table and immutable profile are cached, and the native implementation performs
the crop, atlas blend, stencil, and colour conversion without a Python image
round-trip. KFP features that require extra Kalles buffers—custom GLSL colour
functions, textures, analytic derivatives, and phase channels—remain outside
the scalar-field compatibility path.

the output dimensions are controlled by `--width` and `--height`. The fractal
source can be smaller or larger than the output: `--render-scale` multiplies
the source size, while `--fractal-scale` sets the requested quality floor.
for example, a 3840×2160 output with `--fractal-scale 0.5` renders 1920×1080
source fields and still produces a 4K video.

### repository layout

| file | purpose |
| --- | --- |
| `visualizer.py` | Audio analysis, zoom planning, keyframes, colour, and FFmpeg. |
| `renderer.cpp` / `renderer.h` | Native C ABI, MPFR deep rendering, BLA, OpenMP, and colour paths. |
| `deep_zoom_points.py` | Formula-specific point catalogues and exact decimal centres. |
| `palettes/` | Built-in palette files, including the Kalles Fraktaler default `.kfp`. |
| `profiles.py` | Shared render presets used by the CLI and GUI. |
| `gui.py` | Optional GTK3/PyGObject launcher; the CLI remains the reproducible interface. |
| `live_view.py` | Low-latency fullscreen GTK preview for listening/screen-saver use. |
| `images/` | README preview and GUI screenshots/placeholders. |
| `make_preview.py` | Converts the newest high-resolution render to a GIF or short MP4. |
| `point_sheet.py` | Creates a labelled catalogue contact sheet. |
| `benchmark.py` | Field, compositor, and atlas benchmarks without audio or video encoding. |
| `tests/` | Python and native correctness tests. |
| `shell.nix` / `Makefile` | Reproducible dependencies and the native build. |

## installation and build

the supplied Nix shell provides Python, NumPy, librosa, Pillow, mpmath,
ffmpeg, GCC, GMP, MPFR, GTK3, and PyGObject. OpenCL support is enabled when the host has
the headers and ICD available. A regular Python environment can use
`pip install -r requirements.txt`, followed by `make` if GMP/MPFR and a C++
compiler are installed.

```sh
nix-shell
make
make test
```

`make` creates `mandelbrot.so`. GMP and MPFR are required for native deep
zooms. The Python renderer is useful for shallow previews and tests, but it is
not a replacement for the native Mandelbrot e150+ path. The GTK launcher is an
optional desktop dependency; headless CLI renders do not need PyGObject or GTK.

## running a render

the input song is the first positional argument. It is not fixed to
`song.mp3`; pass any local file that librosa can decode, such as MP3, WAV, or
flac. If the argument is omitted, the default is `song.mp3`.

```sh
python3 visualizer.py path/to/my-song.mp3 \
  --output renders/my-song.mp4
```

paths containing spaces should be quoted:

```sh
python3 visualizer.py "Music/Live set.flac" \
  --output "renders/Live set.mp4"
```

the program creates the output directory when needed. Use `python3
visualizer.py --help` for the parser’s built-in summary.

for a quick desktop launcher:

```sh
python3 gui.py
```
[![gtk gui](images/gui.png)](images/gui.png)
the GUI only starts the same CLI in a child process, so it stays responsive
and every render can still be copied from the log and reproduced in a shell.
profiles, formulas, palettes, catalogue points, and the audio/effects values
are available at the top. The complete technical options—including precision,
iteration, native backend, encoder, cache, and manifest controls—are collapsed
behind the `Technical options` expander by default; expand it when needed. It
uses GTK's configured system theme and standard controls, plus a real scrolled
container, so collapsing the advanced section cannot leave the form at an
empty scroll offset. The selected palette also has a live preview strip; for
KFP themes it shows the cyclic Kalles gradient and its interior colour.

## constructing a command

start with this shape:

```sh
python3 visualizer.py AUDIO --output OUTPUT [options]
```

then choose the parts that matter for the render:

1. Set the input song and output file.
2. Set `--width`, `--height`, and `--fps`.
3. Choose a formula with `--formula`; for Julia, set its constant with
   `--julia-c`.
4. Choose a centre with `--point`, `--random-point`, or `--x-center` plus
   `--y-center`.
5. Set `--max-zoom` and, for a deep custom point, provide enough decimal
   digits in both coordinates.
6. Choose a quality/speed trade-off with `--quality`, `--fractal-scale`, and
   `--keyframe-factor`.
7. Add `--cache-dir` if the render may be resumed or repeated.

profiles are shortcuts for a coherent set of these values. The canonical
presets are `sd60`, `hd60`, `fhd60`, `2k60`, `4k60`, and `8k60`. Every one uses
60 fps, native fractal resolution, e100, Lanczos resampling, and near-lossless
CRF 10 encoding. With no profile option, `4k60` is selected; explicit options
still override the profile.

```sh
python3 visualizer.py music/track.mp3 --profile 4k60 \
  --point random --random-seed 42 --output renders/track-4k60.mp4
```

inspect the available presets with `python3 visualizer.py --list-profiles`.

for a smaller or larger native-resolution export, change only the profile:

```sh
python3 visualizer.py music/track.mp3 --profile 8k60 \
  --point random --random-seed 42 \
  --output renders/track-8k60.mp4
```

CRF 10 is intended to be visually transparent while retaining useful
compression. Add `--lossless` when exact H.264 losslessness matters more than
file size, or choose a larger CRF explicitly when speed and storage matter
more. Ordinary palettes use the faster 4:2:0 pixel path; KFP palettes and
exact-lossless output use 4:4:4 to retain their sharp colour edges.

### common examples

a normal Full HD render using the canonical preset:

```sh
python3 visualizer.py music/track.mp3 \
  --profile fhd60 \
  --point oldwooddish \
  --output renders/track.mp4 \
  --cache-dir cache/track
```

The old names `preview`, `fullhd`, `1080p`, `4k-e150`, `master-e150`, and
`beat` remain accepted as compatibility aliases for the new resolution tiers.
`4k-e150-lossless` remains accepted too and keeps exact-lossless rate control
while using the safe e100 target.

a reproducible random point:

```sh
python3 visualizer.py music/track.mp3 \
  --output renders/random-track.mp4 \
  --point random --random-seed 42 \
  --max-zoom 1e150 --cache-dir cache/random-track
```

`--random-point` is an equivalent flag-style spelling:

```sh
python3 visualizer.py music/track.mp3 \
  --random-point --random-seed 42 --max-zoom 1e150
```

list the points for the formula you want to render:

```sh
python3 visualizer.py --list-points
python3 visualizer.py --list-points --formula burning-ship
python3 visualizer.py --list-points --formula julia
```

each formula has its own point list. Mandelbrot contains screened e150+ entries
from named KFR files in the [MDZ gallery](https://mathr.co.uk/mdz/gallery/) and
deep test views in [FractalShark](https://github.com/mattsaccount364/FractalShark),
including labelled conjugate mirrors. Burning Ship and Tricorn use their own
formula-specific boundary targets: the Burning Ship list includes generated
helm and mini-ship preperiodic roots, while Tricorn includes generated
branching-junction roots instead of the uninformative `-2` cusp. Julia presets
also include their fixed `c` value, so choosing `julia-douady-rabbit`, for
example, changes both
the preset and the Julia constant. `random` and `--random-point` select from
the catalogue for the currently selected formula. Mandelbrot entries are
screened for the production native e150+ path; alternate entries are screened
as Python high-precision boundary targets at the same depth.

to use your own centre, pass a comma-separated pair to `--point`. This works
for every formula and replaces that formula's preset:

```sh
python3 visualizer.py music/track.mp3 \
  --point=-0.743643887037151,0.131825904205330 \
  --max-zoom 1e12 --allow-underspecified-center
```

for a deep Mandelbrot render, replace that short pair with the full decimal
export from your zoom tool. The production guard requires at least
`ceil(log10(max-zoom)) + 16` fractional decimal places in both coordinates.
this prevents a short decimal from silently producing a different deep
target. `--allow-underspecified-center` is for exploratory Mandelbrot renders
only. Alternate formulas accept their own presets or exact pairs and remain
limited to the exploratory Python path at extreme depth.

the two-coordinate form is also available for scripts and Kalles exports:

```sh
python3 visualizer.py music/track.mp3 \
  --x-center=-0.743643887037151000000000000000000000000000000000 \
  --y-center=0.131825904205330000000000000000000000000000000000 \
  --max-zoom 1e32
```

use either `--point` or the `--x-center`/`--y-center` pair, not both. The
`--point=...` and `--x-center=...` forms are convenient when a negative value
would otherwise be mistaken for another command-line option.

### formula examples

the default is Mandelbrot. The native direct renderer also handles the
alternate formulas below; their coordinates describe the visible viewport.

```sh
# Julia set for c = -0.8 + 0.156i, centred on the Julia plane
python3 visualizer.py music/track.mp3 --formula julia --julia-c=-0.8,0.156 \
  --max-zoom 1e8 --output renders/julia.mp4

# Burning Ship, with onset-driven punches and a custom palette
python3 visualizer.py music/track.mp3 --formula burning-ship \
  --max-zoom 1e8 --beat-strength 1.0 \
  --palette-file examples/palette-neon.txt --output renders/ship.mp4

# The remaining built-in is tricorn (Mandelbar).
python3 visualizer.py music/track.mp3 --formula tricorn --max-zoom 1e7
```

use `python3 visualizer.py --list-formulas` for the complete list. The
validated MPFR/BLA e150+ accelerator is currently specific to the Mandelbrot
parameter plane. Alternate formulas use their own Python high-precision
perturbation path for deep targets up to about e300; they are not silently
routed through an incompatible Mandelbrot reference.

### resolution profiles

the six canonical profiles all render the fractal at the output dimensions;
none of them enlarges a small scalar field into a larger video. They share a
60 fps frame rate, e100 maximum zoom, quality-mode atlas rendering, Lanczos
resampling, and CRF 10 near-lossless compression.

| profile | native output/source | max zoom | compression |
| --- | --- | --- | --- |
| `sd60` | 720×480 | e100 | CRF 10 |
| `hd60` | 1280×720 | e100 | CRF 10 |
| `fhd60` | 1920×1080 | e100 | CRF 10 |
| `2k60` | 2560×1440 | e100 | CRF 10 |
| `4k60` | 3840×2160 | e100 | CRF 10 |
| `8k60` | 7680×4320 | e100 | CRF 10 |

`4k60` is the default. The presets deliberately favour image quality over
the old undersampled e150 speed profile. Add `--lossless` for exact H.264
rate control; use `--quality balanced`, a larger `--keyframe-factor`, or a
larger CRF explicitly when rendering speed matters more.

for example, an 8K render is:

```sh
python3 visualizer.py music/track.mp3 \
  --profile 8k60 \
  --point random --random-seed 42 \
  --output renders/track-8k60.mp4 \
  --cache-dir cache/track-8k60
```

the legacy e150 names are accepted, but resolve to the corresponding
near-lossless native profile; the explicit `4k-e150-lossless` name retains
exact-lossless rate control. This keeps old scripts working while making the
six resolution tiers the single source of truth.

use `--estimate` to analyse the song and print the planned source resolution
and keyframe count without rendering video:

```sh
python3 visualizer.py music/track.mp3 \
  --profile 4k60 --estimate
```

### previews and point browsing

after a render, the small helper chooses the newest filename containing one
of the resolution profile names and creates a looping GIF:

```sh
python3 make_preview.py
```

choose a source or make a short MP4 explicitly:

```sh
python3 make_preview.py renders/track-4k60.mp4 \
  --format mp4 --duration 12 --width 1280 \
  --output renders/track-preview.mp4
```

to browse the curated deep points visually:

```sh
python3 point_sheet.py --output renders/deep-zoom-points.png
```

## `visualizer.py` argument reference

unless noted otherwise, the values in parentheses are the defaults.

### input, output, and audio

| argument | description |
| --- | --- |
| `audio` (`song.mp3`) | Positional input audio file. This is how you choose a custom song. |
| `--output PATH` (`fractal_viz.mp4`) | Output video path. |
| `--profile NAME` | Start from `sd60`, `hd60`, `fhd60`, `2k60`, `4k60` (the default), or `8k60`; legacy names remain accepted as aliases. Later options override the profile. |
| `--list-profiles` | Print the built-in profiles and exit. |
| `--width N` (`3840` with the default profile) | Output width in pixels. |
| `--height N` (`2160` with the default profile) | Output height in pixels. |
| `--fps N` (`60` with the default profile) | Output frame rate and audio-analysis frame rate. |
| `--sample-rate HZ` (`44100`) | Sample rate used while loading and analysing audio. |
| `--separation MODE` (`auto`) | `auto` tries Demucs and falls back to the full mix; `demucs` requires Demucs; `spectral` uses frequency-band proxies; `none` uses the full mix. |

### centre and zoom

| argument | description |
| --- | --- |
| `--point VALUE` | Select a slug from the catalogue for the chosen formula, `random`, or an exact `REAL,IMAG` decimal pair. With no point option, that formula's default centre is used. |
| `--random-point` | Select a point at random from the chosen formula's catalogue. Equivalent to `--point random`. |
| `--random-seed N` | Seed random point selection so the same command chooses the same entry. Without it, system randomness is used. |
| `--list-points` | Print every catalogue slug and its stored safe depth, then exit. An audio file is not needed. |
| `--list-formulas` | Print the supported Mandelbrot-family formulas and exit. |
| `--formula NAME` (`mandelbrot`) | Select `mandelbrot`, `julia`, `burning-ship`, or `tricorn`. |
| `--julia-c REAL,IMAG` (`-0.8,0.156`) | Fixed Julia constant used when `--formula julia` is selected. |
| `--x-center VALUE` | Exact real coordinate. Must be paired with `--y-center`. |
| `--y-center VALUE` | Exact imaginary coordinate. Must be paired with `--x-center`. |
| `--base-zoom VALUE` (`1.0`) | Starting zoom. Decimal notation such as `1e0` is accepted. |
| `--max-zoom VALUE` (`1e32`) | Final zoom. Native scaled arithmetic accepts values below about `1e9800`; Mandelbrot catalogue points have individual safety limits, while alternate formula presets are exploratory. |
| `--allow-underspecified-center` | Bypass the decimal-place guard for an exploratory render. It can follow a different deep path from the intended target. |

### audio response

| argument | description |
| --- | --- |
| `--zoom-punch N` (`3.0`) | Contrast applied to loud instrumental events. Larger values make beats change the logarithmic zoom more strongly. |
| `--zoom-speed N` (`-0.04`) | Quiet-time velocity in log-zoom space. `0` removes the default quiet pullback; more-negative values pull back farther between events. |
| `--attack SECONDS` (`0.025`) | Attack time for the audio envelope. Smaller values react faster. |
| `--release SECONDS` (`0.12`) | Release time for the audio envelope. Larger values smooth the response. |
| `--beat-strength N` (`0`) | Add normalised spectral-onset strength to zoom speed. `0` preserves the original loudness-only motion; `1`–`1.5` is a noticeable setting. |

### quality and fractal rendering

| argument | description |
| --- | --- |
| `--render-scale N` (`1.0`) | Multiplier applied to keyframe source dimensions. Must be at least `1`. |
| `--fractal-scale N` (`1.0`) | Requested fractal source multiplier. Quality modes impose floors: atlas `draft`/`balanced` use at least `0.5`, `quality` at least `1`, and `extreme` at least `1.25`. |
| `--quality MODE` (`quality` with a resolution profile) | `draft` is fastest and permits labelled recovery; `balanced` is a strict output-resolution atlas; `quality` preserves native source resolution; `extreme` adds modest supersampling. |
| `--keyframe-factor N` (`2.0`) | Maximum zoom ratio between adjacent atlas levels. Larger values reduce the number of fields but enlarge crops more. |
| `--keyframe-mode MODE` (`atlas`) | `atlas` uses the fixed nested ladder; `legacy` uses the older audio-dependent full-field chunks. |
| `--iteration-base N` (`384`) | Minimum iteration budget for shallow frames. |
| `--iterations-per-decade N` (`500`) | Additional iteration budget per decade of zoom. |
| `--iteration-cap N` (`100000`) | Hard maximum iteration budget. Increase it for unusually slow interior points. |
| `--series-order N` (`3`) | Local BLA polynomial degree. Values `1`–`3` are effective; values through `32` are accepted for compatibility and clamp to the native range. |
| `--series-block N` (`256`) | Requested BLA block length, from `2` to `4096`. The native renderer applies its validated limits. |
| `--renderer MODE` (`auto`) | `auto` uses the native library when available; `native` requires it; `python` forces the Python fallback and is limited to roughly e300. |
| `--native-threads N` (`0`) | OpenMP worker count. `0` selects an automatic runtime/video setting. |
| `--native-backend MODE` (`auto`) | Native backend: `auto`, `scalar`, `avx2`, or `opencl`. OpenCL is an optional direct/shallow backend and is rejected for deep perturbation renders. |

### video encoding and colour

| argument | description |
| --- | --- |
| `--video-preset MODE` (`faster` with a resolution profile) | x264 speed/size preset. Hardware encoders map this to their own speed levels when supported. |
| `--codec NAME` (`auto`) | FFmpeg video encoder. `auto` probes NVENC, QSV, VAAPI, and VideoToolbox where applicable, then falls back to `libx264`. |
| `--crf N` (`10` with a resolution profile) | Quality value from `0` to `51`. CRF 10 is the near-lossless profile target; it is passed as CRF to software encoders and as the corresponding quality/QP control for supported hardware paths. |
| `--lossless` | Use lossless H.264 rate control where supported (`constqp/qp 0` for NVENC, CRF 0 for x264) and preserve 4:4:4 chroma where the encoder accepts it. |
| `--resample MODE` (`lanczos` with a resolution profile) | Crop and final upscale filter: `bilinear` is faster; `lanczos` is sharper and slower. |
| `--palette NAME` (`aurora`) | Colour palette: `aurora`, `fire`, `ocean`, `neon`, `sunset`, `mono`, `midnight`, `ember-night`, `terminal`, or `kalles-default`. The night themes use dark exteriors with white interiors; `kalles-default` matches the bundled Kalles Fraktaler profile in `palettes/kalles-default.kfp`. Its first exterior key is intentionally white and its separate interior colour is black, matching Kalles' defaults. |
| `--palette-file PATH` | Read at least two `#rrggbb` or `r g b` stops from a text file, or import a Kalles `.kfp` gradient and its colour settings. ordinary text palettes use the fast Aurora wave path; `.kfp` uses the Kalles-style transfer path. |
| `--glow N` (`0`) | Add a low-resolution bloom pass after colourisation, from `0` to `1`. It is off by default for the 10-minute target. |
| `--motion-blur N` (`0`) | Blend the current frame with the previous one, from `0` to below `1`. It is off by default. |
| `--encoder-threads N` (`0`) | FFmpeg encoder thread hint. `0` lets FFmpeg choose; a smaller value leaves more CPU for fractal rendering. Hardware encoders may ignore or reinterpret it. |

### caching and inspection

| argument | description |
| --- | --- |
| `--cache-dir PATH` | Store reusable scalar keyframes and audio-analysis data in this directory. Reusing the same settings and centre allows later runs to resume completed keyframes. |
| `--cache-limit-mb N` (`0`) | Maximum cache size in megabytes. `0` means unlimited. Eviction is incremental. |
| `--durable-cache` | Flush each cache tile before replacing its temporary file. Safer after power loss, but slower. |
| `--manifest PATH` | Write the render settings, selected point, formula, command, Git revision, timing, and status to JSON. Without this option the sidecar uses the output filename with a `.json` suffix. |
| `--no-manifest` | Disable the automatic JSON sidecar. |
| `--estimate` | Analyse the song and print source resolution/keyframe count without rendering the video. |

### safety and resource limits

public entry points validate dimensions before allocating fields (at most
100,000,000 pixels), decoded audio (at most 120,000,000 float32 samples),
video frames (at most 10,000,000), frame rate (1–1,000), sample rate
(1–768,000 Hz), and iteration budgets (1–10,000,000). Zoom exponents are
bounded to `10^-300` through `10^9800` for the native scaled path; the Python
fallback remains limited to approximately `10^300`. Coordinate text is capped
at 50,000 characters and each cache entry is bounded before NumPy opens it.

rendered videos, previews, manifests, and cache fields are written beside
their final targets and replaced atomically, so an interrupted operation does
not publish a partial result. Cache loading disables pickle and rejects
malformed, non-finite, oversized, or zip-bomb-like entries. FFmpeg and the
optional Demucs process drain only a bounded diagnostic tail and are started in
a private process group so cancellation also reaches their descendants.

when `--codec auto` is used, the advertised hardware encoders are filtered by
a short real encode probe. NVENC uses `-cq`, QSV uses `-global_quality`, VAAPI
uses `format=nv12,hwupload` with `-qp`, and VideoToolbox uses its mapped
`-q:v` scale. If the selected hardware path is unavailable or fails its
probe, the deterministic `libx264` software path is selected.

### reproducibility and fallback matrix

for reproducible output, keep the same Git revision, Python/native dependency
versions, command-line arguments, audio bytes, and exact centre coordinates.
the cache namespace includes the active field-renderer implementation, but
encoded bytes can still differ across FFmpeg builds, hardware encoders, CPU
instruction sets, and parallel floating-point runtimes. Compare decoded
frames or scalar fields when checking numerical output across machines.

| requested path | actual path | boundary |
| --- | --- | --- |
| `--renderer python` | Python direct/perturbed renderer | All formulas; exploratory fallback supports up to approximately `10^300`. |
| `--renderer auto` + shallow zoom | Native direct CPU renderer when available; Python fallback otherwise | All formulas; OpenCL is used only when explicitly selected. |
| `--renderer auto` + deep Mandelbrot | Native MPFR reference + scaled perturbation/BLA | Production path from approximately `10^12` through the validated catalogue depth. |
| `--renderer auto` + deep alternate formula | Python high-precision perturbation fallback | Julia, Burning Ship, and Tricorn use formula-specific boundary targets. |
| `--native-backend opencl` | Native OpenCL direct renderer | Mandelbrot-only shallow views; unavailable or deep paths fail clearly rather than silently changing numerical mode. |
| `--codec auto` | First passing hardware probe, otherwise `libx264` | The selected encoder is recorded in the manifest; hardware output is not byte-for-byte portable. |

## caching and reruns

keyframe cache entries include the centre, absolute zoom, dimensions, iteration
budget, renderer identity, and approximation settings. Old entries are not
silently reused after a numerical renderer change. Writes are atomic, so an
interrupted tile does not replace a completed tile.

the final video is assembled from the beginning on every run. This is required
for a valid compressed stream, even when most keyframes and the audio analysis
are already cached.

every normal render also writes a JSON manifest beside the output. It records
the command, Git revision, audio path, formula, Julia constant, resolved point,
zoom plan, settings, timings, and whether the render completed. Use
`--no-manifest` when a sidecar is not wanted.

## gui and project tools

gtk3 with PyGObject uses the desktop's configured GTK theme—including its
light/dark choice—on Linux and other platforms with GTK installed. The GUI is
intentionally a launcher rather than a second renderer, which keeps the command
line reproducible:

```sh
python3 gui.py
```

the gui also has a **live view** button for a screen-saver-like listening
preview. It prepares a small absolute-zoom source ladder, colourises smooth
parent/child crops at the display rate, and uses a streamed 8 kHz ffmpeg
energy envelope instead of the full export analysis. The ladder follows the
form's `base zoom` and `max zoom`, reaches the selected preview endpoint, and
then resets to the base view when the song loops. This keeps startup and cpu
use low while avoiding a stale shallow crop for a deep selection. The selected
range uses source fields about every three-quarters of a decade when it fits
the live budget (up to 168 fields for longer ranges). The first three fields
are prepared before playback while the rest render in the background. This keeps startup short
without making the live crop magnify a parent by thousands of times. press `esc`
to close it or `f11` to toggle fullscreen; the song loops until the window is
closed.
The standalone form is:

```sh
python3 live_view.py song.mp3 --formula mandelbrot --palette aurora
```

the live view is intentionally a preview: it caps the fractal source at
480×270 with the native backend (240×135 for the python fallback), then
upscales it to the fullscreen/window aspect with GTK. It uses the older fast,
factor-eight-style atlas behaviour rather than the strict quality-render path,
prepares at most 168 absolute-zoom fields, and caps the interactive preview at
e300. The initial fields are displayed immediately while the remaining ladder
is prepared in the background. Final renders still use the export pipeline and
its deep-zoom precision, atlas, and encoding settings. `ffplay` is used
for playback when installed; without it the visual preview remains usable.

the live view is independent of the export profiles. It keeps its own small
fast source ladder and only enlarges the RGB result in GTK; selecting `8k60`
for an export never makes the live window allocate an 8K surface.

the repository also includes `examples/make_test_tone.py` for a dependency-free
audio smoke test, `point_sheet.py` for catalogue previews, and
`make_preview.py` for GIF/MP4 exports. Generated audio, videos, GIFs, caches,
and native build products are ignored by Git.

## benchmarking

`benchmark.py` measures native work without decoding audio or encoding video.
the usual field probe is:

```sh
python3 benchmark.py --renderer native \
  --zoom-log 150 --reference-zoom-log 150 \
  --width 1920 --height 1080 --iterations 75384 \
  --threads 6 --repeat 1 --stats
```

the benchmark can also sweep the atlas or isolate the compositor:

```sh
python3 benchmark.py --stage atlas --renderer native \
  --zoom-log 150 --reference-zoom-log 150 \
  --width 512 --height 288 --iterations 75384 \
  --threads 6 --keyframe-factor 4 --stats

python3 benchmark.py --stage compositor --renderer native \
  --width 1920 --height 1080 --threads 6 --repeat 3
```

benchmark arguments:

| argument | description |
| --- | --- |
| `--stage` (`field`) | `field` renders one field, `atlas` renders every ladder level, and `compositor` measures the native colour/composite pass. |
| `--width`, `--height` (`256`) | Benchmark dimensions. |
| `--zoom` (`1e100`) | Decimal zoom for a field benchmark. |
| `--zoom-log` | Direct base-10 logarithm of the zoom; useful for exact atlas levels. |
| `--reference-zoom-log` | Reference/BLA setup depth. It must be at least the rendered depth. |
| `--iterations` (`20000`) | Iteration budget for the field. |
| `--x-center`, `--y-center` | Decimal benchmark centre; defaults to the bundled centre. |
| `--formula` (`mandelbrot`) | Formula to benchmark: `mandelbrot`, `julia`, `burning-ship`, or `tricorn`. |
| `--julia-c` (`-0.8,0.156`) | Fixed Julia constant when benchmarking the Julia formula. |
| `--renderer` (`auto`) | `auto`, `native`, or `python`. |
| `--backend` (`auto`) | `auto`, `scalar`, `avx2`, or `opencl`. OpenCL is shallow only. |
| `--threads` (`0`) | Native OpenMP worker count. |
| `--series-order` (`3`) | Local BLA degree. |
| `--disable-series` | Disable the validated image-wide series for comparison. |
| `--series-block` (`256`) | Requested BLA block length. |
| `--local-references` | Exercise adaptive secondary references for a deep field. |
| `--repeat` (`2`) | Number of repeated field/compositor timings. |
| `--keyframe-factor` (`2.0`) | Atlas spacing for `--stage atlas`. |
| `--iteration-base`, `--iterations-per-decade`, `--iteration-cap` | Match the visualizer’s iteration policy. |
| `--stats` | Collect native BLA, fallback, glitch, and timing counters. |
| `--verbose` | Print one line per atlas level. |
| `--json` | Emit the result as JSON. |

## deep-zoom notes

the deep-zoom design follows the standard reference-orbit and perturbation
approach described in [mathr’s deep-zoom
notes](https://mathr.co.uk/web/deep-zoom.html). The renderer keeps the MPFR
reference separate from the small per-pixel offsets, then uses scaled values
and BLA maps in the hot loop. The reference orbit is built once for a render;
additional depth tiers reuse its compact orbit and rebuild only the
radius-dependent approximation bounds.

the bundled centre contains 129 fractional decimal places. It is suitable for
the project’s tested e100 path, but it is intentionally rejected for an
unqualified e150 render. Use a catalogue point or provide the full-precision
centre from a zoom tool.

for further native details, see the public declarations in
[`renderer.h`](renderer.h) and the source-attributed catalogue in
[`deep_zoom_points.py`](deep_zoom_points.py).

## license

this project is released under the [MIT License](LICENSE).

## troubleshooting

- **Audio file not found:** pass the path as the first argument, for example
  `python3 visualizer.py "Music/track.mp3"`.
- **Native renderer unavailable:** run `nix-shell` followed by `make`. Deep
  renders cannot use the Python fallback beyond its supported range.
- **Centre precision error:** use `--list-points`, lower `--max-zoom`, or pass
  the full decimal coordinates. Only use `--allow-underspecified-center` when
  the exact deep target is not important.
- **Demucs error:** use `--separation none` for full-mix control, or install
  Demucs and keep `--separation auto`/`demucs`.
- **Hardware encoder error:** leave `--codec auto` enabled so the complete
  hardware path is probed and `libx264` is selected when necessary.
- **Alternate formula is slow at extreme depth:** the Mandelbrot e150+ path
  uses the validated native BLA implementation; Julia, Burning Ship, and
  Tricorn use a correctness-oriented Python perturbation fallback for deep
  boundary targets.
- **GUI will not start:** install GTK3 and PyGObject for the system Python, or
  enter the supplied `nix-shell`. Headless CLI renders do not need GUI
  dependencies.
