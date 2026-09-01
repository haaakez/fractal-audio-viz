# Mandelbrot music visualizer

This is a command-line renderer for audio-reactive Mandelbrot zooms. It reads
one audio file, turns the audio into animation controls, renders a sequence of
deep-zoom source images, and sends the final RGB frames to FFmpeg.

## How it works

The render is split into five stages.

1. **Audio analysis.** The input is loaded as mono audio and resampled to
   `--sample-rate`. A frame-aligned loudness envelope drives the camera. YIN
   pitch tracking controls the colour movement. With `--separation auto`, the
   program tries Demucs first; if no usable vocal/instrumental split is
   available, the whole song drives both controls. `--separation spectral`
   enables frequency-band proxies instead.

2. **Zoom planning.** The camera moves in logarithmic zoom space from
   `--base-zoom` to `--max-zoom`. Instrumental energy changes the speed. The
   default `--zoom-speed -0.04` allows small pullbacks during quiet passages,
   while the final frame is still placed exactly at the requested maximum.

3. **Source images.** The default `atlas` mode builds a fixed ladder of nested
   keyframes. A keyframe covers one `--keyframe-factor` interval. Frames between
   keyframes are centred crops of the source images, so the expensive fractal
   calculation is not repeated for every video frame. The older full-field
   chunk renderer remains available as `--keyframe-mode legacy`.

4. **Mandelbrot rendering.** Shallow views use the direct native renderer. At
   deep zooms, the C++ backend builds a high-precision MPFR reference orbit and
   renders pixel offsets with perturbation arithmetic. Scaled mantissa/exponent
   values preserve tiny offsets at e150 and beyond. BLA maps, an adaptive
   image-wide series, OpenMP, and depth-specific reference tiers reduce the
   per-pixel cost. Large difficult tiles can be split around local secondary
   references rather than making the whole tile fall back to a slow path.

5. **Colour and encoding.** Keyframes store scalar iteration data. The native
   colour path or Pillow turns that data into RGB after cropping, and FFmpeg
   writes the video with the original audio. `--codec auto` probes available
   hardware encoders and falls back to `libx264`.

The output dimensions are controlled by `--width` and `--height`. The fractal
source can be smaller or larger than the output: `--render-scale` multiplies
the source size, while `--fractal-scale` sets the requested quality floor.
For example, a 3840×2160 output with `--fractal-scale 0.5` renders 1920×1080
source fields and still produces a 4K video.

### Repository layout

| File | Purpose |
| --- | --- |
| `visualizer.py` | Audio analysis, zoom planning, keyframes, colour, and FFmpeg. |
| `renderer.cpp` / `renderer.h` | Native C ABI, MPFR deep rendering, BLA, OpenMP, and colour paths. |
| `deep_zoom_points.py` | Source-attributed deep-zoom catalogue and exact decimal centres. |
| `benchmark.py` | Field, compositor, and atlas benchmarks without audio or video encoding. |
| `tests/` | Python and native correctness tests. |
| `shell.nix` / `Makefile` | Reproducible dependencies and the native build. |

## Installation and build

The supplied Nix shell provides Python, NumPy, librosa, Pillow, mpmath,
FFmpeg, GCC, GMP, and MPFR. OpenCL support is enabled when the host has the
headers and ICD available.

```sh
nix-shell
make
make test
```

`make` creates `mandelbrot.so`. GMP and MPFR are required for native deep
zooms. The Python renderer is useful for shallow previews and tests, but it is
not a replacement for the native e150+ path.

## Running a render

The input song is the first positional argument. It is not fixed to
`song.mp3`; pass any local file that librosa can decode, such as MP3, WAV, or
FLAC. If the argument is omitted, the default is `song.mp3`.

```sh
python3 visualizer.py path/to/my-song.mp3 \
  --output renders/my-song.mp4
```

Paths containing spaces should be quoted:

```sh
python3 visualizer.py "Music/Live set.flac" \
  --output "renders/Live set.mp4"
```

The program creates the output directory when needed. Use `python3
visualizer.py --help` for the parser’s built-in summary.

## Constructing a command

Start with this shape:

```sh
python3 visualizer.py AUDIO --output OUTPUT [options]
```

Then choose the parts that matter for the render:

1. Set the input song and output file.
2. Set `--width`, `--height`, and `--fps`.
3. Choose a centre with `--point`, `--random-point`, or `--x-center` plus
   `--y-center`.
4. Set `--max-zoom` and, for a deep custom point, provide enough decimal
   digits in both coordinates.
5. Choose a quality/speed trade-off with `--quality`, `--fractal-scale`, and
   `--keyframe-factor`.
6. Add `--cache-dir` if the render may be resumed or repeated.

### Common examples

A normal 1080p render using a named e150-capable point:

```sh
python3 visualizer.py music/track.mp3 \
  --output renders/track.mp4 \
  --width 1920 --height 1080 --fps 30 \
  --point oldwooddish --max-zoom 1e150 \
  --cache-dir cache/track
```

A reproducible random point:

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

List the catalogue before choosing:

```sh
python3 visualizer.py --list-points
```

The catalogue is built from named KFR files in the [MDZ
gallery](https://mathr.co.uk/mdz/gallery/) and deep test views in
[FractalShark](https://github.com/mattsaccount364/FractalShark). It currently
contains 24 entries, including labelled exact conjugate mirrors. Random
selection only uses entries whose stored coordinates and renderer screening
cover the requested `--max-zoom`.

To use your own centre, pass a comma-separated pair to `--point`:

```sh
python3 visualizer.py music/track.mp3 \
  --point=-0.743643887037151,0.131825904205330 \
  --max-zoom 1e12 --allow-underspecified-center
```

For a deep render, replace that short pair with the full decimal export from
your zoom tool. The production guard requires at least
`ceil(log10(max-zoom)) + 16` fractional decimal places in both coordinates.
This prevents a short decimal from silently producing a different deep
target. `--allow-underspecified-center` is for exploratory renders only.

The two-coordinate form is also available for scripts and Kalles exports:

```sh
python3 visualizer.py music/track.mp3 \
  --x-center=-0.743643887037151000000000000000000000000000000000 \
  --y-center=0.131825904205330000000000000000000000000000000000 \
  --max-zoom 1e32
```

Use either `--point` or the `--x-center`/`--y-center` pair, not both. The
`--point=...` and `--x-center=...` forms are convenient when a negative value
would otherwise be mistaken for another command-line option.

### 4K/e150 speed profile

This profile writes 4K output while keeping the fractal source at 2K. It is a
reasonable starting point for the ten-minute target on a six-core low-power
CPU; actual time depends on the song, selected point, and encoder.

```sh
python3 visualizer.py music/track.mp3 \
  --output renders/track-4k-e150.mp4 \
  --width 3840 --height 2160 --fps 30 \
  --point random --random-seed 42 --max-zoom 1e150 \
  --quality balanced --fractal-scale 0.5 \
  --keyframe-factor 4 --codec auto \
  --cache-dir cache/track-4k-e150
```

For a native-density master, use `--fractal-scale 1 --keyframe-factor 2`.
That produces about 499 source levels at e150 and is substantially slower.
The speed profile uses about 251 levels and 1920×1080 source fields.

Use `--estimate` to analyse the song and print the planned source resolution
and keyframe count without rendering video:

```sh
python3 visualizer.py music/track.mp3 \
  --width 3840 --height 2160 --quality balanced \
  --fractal-scale 0.5 --keyframe-factor 4 \
  --max-zoom 1e150 --estimate
```

## `visualizer.py` argument reference

Unless noted otherwise, the values in parentheses are the defaults.

### Input, output, and audio

| Argument | Description |
| --- | --- |
| `audio` (`song.mp3`) | Positional input audio file. This is how you choose a custom song. |
| `--output PATH` (`fractal_viz.mp4`) | Output video path. |
| `--width N` (`1920`) | Output width in pixels. |
| `--height N` (`1080`) | Output height in pixels. |
| `--fps N` (`30`) | Output frame rate and audio-analysis frame rate. |
| `--sample-rate HZ` (`44100`) | Sample rate used while loading and analysing audio. |
| `--separation MODE` (`auto`) | `auto` tries Demucs and falls back to the full mix; `demucs` requires Demucs; `spectral` uses frequency-band proxies; `none` uses the full mix. |

### Centre and zoom

| Argument | Description |
| --- | --- |
| `--point VALUE` | Select a catalogue slug, `random`, or an exact `REAL,IMAG` decimal pair. With no point option, the bundled centre is used. |
| `--random-point` | Select a safe catalogue point at random. Equivalent to `--point random`. |
| `--random-seed N` | Seed random point selection so the same command chooses the same entry. Without it, system randomness is used. |
| `--list-points` | Print every catalogue slug and its stored safe depth, then exit. An audio file is not needed. |
| `--x-center VALUE` | Exact real coordinate. Must be paired with `--y-center`. |
| `--y-center VALUE` | Exact imaginary coordinate. Must be paired with `--x-center`. |
| `--base-zoom VALUE` (`1.0`) | Starting zoom. Decimal notation such as `1e0` is accepted. |
| `--max-zoom VALUE` (`1e32`) | Final zoom. Native scaled arithmetic accepts values below about `1e9800`, but the centre must have enough digits and each catalogue point has its own safe limit. |
| `--allow-underspecified-center` | Bypass the decimal-place guard for an exploratory render. It can follow a different deep path from the intended target. |

### Audio response

| Argument | Description |
| --- | --- |
| `--zoom-punch N` (`3.0`) | Contrast applied to loud instrumental events. Larger values make beats change the logarithmic zoom more strongly. |
| `--zoom-speed N` (`-0.04`) | Quiet-time velocity in log-zoom space. `0` removes the default quiet pullback; more-negative values pull back farther between events. |
| `--attack SECONDS` (`0.025`) | Attack time for the audio envelope. Smaller values react faster. |
| `--release SECONDS` (`0.12`) | Release time for the audio envelope. Larger values smooth the response. |

### Quality and fractal rendering

| Argument | Description |
| --- | --- |
| `--render-scale N` (`1.0`) | Multiplier applied to keyframe source dimensions. Must be at least `1`. |
| `--fractal-scale N` (`1.0`) | Requested fractal source multiplier. Quality modes impose floors: atlas `draft`/`balanced` use at least `0.5`, `quality` at least `1`, and `extreme` at least `1.25`. |
| `--quality MODE` (`balanced`) | `draft` is fastest and permits labelled recovery; `balanced` is a strict output-resolution atlas; `quality` preserves native source resolution; `extreme` adds modest supersampling. |
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

### Video encoding and colour

| Argument | Description |
| --- | --- |
| `--video-preset MODE` (`ultrafast`) | x264 speed/size preset. Hardware encoders map this to their own speed levels when supported. |
| `--codec NAME` (`auto`) | FFmpeg video encoder. `auto` probes NVENC, QSV, VAAPI, and VideoToolbox where applicable, then falls back to `libx264`. |
| `--crf N` (`18`) | Quality value from `0` to `51`. It is passed as CRF to software encoders and as the corresponding quality/QP control for supported hardware paths. |
| `--resample MODE` (`bilinear`) | Crop filter: `bilinear` is faster; `lanczos` gives a smoother, slower resize. |
| `--palette NAME` (`aurora`) | Colour palette: `aurora`, `fire`, `ocean`, `neon`, `sunset`, or `mono`. `aurora` uses the fused native colour path. |
| `--encoder-threads N` (`0`) | FFmpeg encoder thread hint. `0` lets FFmpeg choose; a smaller value leaves more CPU for fractal rendering. Hardware encoders may ignore or reinterpret it. |

### Caching and inspection

| Argument | Description |
| --- | --- |
| `--cache-dir PATH` | Store reusable scalar keyframes and audio-analysis data in this directory. Reusing the same settings and centre allows later runs to resume completed keyframes. |
| `--cache-limit-mb N` (`0`) | Maximum cache size in megabytes. `0` means unlimited. Eviction is incremental. |
| `--durable-cache` | Flush each cache tile before replacing its temporary file. Safer after power loss, but slower. |
| `--estimate` | Analyse the song and print source resolution/keyframe count without rendering the video. |

## Caching and reruns

Keyframe cache entries include the centre, absolute zoom, dimensions, iteration
budget, renderer identity, and approximation settings. Old entries are not
silently reused after a numerical renderer change. Writes are atomic, so an
interrupted tile does not replace a completed tile.

The final video is assembled from the beginning on every run. This is required
for a valid compressed stream, even when most keyframes and the audio analysis
are already cached.

## Benchmarking

`benchmark.py` measures native work without decoding audio or encoding video.
The usual field probe is:

```sh
python3 benchmark.py --renderer native \
  --zoom-log 150 --reference-zoom-log 150 \
  --width 1920 --height 1080 --iterations 75384 \
  --threads 6 --repeat 1 --stats
```

The benchmark can also sweep the atlas or isolate the compositor:

```sh
python3 benchmark.py --stage atlas --renderer native \
  --zoom-log 150 --reference-zoom-log 150 \
  --width 512 --height 288 --iterations 75384 \
  --threads 6 --keyframe-factor 4 --stats

python3 benchmark.py --stage compositor --renderer native \
  --width 1920 --height 1080 --threads 6 --repeat 3
```

Benchmark arguments:

| Argument | Description |
| --- | --- |
| `--stage` (`field`) | `field` renders one field, `atlas` renders every ladder level, and `compositor` measures the native colour/composite pass. |
| `--width`, `--height` (`256`) | Benchmark dimensions. |
| `--zoom` (`1e100`) | Decimal zoom for a field benchmark. |
| `--zoom-log` | Direct base-10 logarithm of the zoom; useful for exact atlas levels. |
| `--reference-zoom-log` | Reference/BLA setup depth. It must be at least the rendered depth. |
| `--iterations` (`20000`) | Iteration budget for the field. |
| `--x-center`, `--y-center` | Decimal benchmark centre; defaults to the bundled centre. |
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

## Deep-zoom notes

The deep-zoom design follows the standard reference-orbit and perturbation
approach described in [mathr’s deep-zoom
notes](https://mathr.co.uk/web/deep-zoom.html). The renderer keeps the MPFR
reference separate from the small per-pixel offsets, then uses scaled values
and BLA maps in the hot loop. The reference orbit is built once for a render;
additional depth tiers reuse its compact orbit and rebuild only the
radius-dependent approximation bounds.

The bundled centre contains 129 fractional decimal places. It is suitable for
the project’s tested e100 path, but it is intentionally rejected for an
unqualified e150 render. Use a catalogue point or provide the full-precision
centre from a zoom tool.

For further native details, see the public declarations in
[`renderer.h`](renderer.h) and the source-attributed catalogue in
[`deep_zoom_points.py`](deep_zoom_points.py).

## Troubleshooting

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
