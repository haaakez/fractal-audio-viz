# fractal audio viz

turn a song into a full-length, music-driven fractal zoom. render from the gui
or command line; choose a fractal, palette, resolution, and quality.

[render preview](images/preview.GIF)

## start here

```sh
nix-shell                 # includes the build and gui dependencies
make                      # builds the native c++ renderer
python3 gui.py
```

choose a song in the window and press **render**. the audio field may be empty
at launch; audio is only needed when you start a render or live view.

for a terminal render:

```sh
python3 visualizer.py "song.mp3" --profile fhd60 --output renders/song.mp4
```

without nix, install `requirements.txt`, ffmpeg, a c++ compiler, gmp, and mpfr;
gtk 3 and pygobject are needed only for the gui. `make test` runs the test
suite. build `mandelbrot.so`/`.dll` for native deep rendering; python is a
slower fallback, practical to roughly `1e300`.

## what it does

- analyses the whole song; any local audio file supported by librosa works.
  loudness drives zoom, pitch adds colour movement, and onset strength can add
  beat response. the last frame reaches the requested zoom; quiet-time motion
  is tunable. `--separation auto` tries demucs and falls back to the full mix;
  `spectral` uses frequency-band proxies, and `none` uses the full mix.
- plans a logarithmic camera path from `--base-zoom` to `--max-zoom` and builds
  nested atlas keyframes instead of calculating every video frame from scratch.
- renders mandelbrot, julia, burning ship, or tricorn. the native c++ renderer
  uses mpfr reference orbits and perturbation/bla for deep views; cpu, avx2,
  and compatible opencl gpu paths are available.
- colours and assembles the atlas, then encodes the complete song with audio.
  ordinary palettes use the native colour path; imported `.kfp` files use the
  kalles 1024-entry cyclic palette, sine interpolation, distance/slope shading,
  dithering, and optional 3d relief and glitch highlights. atlas fields are
  joined before `.kfp` shading to avoid tile seams. `.kfp` channel order matches
  kalles' preview. resized `.kfp` output uses lanczos; ordinary output
  defaults to nearest-neighbour. glow and motion blur are optional.

## render settings

the default profile is `4k60`. all canonical profiles are 60 fps, zoom to
`1e100`, use balanced output-density atlas rendering, and target crf 10.
crf 10 is near-lossless, not mathematically lossless; add `--lossless` for
lossless h.264 where supported.

| profile | output size |
| --- | ---: |
| `sd60` | 720×480 |
| `hd60` | 1280×720 |
| `fhd60` | 1920×1080 |
| `2k60` | 2560×1440 |
| `4k60` | 3840×2160 |
| `8k60` | 7680×4320 |

the standard source is full output density. for a faster quarter-size source,
use `--source-mode upscaled` (or `--upscaling`). `native` also forces
output-density fields; `lossless-compressed` is the default fused native
pipeline, not an uncompressed video setting. `draft` favours speed, `balanced`
is the profile default, `quality` ensures full density, and `extreme` renders
at least 1.25× density. `--fractal-scale` and `--render-scale` set source
density; a larger `--keyframe-factor` uses fewer atlas fields with larger
zooms between them. options after `--profile` override its defaults. legacy
profile aliases `preview`, `fullhd`, `1080p`, and `beat` remain accepted.

for example, select a kalles palette and preserve its relief/glitch look:

```sh
python3 visualizer.py "song.mp3" --profile fhd60 \
  --palette-file palettes/my-palette.kfp --kfp-3d --kfp-glitches \
  --output renders/kalles.mp4
```

## command reference

run `python3 visualizer.py --help` for defaults and validation. use
`--list-profiles`, `--list-formulas`, or `--list-points` to inspect built-ins.
palettes are `aurora`, `fire`, `ocean`, `neon`, `sunset`, `mono`, `midnight`,
`ember-night`, `terminal`, and `kalles-default`. text palette files accept at
least two `#rrggbb` or `r g b` colour stops; `.kfp` imports its own colour
settings.

| purpose | options |
| --- | --- |
| input/output | optional `audio` (otherwise finds `song.mp3` beside the app); `--output`; `--profile`; `--width`; `--height`; `--fps`; `--sample-rate` |
| audio | `--separation auto\|demucs\|spectral\|none`; `--zoom-punch`; `--zoom-speed`; `--attack`; `--release`; `--beat-strength` |
| formula/point | `--formula mandelbrot\|julia\|burning-ship\|tricorn`; `--julia-c real,imag`; `--point slug\|random\|real,imag`; `--random-point`; `--random-seed`; `--x-center` + `--y-center` |
| camera | `--base-zoom`; `--max-zoom`; `--allow-underspecified-center` (exploratory only) |
| quality/atlas | `--source-mode lossless-compressed\|native\|upscaled`; `--upscaling`; `--quality`; `--fractal-scale`; `--render-scale`; `--keyframe-factor`; `--keyframe-mode atlas\|legacy`; `--iteration-base`; `--iterations-per-decade`; `--iteration-cap` |
| renderer | `--renderer auto\|native\|python`; `--native-backend auto\|scalar\|avx2\|opencl`; `--native-threads`; `--series-order`; `--series-block` |
| colour | `--palette`; `--palette-file` (text stops or `.kfp`); `--kfp-3d` / `--no-kfp-3d`; `--kfp-glitches` / `--no-kfp-glitches`; `--resample nearest\|bilinear\|lanczos`; `--glow`; `--motion-blur` |
| encoding | `--codec auto\|encoder`; `--video-preset`; `--crf`; `--lossless`; `--encoder-threads` |
| resume/inspect | `--cache-dir`; `--cache-limit-mb`; `--durable-cache`; `--manifest`; `--no-manifest`; `--estimate` |

`auto` renderer/backend uses the native library and selects a compatible cpu or
gpu path when available; `--native-backend opencl` explicitly requests opencl
(which needs a supported device, driver, and build headers). opencl accelerates
compatible scalar `.kfp` profiles; features needing orbit, texture, or
multi-colour metadata use the complete cpu path. native threads default to a
maximum of six while reserving two logical cpus; encoder threads default to two.
set `--native-threads N` or `--encoder-threads N` to choose a limit.
`--codec auto` probes available hardware encoders, then falls back to `libx264`.

## deep zoom and repeatability

use `--list-points --formula formula` to find catalogue points, or pass an exact
centre with `--point=real,imag` / `--x-center` and `--y-center`. julia points
may include their own constant. for deep custom zooms, provide at least
`ceil(log10(max-zoom)) + 16` fractional digits in both coordinates;
`--allow-underspecified-center` can instead render a different nearby target.
native scaled arithmetic supports zoom exponents to about `1e9800`; the python
fallback is practical to roughly `1e300`. catalogue points have individual
safe depths; mandelbrot uses mpfr reference/perturbation and bla maps, while
the other formulas use matching formula-specific references.

use the same revision, dependencies, command, audio, and exact centre for
repeatable fields. encoded bytes can still differ with ffmpeg, hardware
encoders, and thread counts. cache entries are keyed by render settings and
written atomically; cached fields/audio are reused, but each run assembles a
new complete video from the beginning. a json manifest is written beside the
video by default. `--no-manifest` disables it. output sizes, coordinates,
iterations, and other resource-heavy inputs are validated; cancelled ffmpeg
and demucs child processes are cleaned up.

## gui, live view, and tools

`python3 gui.py` launches the gtk desktop app; it follows the system gtk theme,
previews the selected palette, runs the same renderer as the cli, and prints a
copyable command in its log. audio can be selected after launch. the live view
is also available from the gui or directly:

```sh
python3 live_view.py "song.mp3" --formula mandelbrot --palette aurora
```

live view renders an 854×480 source, prepares source coverage for the first
60% of the song before playback, then fills the rest in the background. it
holds the last complete image while a field renders, caps the atlas at 224
fields and zoom at `1e300`, and is independent of export resolution. `esc`
closes it; `f11` toggles fullscreen; `ffplay` provides audio when installed.

- `python3 make_preview.py` makes a gif from the latest render; options include
  input/directory, `--format`, `--start`, `--duration`, `--width`, `--fps`,
  `--with-audio`, and `--output`.
- `python3 point_sheet.py --output renders/points.png` previews deep points.
- `python3 benchmark.py --help` lists field, atlas, and compositor benchmarks;
  options cover stage, size, zoom/reference depth, iterations, centre/formula,
  renderer/backend, threads, series settings, local references, repeats,
  atlas factor, iteration policy, stats, verbosity, and json output.
- `examples/make_test_tone.py` creates a dependency-free test song.

## release bundles

```sh
make package-linux       # portable x86_64 .tar.gz; python/ffmpeg/gtk remain system deps
make package-windows     # windows source/runtime .zip (windows/msys2)
make package-windows-exe # standalone windows gui/cli .zip (windows build environment)
```

the linux archive still needs python, `requirements.txt`, and ffmpeg; use
`run_gui.sh`, `render.sh`, or `live-view.sh` after unpacking. the windows
source archive has matching `.bat` launchers and needs python/ffmpeg/gtk. the
standalone windows archive opens the same gui as `gui.py`, includes python,
gtk, the native renderer, and ffmpeg, lets you choose audio after launch, and
has `render.bat` for terminal jobs. github actions builds the archives on a
manual run or a `v*` tag.

## files and help

`visualizer.py` is the cli and render pipeline; `renderer.cpp`/`renderer.h`
contain the native engine; `gui.py` and `live_view.py` are the desktop and
preview apps; `profiles.py`, `deep_zoom_points.py`, and `palettes/` hold presets,
points, and palettes. `benchmark.py`, `make_preview.py`, and `point_sheet.py`
are helper tools. `Makefile` and `shell.nix` build the project; `tests/` has the
test suite. see [`renderer.h`](renderer.h) and
[`deep_zoom_points.py`](deep_zoom_points.py) for native details and point
sources. deep-zoom background: [mathr's notes](https://mathr.co.uk/web/deep-zoom.html).

licensed under the [mit license](LICENSE).
