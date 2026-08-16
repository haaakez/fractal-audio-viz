# Mandelbrot music visualizer

This project renders an audio-reactive Mandelbrot zoom locally. The Python
front end handles audio, animation and FFmpeg; the C++ C-ABI backend handles
deep numerical rendering, MPFR reference orbits, perturbation, BLA maps,
OpenMP and the fused atlas/composite path.

## Build

Inside the supplied Nix shell:

```sh
nix-shell
make
make test
```

`make` builds `mandelbrot.so`. MPFR/GMP are required for native deep zooms.
The supplied shell opts the local build into `-O3`, `-march=native`, link-time
optimization and `-fno-math-errno` for the current CPU; the Python fallback
remains useful for shallow tests and previews.

## Render

```sh
python3 visualizer.py song.mp3 \
  --output fractal_viz.mp4 \
  --width 1920 --height 1080 --fps 30 \
  --max-zoom 1e100 \
  --quality quality \
  --cache-dir keyframes
```

Resolution and frame rate are controlled directly by `--width`, `--height`
and `--fps`. The default zoom response has a small negative quiet-time speed,
so the camera eases backward between strong instrumental hits; use
`--zoom-speed 0` for a strictly forward path or a more negative value for
larger pullbacks. `--zoom-punch` controls how strongly loud events overcome
that bias. Attack and release are configurable with `--attack` and
`--release` (seconds).

Quality modes:

- `draft`: fastest preview; may undersample source keyframes.
- `balanced`: one output-resolution tile per factor-two zoom level.
- `quality`: the same nested atlas with the requested source scale preserved.
- `extreme`: modest additional source supersampling.

The default `--keyframe-mode atlas` renders a fixed logarithmic ladder of
nested tiles. Each frame is composed from two adjacent tiles, so the centre
keeps native resolution while the outer region is reused from the parent.
The ladder is independent of audio timing and its absolute tiles can be
reused by later renders with the same centre, scale and numerical settings.
Only the parent/current/child window is retained in memory, including when
the audio path briefly zooms backwards.
Use `--keyframe-mode legacy` only for visual regression against the previous
full-field chunk renderer.

For a 500×500 preview, `--quality quality --keyframe-factor 2` renders
approximately 1000×1000 source fields. This costs more memory and fractal
work, but prevents the repeated 4×/8× enlargement that causes visible
keyframe seams.

The default `aurora` colour path keeps the original two-anchor blue/yellow
gradient. The song's robust average pitch leaves those hues unchanged; lower
and higher notes rotate both anchors in opposite hue directions. Vocals
modulate the gradient, while instrumental energy owns the zoom. When separation
confidence is insufficient, the full mix drives both controls as a deliberate
fallback.

Audio separation uses Demucs when available in `--separation auto`. If Demucs
cannot produce reliable stems, the full mix drives both zoom and colour. Use
`--separation spectral` only when frequency-band proxy controls are desired.

Available palettes are `aurora`, `fire`, `ocean`, `neon`, `sunset` and `mono`.
`aurora` uses the native fused colour path; custom palettes use the cached
Python palette path.

## Benchmark

The benchmark does not decode audio or encode video. Field benchmarks report
reference/BLA setup separately from repeated pixel-render timings:

```sh
python3 benchmark.py --renderer native --zoom 1e100 \
  --width 256 --height 256 --iterations 20000 --series-block 256 --repeat 2
```

It reports deep field-render time, pixels per second, reference setup time,
and a one-shot total including that setup. To measure the separate
output-stage bottleneck, use:

For an exact atlas-level probe, use a reference matching the final zoom:

```sh
python3 benchmark.py --renderer native --zoom-log 64.721449 \
  --reference-zoom-log 100 --width 960 --height 540 \
  --iterations 32895 --series-block 256 --threads 6 --repeat 1 --stats
```

Add `--local-references` to exercise the production deep-tile fallback used
by the atlas renderer; without it this command deliberately measures one
global reference only.

```sh
python3 benchmark.py --stage compositor --renderer native \
  --width 1920 --height 1080 --threads 6 --repeat 3
```

The one-field benchmark is not representative of a complete nested zoom: the
deep renderer has different BLA tiers at different depths. Sweep every atlas
level without writing cache files or encoding video with:

```sh
python3 benchmark.py --stage atlas --renderer native \
  --width 128 --height 128 --zoom 1e100 --threads 6 \
  --stats --json > atlas-benchmark.json
```

This reports total, median, p95, and worst tile time. `--stats` adds native
BLA/fallback counters and is intended for diagnosis; omit it for the least
perturbed timing. An e100 factor-two sweep contains 334 source levels, so use
a small resolution while locating slow depth bands.

Pass `--x-center` and `--y-center` to benchmark a different deep-zoom target;
the defaults use the bundled centre.

The video renderer also prints separate tile and frame/crop/pipe timings so
fractal work and encoding backpressure can be distinguished.

Use `--estimate` with the normal visualizer command to analyse the song and
print the source resolution and keyframe count without rendering. Video
encoding can be tuned with `--codec`, `--crf` and `--video-preset`; the default
`libx264 --preset ultrafast --crf 18` favours throughput on low-power CPUs.

## Numerical design

Deep fields use a reusable high-precision reference orbit. Pixel offsets use
perturbation arithmetic; the native BLA hierarchy applies polynomial maps
through degree three and replays blocks that approach the escape boundary.
The `--series-order` option selects degree one through three. When a deep
perturbation is far below the BLA radius, the native path automatically uses
the mathematically equivalent linear branch; it restores the requested
higher-order series near the escape boundary. A conservative scaled Brent
cycle check terminates settled interior pixels without mistaking ordinary
boundary transients for interiors. Degree-three maps remain capped at 256
iterations per map. Long linear BLA blocks can be faster at benign locations
but can amplify perturbation glitches near the bundled deep-zoom centre.
`--series-block` defaults to the conservative 256-step tier for video
rendering; larger values remain available for controlled benchmarks and
known-stable locations.

At log10 zooms of 40 and deeper, sufficiently large native atlas tiles use an
adaptive Kalles-style secondary-reference fallback: the source tile starts as
a small local grid and only a cell that exceeds a short native deadline is
split again, up to a 32x32 grid. Each view gets a high-precision reference at
its own centre. This prevents one narrow near-parabolic feature from making
the whole tile fall back to a pathological tail while keeping easy tiles
cheap. A final small-cell retry is bounded, and an atlas tile deadline
prevents one unresolved feature from stopping the complete video. Pixels that
still miss both limits are recovered from the parent's central world-space
crop, with neighbourhood interpolation only as a last resort; they are never
replaced by one constant iteration value that would create a rectangular
colour block. The split also handles odd source sizes such as the 125x125 tile
produced by a 500x500 render with `--fractal-scale 0.25`; the cache namespace
is versioned so older single-reference atlas fields are not reused.

Keyframe fields are cached by renderer identity, absolute zoom, dimensions,
centre, iteration budget and approximation settings. Cache writes and final
video writes are atomic, so interrupted renders do not replace a completed
output with a partial file. Normal cache writes do not wait on storage flushes;
use `--durable-cache` when surviving a sudden power loss matters more than
throughput. `--cache-limit-mb` uses incremental LRU eviction; with its default
value of zero, cache growth remains intentionally unlimited.

Re-running the same command with the same cache directory resumes at the
completed absolute keyframes and reuses the audio analysis. The final video
is still assembled from the beginning, which keeps FFmpeg output valid and
avoids pretending that an interrupted compressed stream can be safely
concatenated.

For a local 500×500/60 FPS test:

```sh
python3 visualizer.py song.mp3 --width 500 --height 500 --fps 60 \
  --max-zoom 1e100 --quality quality --renderer native \
  --native-threads 6 --encoder-threads 2 --cache-dir keyframes
```

Use `--cache-limit-mb 4096` for large quality masters when disk usage should
be bounded. The default `0` retains all valid entries for maximum resume reuse.
