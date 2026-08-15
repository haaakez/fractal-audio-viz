# Mandelbrot music visualizer

This project renders an audio-reactive Mandelbrot zoom locally. The Python
front end handles audio, animation and FFmpeg; the C++ C-ABI backend handles
deep numerical rendering, MPFR reference orbits, perturbation, BLA maps,
OpenMP and the fused crop/colour path.

## Build

Inside the supplied Nix shell:

```sh
nix-shell
make
make test
```

`make` builds `mandelbrot.so`. MPFR/GMP are required for native deep zooms.
The Python fallback remains useful for shallow tests and previews.

## Render

```sh
python3 visualizer.py song.mp3 \
  --output fractal_viz.mp4 \
  --width 1920 --height 1080 --fps 30 \
  --max-zoom 1e100 \
  --quality balanced \
  --cache-dir keyframes
```

Quality modes:

- `draft`: fastest preview; may undersample source keyframes.
- `balanced`: output-resolution source fields and transition blending.
- `quality`: source fields are sized for the complete factor-two crop.
- `extreme`: additional source supersampling and longer blending.

For a 500×500 preview, `--quality quality --keyframe-factor 2` renders
approximately 1000×1000 source fields. This costs more memory and fractal
work, but prevents the repeated 4×/8× enlargement that causes visible
keyframe seams.

Audio separation uses Demucs when available in `--separation auto`. If Demucs
cannot produce reliable stems, the full mix drives both zoom and colour. Use
`--separation spectral` only when frequency-band proxy controls are desired.

Available palettes are `aurora`, `fire`, `ocean`, `neon`, `sunset` and `mono`.
`aurora` uses the native fused colour path; custom palettes use the cached
Python palette path.

## Benchmark

The benchmark does not decode audio or encode video:

```sh
python3 benchmark.py --renderer native --zoom 1e100 \
  --width 256 --height 256 --iterations 20000
```

It reports field-render time and pixels per second. The video renderer also
prints separate keyframe and crop/colour/pipe timings so fractal work and
encoding backpressure can be distinguished.

## Numerical design

Deep fields use a reusable high-precision reference orbit. Pixel offsets use
perturbation arithmetic; the native BLA hierarchy applies polynomial maps
through degree three and replays blocks that approach the escape boundary.
The `--series-order` option selects degree one through three. The ordinary
double tail is entered only after the perturbation is large enough that its
relative precision is safe.

Keyframe fields are cached by renderer identity, absolute zoom, dimensions,
centre, iteration budget and approximation settings. Cache writes and final
video writes are atomic, so interrupted renders do not replace a completed
output with a partial file.

Use `--cache-limit-mb 4096` for large quality masters when disk usage should
be bounded. The default `0` retains all valid entries for maximum resume reuse.
