{ pkgs ? import <nixpkgs> {} }:

let
  python = pkgs.python3.withPackages (ps: with ps; [
    numpy
    librosa
    pillow
    mpmath
  ] ++ pkgs.lib.optional (ps ? demucs) ps.demucs);
in
pkgs.mkShell {
  packages = [
    python
    pkgs.ffmpeg
    pkgs.gcc
    pkgs.gnumake
    pkgs.git
    pkgs.pkg-config
    pkgs.gmp
    pkgs.mpfr
  ] ++ pkgs.lib.optional (pkgs ? gh) pkgs.gh;

  shellHook = ''
    echo "Fractal visualizer shell ready. Run: make && python visualizer.py song.mp3"
    echo "Run 'make test' for correctness checks and 'make benchmark' for native throughput."
    echo "Use --quality balanced for stable output, --quality quality for factor-sized keyframes."
    echo "Set OMP_NUM_THREADS and --native-threads to avoid CPU oversubscription."
    echo "Demucs is used automatically when available; --separation spectral enables proxies explicitly."
  '';
}
