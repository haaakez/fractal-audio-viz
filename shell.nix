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
    pkgs.pkg-config
    pkgs.gmp
    pkgs.mpfr
  ];

  shellHook = ''
    echo "Fractal visualizer shell ready. Run: make && python visualizer.py song.mp3"
    echo "OpenMP uses all available CPUs by default; set OMP_NUM_THREADS to limit it."
    echo "Optional vocal separation: install Demucs in this environment and use --separation auto."
  '';
}
