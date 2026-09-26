{ pkgs ? import <nixpkgs> {} }:

let
  python = pkgs.python3.withPackages (ps: with ps; [
    numpy
    librosa
    pillow
    mpmath
    pygobject3
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
    pkgs.gtk3
    pkgs.gobject-introspection
  ]
  ++ pkgs.lib.optional (pkgs ? gh) pkgs.gh
  ++ pkgs.lib.optionals ((pkgs ? opencl-headers) && (pkgs ? ocl-icd)) [
    pkgs.opencl-headers
    pkgs.ocl-icd
  ];

  shellHook = ''
    # GTK's file chooser reads org.gtk.Settings.FileChooser when it opens.
    # The GTK package keeps that schema in its gsettings-schemas output, but
    # the development shell does not export the directory automatically.
    # Without this, clicking any GUI Browse button aborts with
    # "No GSettings schemas are installed".
    if [ -z "''${GSETTINGS_SCHEMA_DIR:-}" ]; then
      export GSETTINGS_SCHEMA_DIR="${pkgs.gtk3}/share/gsettings-schemas/gtk+3-${pkgs.gtk3.version}/glib-2.0/schemas"
    fi
    # This renderer is compiled locally for the current CPU.  Nix normally
    # strips -march=native to keep builds portable; opting in here is useful
    # for the intended single-machine 15 W rendering workflow.
    export NIX_ENFORCE_NO_NATIVE=0
    echo "Fractal visualizer shell ready. Run: make && python visualizer.py song.mp3"
    echo "Run 'make test' for correctness checks and 'make benchmark' for native throughput."
    echo "Atlas keyframes are the default; use --keyframe-mode legacy for regression comparisons."
    echo "Use --quality balanced for stable output, --quality extreme for modest supersampling."
    echo "Set OMP_NUM_THREADS and --native-threads to avoid CPU oversubscription."
    echo "OpenCL is optional; make advertises backend=2 only when a double-capable device is visible."
    echo "Demucs is used automatically when available; --separation spectral enables proxies explicitly."
  '';
}
