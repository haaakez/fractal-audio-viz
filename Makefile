CXX ?= g++
PKG_CONFIG ?= pkg-config
ifeq ($(OS),Windows_NT)
  PYTHON ?= python
else
  PYTHON ?= python3
endif
DIST_DIR ?= dist
WINDOWS_RUNTIME_DIR ?= .release-runtime
FFMPEG_DIR ?=

CXXFLAGS ?= -O3 -std=c++17 -fPIC -Wall -Wextra -Wpedantic -flto -fno-math-errno
# Keep the fast, machine-tuned development build by default. Release targets
# explicitly clear this so their archives run on other CPUs of the same OS.
NATIVE_ARCH_FLAGS ?= -march=native
OPENMP_FLAGS ?= -fopenmp
MPFR_AVAILABLE := $(shell $(PKG_CONFIG) --exists mpfr gmp && echo yes)
OPENCL_AVAILABLE := $(shell $(PKG_CONFIG) --exists OpenCL && echo yes)

ifeq ($(MPFR_AVAILABLE),yes)
  MPFR_CFLAGS := $(shell $(PKG_CONFIG) --cflags mpfr gmp)
  MPFR_LIBS := $(shell $(PKG_CONFIG) --libs mpfr gmp)
  CPPFLAGS += -DFRACTAL_HAVE_MPFR $(MPFR_CFLAGS)
else
  $(warning MPFR/GMP were not found; shallow rendering can build, deep rendering will use Python fallback)
  MPFR_LIBS :=
endif

ifeq ($(OPENCL_AVAILABLE),yes)
  OPENCL_CFLAGS := $(shell $(PKG_CONFIG) --cflags OpenCL)
  OPENCL_LIBS := $(shell $(PKG_CONFIG) --libs OpenCL)
  CPPFLAGS += -DFRACTAL_HAVE_OPENCL $(OPENCL_CFLAGS)
else
  $(warning OpenCL headers/ICD were not found; backend=2 will be unavailable)
  OPENCL_LIBS :=
endif

.PHONY: all test benchmark gui live preview package package-linux package-windows package-windows-exe clean

all: mandelbrot.so

mandelbrot.so: renderer.cpp renderer.h opencl/mandelbrot.cl
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(NATIVE_ARCH_FLAGS) $(OPENMP_FLAGS) -shared -o $@ $< $(MPFR_LIBS) $(OPENCL_LIBS)

# MinGW uses the same C++ source and C ABI.  The Windows workflow supplies
# GMP/MPFR and copies the small set of MinGW runtime DLLs beside this file.
mandelbrot.dll: renderer.cpp renderer.h opencl/mandelbrot.cl
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(NATIVE_ARCH_FLAGS) $(OPENMP_FLAGS) -shared -o $@ $< $(MPFR_LIBS) $(OPENCL_LIBS)

test: mandelbrot.so
	python3 -m unittest discover -s tests -p 'test_*.py' -v

benchmark: mandelbrot.so
	python3 benchmark.py --renderer native --zoom 1e100 --width 256 --height 256

gui:
	python3 gui.py

live:
	python3 live_view.py

preview:
	$(PYTHON) make_preview.py

# Release builds are deliberately portable: do not bake this workstation's
# CPU instruction set into an archive intended for another machine.
package-linux:
	$(MAKE) -B NATIVE_ARCH_FLAGS= mandelbrot.so
	$(PYTHON) packaging/release.py linux --native mandelbrot.so --output $(DIST_DIR)

package-windows:
	$(MAKE) -B NATIVE_ARCH_FLAGS= mandelbrot.dll
	$(PYTHON) packaging/release.py windows --native mandelbrot.dll --output $(DIST_DIR)

package-windows-exe:
	$(MAKE) -B NATIVE_ARCH_FLAGS= mandelbrot.dll
	$(PYTHON) packaging/build_windows_exe.py --native mandelbrot.dll --runtime-dir $(WINDOWS_RUNTIME_DIR) --ffmpeg-dir $(FFMPEG_DIR) --output $(DIST_DIR)

package: package-linux

clean:
	rm -f mandelbrot.so mandelbrot.dll
