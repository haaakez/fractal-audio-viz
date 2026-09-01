CXX ?= g++
PKG_CONFIG ?= pkg-config

CXXFLAGS ?= -O3 -march=native -std=c++17 -fPIC -Wall -Wextra -Wpedantic -flto -fno-math-errno
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

.PHONY: all test benchmark gui preview clean

all: mandelbrot.so

mandelbrot.so: renderer.cpp renderer.h opencl/mandelbrot.cl
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(OPENMP_FLAGS) -shared -o $@ $< $(MPFR_LIBS) $(OPENCL_LIBS)

test: mandelbrot.so
	python3 -m unittest discover -s tests -p 'test_*.py' -v

benchmark: mandelbrot.so
	python3 benchmark.py --renderer native --zoom 1e100 --width 256 --height 256

gui:
	python3 gui.py

preview:
	python3 make_preview.py

clean:
	rm -f mandelbrot.so
