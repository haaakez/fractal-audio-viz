CXX ?= g++
PKG_CONFIG ?= pkg-config

CXXFLAGS ?= -O3 -march=native -std=c++17 -fPIC -Wall -Wextra -Wpedantic
OPENMP_FLAGS ?= -fopenmp
MPFR_AVAILABLE := $(shell $(PKG_CONFIG) --exists mpfr gmp && echo yes)

ifeq ($(MPFR_AVAILABLE),yes)
  MPFR_CFLAGS := $(shell $(PKG_CONFIG) --cflags mpfr gmp)
  MPFR_LIBS := $(shell $(PKG_CONFIG) --libs mpfr gmp)
  CPPFLAGS += -DFRACTAL_HAVE_MPFR $(MPFR_CFLAGS)
else
  $(warning MPFR/GMP were not found; shallow rendering can build, deep rendering will use Python fallback)
  MPFR_LIBS :=
endif

.PHONY: all clean

all: mandelbrot.so

mandelbrot.so: renderer.cpp
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(OPENMP_FLAGS) -shared -o $@ $< $(MPFR_LIBS)

clean:
	rm -f mandelbrot.so
