# C++ compiler
CXX = mpicxx

# base compiler opts
CXXFLAGS = -O3 -Wall -DNOCHECK
LDFLAGS =

# blas (uses fortran interface internally, so no CXXFLAGS)
BLAS_LDFLAGS =
BLAS_LDLIBS = -framework Accelerate
