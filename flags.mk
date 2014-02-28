# C++ compiler
CXX = CC

# base compiler opts
CXXFLAGS = -O3 -Wall -DNOCHECK # -DENABLE_CONSISTENCY_CHECK
LDFLAGS =

# blas (uses fortran interface internally, so no CXXFLAGS)
BLAS_LDFLAGS =
BLAS_LDLIBS =
