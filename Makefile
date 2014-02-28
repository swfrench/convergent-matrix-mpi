# pull in config
include flags.mk

# local include dir
I = include

# compilation
CXXFLAGS += -I$I -I.

# dirs
O = obj
B = bin

# build products
OBJ = $O/run_ext_test.o
BIN = $B/run_ext_test.x

####

all : $(BIN)

$(BIN) : $O $B $(OBJ)
	$(CXX) $(LDFLAGS) $(BLAS_LDFLAGS) \
		$(OBJ) $(LDLIBS) $(BLAS_LDLIBS) -o $@

$O/%.o : example/%.cpp $I/*.hpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

$O :
	mkdir -p $O

$B :
	mkdir -p $B

.PHONY : clean
clean :
	rm -rf $O

.PHONY : distclean
distclean :
	rm -rf $O $B
