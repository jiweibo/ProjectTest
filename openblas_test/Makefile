LIBRARIES := openblas
INCLUDE_DIRS := /usr/local/OpenBLAS
LIBRARY_DIRS := /usr/local/OpenBLAS/lib
CXXFLAGS := -std=c++11

test_sgemm : test_sgemm.cc
	@ echo g++ $< -o $@ -I$(INCLUDE_DIRS) -L$(LIBRARY_DIRS) -l$(LIBRARIES)
	g++ $< -o $@ -I$(INCLUDE_DIRS) -L$(LIBRARY_DIRS) -l$(LIBRARIES)

.PHONY : clean
clean : 
	@- rm -rf test_sgemm
