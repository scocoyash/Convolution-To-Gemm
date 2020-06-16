PROGRAM = gemm_testing

ODIR=obj
CXX=g++
CXXFLAGS=-fopenmp -O3

$(PROGRAM):
	$(CXX) $(CXXFLAGS) convolution_naive.cpp -o $(ODIR)/convolution_naive
	$(CXX) $(CXXFLAGS) gemm_naive.cpp -o $(ODIR)/gemm_naive
	$(CXX) $(CXXFLAGS) gemm_with_caching.cpp -o $(ODIR)/gemm_with_caching
	$(CXX) $(CXXFLAGS) gemm_with_tiling.cpp -o $(ODIR)/gemm_with_tiling
	$(CXX) $(CXXFLAGS) gemm_with_tiling_and_caching.cpp -o $(ODIR)/gemm_with_tiling_and_caching
	$(CXX) $(CXXFLAGS) omp_gemm_with_caching.cpp -o $(ODIR)/omp_gemm_with_caching
	$(CXX) $(CXXFLAGS) omp_gemm_with_tiling.cpp -o $(ODIR)/omp_gemm_with_tiling
	$(CXX) $(CXXFLAGS) omp_gemm_with_tiling_and_caching.cpp -o $(ODIR)/omp_gemm_with_tiling_and_caching


.PHONY: clean

clean:
	rm -f $(ODIR)/*.o