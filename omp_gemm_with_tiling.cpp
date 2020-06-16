#include <immintrin.h>
#include <stdio.h>
#include <cassert>
#include <iostream>
#include <sys/time.h>
#include <stdio.h>
#include <unistd.h>
#include <cstdlib>
#include <algorithm>
#include <omp.h>

// Global Params
const int H = 512;
const int W = H;
const int C = H;
float x[H * C] __attribute__((aligned(256)));
float y[C * W] __attribute__((aligned(256)));
float out_tiling[H * W]  __attribute__((aligned(256)));

inline void omp_gemm_tiling
(
    const int M,                       // H
    const int N,                       // W
    const int K,                       // C
    const float *A,                    // M x K
    const float *B,                    // K x N
    float *C                           // M x N
)
{
  int block_size = 64;
  #pragma omp parallel for
  for(int i = 0; i < N; i += block_size) {
    int imin = std::min( i + block_size, N);
    for(int j = 0; j < M; j += block_size) {
      int jmin = std::min( j + block_size, M);
      C[i*N+j] = 0.0;
      for(int k = 0; k < K; k += block_size) {
        int kmin = std::min( k + block_size, K);
        for(int x = i; x < imin; x++) {
          for(int y = j; y < jmin; y++) {
            size_t c_idx = x * M + y;
            for(int z = k; z < kmin; z++) {
              C[c_idx] +=  A[x * M + z] * B[z * M  + y];
            }
          }
        }
      }
    }
  }
}

int main() {
  
  struct timeval start, end;

  // Generate random data
  srand((unsigned int)0x100);
  std::cout << "Building Matrix: ";
  for(int i = 0; i < H; i++) {
    for(int j = 0; j < C; j++) {
      x[i*C + j] = float(rand()%100) / 100.0;//drand48();
    }
  }
  for(int i = 0; i < H; i++) {
    for(int j = 0; j < W; j++) {
      y[i*W + j] = float(rand()%100) / 100.0;//drand48();
    }
  }
  for(int i = 0; i < H; i++){
    for(int j = 0; j < W; j++) {
      out_tiling[i*W + j] = 0.0;
    }
  }
  std::cout << "Done" << std::endl;

  gettimeofday(&start, NULL);
  omp_gemm_tiling(H, W, C, x, y, out_tiling);
  gettimeofday(&end, NULL);
  long seconds  = end.tv_sec  - start.tv_sec;
  long useconds = end.tv_usec - start.tv_usec;  
  float mtime = ((seconds) * 1000 + useconds/1000.0);
  printf("Omp Gemm with Tiling Elapsed time: %f milliseconds GFlops: %f\n", mtime, ((float) 2*H*W*C)/(mtime*1e6));
  
  return 0;
}