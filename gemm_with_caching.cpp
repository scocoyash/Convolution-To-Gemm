#include <immintrin.h>
#include <stdio.h>
#include <cassert>
#include <iostream>
#include <sys/time.h>
#include <stdio.h>
#include <unistd.h>
#include <cstdlib>
#include <algorithm>

// Global Params
const int H = 512;
const int W = H;
const int C = H;
float x[H * C] __attribute__((aligned(256)));
float y[C * W] __attribute__((aligned(256)));
float out_caching[H * W]  __attribute__((aligned(256)));

inline void gemm_caching
(
    const int M,                       // H
    const int N,                       // W
    const int K,                       // C
    const float *A,                    // M x K
    const float *B,                    // K x N
    float *C                           // M x N
)
{   // ikj algorithm
  for (int m=0; m<M; ++m)
  {
    for (int k=0; k<K; ++k)
    {
      size_t A_idx = m*K + k;
      for (int n=0; n<N; ++n)
      { 
        size_t B_idx = n + k*N;
        C[m*N + n] += A[A_idx] * B[B_idx];
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
      out_caching[i*W + j] = 0.0;
    }
  }
  std::cout << "Done" << std::endl;

  gettimeofday(&start, NULL);
  gemm_caching(H, W, C, x, y, out_caching);
  gettimeofday(&end, NULL);
  long seconds  = end.tv_sec  - start.tv_sec;
  long useconds = end.tv_usec - start.tv_usec;  
  float mtime = ((seconds) * 1000 + useconds/1000.0);
  printf("Caching Elapsed time: %f milliseconds GFlops: %f\n", mtime, ((float) 2*H*W*C)/(mtime*1e6));

  return 0;
}