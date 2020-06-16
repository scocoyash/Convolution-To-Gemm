#include <immintrin.h>
#include <stdio.h>
#include <cassert>
#include <iostream>
#include <sys/time.h>
#include <stdio.h>
#include <unistd.h>
#include <cstdlib>

// Global Params
const int num_filters = 1;
const int H = 512;
const int W = H;
const int C = H;
const int F_H = 3;
const int F_W = 3;
const int out_H = (H - F_H)/1 + 1;
const int out_W = (W - F_W)/1 + 1;
float input[H][W][C];
float filters[F_H][F_W][C];
float output[out_H][out_W][num_filters];


void normalConvolution(int input_height, int input_width, int input_channels,
                        int num_kernels, int kernel_height, int kernel_width, 
                        int output_height, int output_width) { 
    // performing a normal convolution
    for (int filter = 0; filter < num_kernels; filter++) {
        for (int channel = 0; channel < input_channels; channel++) {
            for (int out_h = 0; out_h < output_height; out_h++) {
                for (int out_w = 0; out_w < output_width; out_w++) {
                    for (int k_h = 0; k_h < kernel_height; k_h++) {
                        for (int k_w = 0; k_w < kernel_width; k_w++) {
                                output[filter][out_h][out_w] += 
                                filters[channel][k_h][k_w] * 
                                input[channel][out_h + k_h][out_w + k_w];
                        }
                    }
                }
            }
        }
    }
    std::cout << "Normal Convolution Completed" << std::endl;
}

int main() {
  
  struct timeval start, end;

  // Generate random data
  srand((unsigned int)0x100);
  std::cout << "Building Matrices: ";
  for(int i = 0; i < H; i++) {
    for(int j = 0; j < W; j++) {
        if(i < out_H && j < out_W)
            output[i][j][0] = 0;
      for(int k = 0; k < C; k++) {
          input[i][j][k] = float(rand()%100) / 100.0;
      }
    }
  }
  for(int i = 0; i < F_H; i++) {
    for(int j = 0; j < F_W; j++) {
      for(int k = 0; k < C; k++) {
          filters[i][j][k] = float(rand()%100) / 100.0;
      }
    }
  }
  std::cout << "Done." << std::endl;

  gettimeofday(&start, NULL);
  // normal convolution
  normalConvolution(H, W, C, 1, F_H, F_W, out_H, out_W);
  gettimeofday(&end, NULL);
  long seconds  = end.tv_sec  - start.tv_sec;
  long useconds = end.tv_usec - start.tv_usec;  
  float mtime = ((seconds) * 1000 + useconds/1000.0);
  printf("Elapsed time: %f milliseconds GFlops: %f\n", mtime, ((float) 2*H*W*C)/(mtime*1e6));

 return 1;
}