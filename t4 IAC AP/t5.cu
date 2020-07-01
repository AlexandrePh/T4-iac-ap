#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "exec_time.h"

#define DATASET_SIZE 1024


__global__
 void multiplication(float *A, float* B, float *C, int N){
   int ROW = blockIdx.y*blockDim.y+threadIdx.y;
   int COL = blockIdx.x*blockDim.x+threadIdx.x;
   if (ROW < N && COL < N) {
     float tmp_sum = 0.0f;
     for (int i = 0; i < N; i++) {
           tmp_sum += A[ROW * N + i] * B[i * N + COL];
       }
       C[ROW * N + COL] = tmp_sum;
     }
}
// Kernel function to add the elements of two arrays

int main_func(int argc, char **argv)
{
  int dimensions = 1000000
  float *h_x, *h_y,*h_result;
  float *d_x, *d_y,*d_result;

  sscanf (argv[1],"%d",&dimensions);
  printf("%d\n",dimensions );
  cudaError_t cudaError;
  int i;
  struct timeval start, stop;

  // Disable buffering entirely
  setbuf(stdout, NULL);

  // Allocating arrays on host
  printf("Allocating arrays h_x and h_y on host...");
  gettimeofday(&start, NULL);

  h_x = (float*)malloc(dimensions*dimensions*sizeof(float));
  h_y = (float*)malloc(dimensions*dimensions*sizeof(float));
  h_result = (float*)malloc(dimensions*dimensions*sizeof(float));
  // check malloc memory allocation
  if (h_x == NULL || h_y == NULL) {
	printf("Error: malloc unable to allocate memory on host.");
        return 1;
  }

  gettimeofday(&stop, NULL);
  printf("%f ms\n", timedifference_msec(start, stop));

  // Allocating array on device
  printf("Allocating array d_x and d_y and d_result on device...");
  gettimeofday(&start, NULL);

  cudaError = cudaMalloc(&d_x, dimensions*dimensions*sizeof(float));

  // check cudaMalloc memory allocation
  if (cudaError != cudaSuccess) {
	printf("cudaMalloc d_x returned error %s (code %d)\n", cudaGetErrorString(cudaError), cudaError);
        return 1;
  }

  cudaError = cudaMalloc(&d_y, dimensions*dimensions*sizeof(float));

  // check cudaMalloc memory allocation
  if (cudaError != cudaSuccess) {
	printf("cudaMalloc d_y returned error %s (code %d)\n", cudaGetErrorString(cudaError), cudaError);
        return 1;
  }

  cudaError = cudaMalloc(&d_result, dimensions*dimensions*sizeof(float));

  // check cudaMalloc memory allocation
  if (cudaError != cudaSuccess) {
  printf("cudaMalloc d_result returned error %s (code %d)\n", cudaGetErrorString(cudaError), cudaError);
        return 1;
  }
  gettimeofday(&stop, NULL);
  printf("%f ms\n", timedifference_msec(start, stop));

  // Initialize host memory
  printf("Initializing array h_x and h_y on host...");
  gettimeofday(&start, NULL);

  for (i =0; i < dimensions*dimensions; ++i) {
	h_x[i] = 1.0f;
	h_y[i] = 2.0f;
  h_result[i] = 0.0f;
  }

  gettimeofday(&stop, NULL);
  printf("%f ms\n", timedifference_msec(start, stop));

  // Copy array from host to device
  printf("Copying arrays from host to device...");
  gettimeofday(&start, NULL);

  cudaError = cudaMemcpy(d_x, h_x, dimensions*dimensions*sizeof(float), cudaMemcpyHostToDevice);

  if (cudaError != cudaSuccess) {
	printf("cudaMemcpy (h_x -> d_x) returned error %s (code %d), line(%d)\n", cudaGetErrorString(cudaError), cudaError, __LINE__);
        return 1;
  }

  cudaError = cudaMemcpy(d_y, h_y,dimensions*dimensions*sizeof(float), cudaMemcpyHostToDevice);

  if (cudaError != cudaSuccess) {
	printf("cudaMemcpy (h_y -> d_y) returned error %s (code %d), line(%d)\n", cudaGetErrorString(cudaError), cudaError, __LINE__);
        return 1;
  }
  cudaError = cudaMemcpy(d_result, h_result, dimensions*dimensions*sizeof(float), cudaMemcpyHostToDevice);

  if (cudaError != cudaSuccess) {
	printf("cudaMemcpy (h_result -> d_result) returned error %s (code %d), line(%d)\n", cudaGetErrorString(cudaError), cudaError, __LINE__);
        return 1;
  }

  gettimeofday(&stop, NULL);
  printf("%f ms\n", timedifference_msec(start, stop));

  // Run kernel on elements on the GPU
  printf("Running kernel on elemnts of d_x and d_y...");
  gettimeofday(&start, NULL);
  int blockSize = blockDim.x;
  int numBlocks = (dimensions + blockSize - 1) / blockSize;
  // add<<<numBlocks, blockSize>>>(DATASET_SIZE, d_x, d_y);
  multiplication<<<numBlocks,blockSize>>>(d_x,d_y,d_result,dimensions)


  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();

  gettimeofday(&stop, NULL);
  printf("%f ms\n", timedifference_msec(start, stop));

  // Copy array from device to host
  printf("Copying array from device (d_y) to host (h_y)...");
  gettimeofday(&start, NULL);

  cudaError = cudaMemcpy(h_result, d_result, dimensions*dimensions*sizeof(float), cudaMemcpyDeviceToHost);

  if (cudaError != cudaSuccess)
  {
	printf("cudaMemcpy (d_result -> h_result) returned error %s (code %d), line(%d)\n", cudaGetErrorString(cudaError), cudaError, __LINE__);
	return 1;
  }

  gettimeofday(&stop, NULL);
  printf("%f ms\n", timedifference_msec(start, stop));

  // Check for errors (all values should be 3.0f)
  printf("Checking for processing errors...");
  gettimeofday(&start, NULL);



  gettimeofday(&stop, NULL);
  printf("%f ms\n", timedifference_msec(start, stop));
  printf("Max error: %f\n", maxError);

  // Free memory
  printf("Freeing memory...");
  gettimeofday(&start, NULL);
  cudaFree(d_x);
  cudaFree(d_y);
  free(h_x);
  free(h_y);
  gettimeofday(&stop, NULL);
  printf("%f ms\n", timedifference_msec(start, stop));

  return 0;
}
