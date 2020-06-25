#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "exec_time.h"

#define DATASET_SIZE 1000000
#define THREADS_PER_BLOCK 256

// Kernel function to add the elements of two arrays
__global__
void add(int n, float *d_x, float *d_y)
{

  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int threds_in_dim = gridDim.x * blockDim.x;
  if (index > threads_in_dim ){
    return;
  }
  while (index < n){
    d_y[index] = d_x[index] + d_y[index];
    index += threads_in_dim;
  }


}

int main_func(int argc, char **argv){



  float *h_x, *h_y;
  float *d_x, *d_y;
  cudaError_t cudaError;
  int i;
  struct timeval start, stop;

  // Disable buffering entirely
  setbuf(stdout, NULL);

  // Allocating arrays on host
  printf("Allocating arrays h_x and h_y on host...");
  gettimeofday(&start, NULL);

  h_x = (float*)malloc(DATASET_SIZE*sizeof(float));
  h_y = (float*)malloc(DATASET_SIZE*sizeof(float));

  // check malloc memory allocation
  if (h_x == NULL || h_y == NULL) {
	printf("Error: malloc unable to allocate memory on host.");
        return 1;
  }

  gettimeofday(&stop, NULL);
  printf("%f ms\n", timedifference_msec(start, stop));

  // Allocating array on device
  printf("Allocating array d_x and d_y on device...");
  gettimeofday(&start, NULL);

  cudaError = cudaMalloc(&d_x, DATASET_SIZE*sizeof(float));

  // check cudaMalloc memory allocation
  if (cudaError != cudaSuccess) {
	printf("cudaMalloc d_x returned error %s (code %d)\n", cudaGetErrorString(cudaError), cudaError);
        return 1;
  }

  cudaError = cudaMalloc(&d_y, DATASET_SIZE*sizeof(float));

  // check cudaMalloc memory allocation
  if (cudaError != cudaSuccess) {
	printf("cudaMalloc d_y returned error %s (code %d)\n", cudaGetErrorString(cudaError), cudaError);
        return 1;
  }

  gettimeofday(&stop, NULL);
  printf("%f ms\n", timedifference_msec(start, stop));

  // Initialize host memory
  printf("Initializing array h_x and h_y on host...");
  gettimeofday(&start, NULL);

  for (i =0; i < DATASET_SIZE; ++i) {
	h_x[i] = 1.0f;
	h_y[i] = 2.0f;
  }

  gettimeofday(&stop, NULL);
  printf("%f ms\n", timedifference_msec(start, stop));

  // Copy array from host to device
  printf("Copying arrays from host to device...");
  gettimeofday(&start, NULL);

  cudaError = cudaMemcpy(d_x, h_x, DATASET_SIZE*sizeof(float), cudaMemcpyHostToDevice);

  if (cudaError != cudaSuccess) {
	printf("cudaMemcpy (h_x -> d_x) returned error %s (code %d), line(%d)\n", cudaGetErrorString(cudaError), cudaError, __LINE__);
        return 1;
  }

  cudaError = cudaMemcpy(d_y, h_y, DATASET_SIZE*sizeof(float), cudaMemcpyHostToDevice);

  if (cudaError != cudaSuccess) {
	printf("cudaMemcpy (h_x -> d_x) returned error %s (code %d), line(%d)\n", cudaGetErrorString(cudaError), cudaError, __LINE__);
        return 1;
  }

  gettimeofday(&stop, NULL);
  printf("%f ms\n", timedifference_msec(start, stop));

  // Run kernel on elements on the GPU
  printf("Running kernel on elemnts of d_x and d_y..."); gettimeofday(&start, NULL);
  int blockSize = THREADS_PER_BLOCK;
  int numBlocks = (DATASET_SIZE + blockSize - 1) / blockSize;
  add<<<numBlocks, blockSize>>>(DATASET_SIZE, d_x, d_y);
  // Wait for GPU to finish before accessing on host cudaDeviceSynchronize();
  gettimeofday(&stop, NULL);
  printf("%f ms\n", timedifference_msec(start, stop));



  gettimeofday(&stop, NULL);
  printf("%f ms\n", timedifference_msec(start, stop));

  // Copy array from device to host
  printf("Copying array from device (d_y) to host (h_y)...");
  gettimeofday(&start, NULL);

  cudaError = cudaMemcpy(h_y, d_y, DATASET_SIZE*sizeof(float), cudaMemcpyDeviceToHost);

  if (cudaError != cudaSuccess)
  {
	printf("cudaMemcpy (d_y -> h_y) returned error %s (code %d), line(%d)\n", cudaGetErrorString(cudaError), cudaError, __LINE__);
	return 1;
  }

  gettimeofday(&stop, NULL);
  printf("%f ms\n", timedifference_msec(start, stop));

  // Check for errors (all values should be 3.0f)
  printf("Checking for processing errors...");
  gettimeofday(&start, NULL);

  float maxError = 0.0f;
  float diffError = 0.0f;
  for (i = 0; i < DATASET_SIZE; i++) {
    maxError = (maxError > (diffError=fabs(h_y[i]-3.0f)))? maxError : diffError;
    //printf("%d -> %f\n", i, h_y[i]);
  }

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
