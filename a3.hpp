/*  Navaneethakrishnan
 *  Ramanathan
 *  nramanat
 */
#include <memory>
#include <algorithm>
#include <cuda_runtime_api.h>
#include <math.h>

#ifndef A3_HPP
#define A3_HPP

__global__ void compute_gaussian_kde(float input_value, int input_index, float* x_in, float* y_out, float n, float h);

void gaussian_kde(int n, float h, const std::vector<float>& x, std::vector<float>& y) {

    float *x_in;
    float *y_out;
    int size = n*sizeof(float);

    cudaMalloc((void **)&x_in, size);
    cudaMalloc((void **)&y_out, size);

    cudaMemcpy(x_in, &x[0], size, cudaMemcpyHostToDevice);
    cudaMemcpy(y_out, &y[0], size, cudaMemcpyHostToDevice);

    int threads_per_block = 1024;
    int blocks_per_grid = (n + threads_per_block - 1) / threads_per_block;

    for(int i=0; i < size; i++)
    {
        compute_gaussian_kde<<<blocks_per_grid, threads_per_block, threads_per_block*sizeof(float)>>>(x[i], i, x_in, y_out, n, h);
    }

    cudaDeviceSynchronize();

    cudaMemcpy(&y[0], y_out, size, cudaMemcpyDeviceToHost);

    cudaFree(x_in);
    cudaFree(y_out);
} // gaussian_kde

__global__ void compute_gaussian_kde(float input_value, int input_index, float* x_in, float* y_out, float n, float h)
{
    extern __shared__ float fh_x[];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    const float pi = 22/7;

    float x = (input_value - x_in[i]) / h;
    fh_x[tid] = (1 / pow((2 * pi),0.5)) * exp(-(x*x) / 2);

    __syncthreads();

    for (int s = blockDim.x >> 1; s > 0; s >>= 1) 
    {
	if (tid < s)
        {
            fh_x[tid] += fh_x[tid + s];
        }
		__syncthreads();
    }

    if(tid == 0)
    {
        y_out[input_index] = (fh_x[0]) / (n * h);
    }
}

#endif // A3_HPP
