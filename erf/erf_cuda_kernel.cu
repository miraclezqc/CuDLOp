// MIT License

// Copyright (c) Microsoft Corporation.

// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

#define NUM_BLOCK 512
#define PI 3.141592653589793

inline __host__ __device__ float runtime_sqrt(float x) { return sqrtf(x); }
inline __host__ __device__ double runtime_sqrt(double x) { return sqrt(x); }

inline __host__ __device__ float runtime_exp(float x) { return expf(x); }
inline __host__ __device__ double runtime_exp(double x) { return exp(x); }

template <typename scalar_t>
__global__ void  kernel0(const scalar_t* input, scalar_t* output, const int elements) {
        if (((int)threadIdx.x < (min(((-NUM_BLOCK * (int)blockIdx.x) + (elements-1)), (NUM_BLOCK-1)) + 1))) {
          float a;
          float b;
          a = 8.0/(3.0*PI)*(PI-3.0)/(4.0-PI);
          if ((input[(((int)blockIdx.x * NUM_BLOCK) + (int)threadIdx.x)] >= 0)) {
            b = 1.0;
          }
          else {
            b = -1.0;
          }
          output[(((int)blockIdx.x * NUM_BLOCK) + (int)threadIdx.x)] = (b * runtime_sqrt((1 - runtime_exp(((((0 - input[(((int)blockIdx.x * NUM_BLOCK) + (int)threadIdx.x)]) * input[(((int)blockIdx.x * NUM_BLOCK) + (int)threadIdx.x)]) * (4.0/PI + ((a * input[(((int)blockIdx.x * NUM_BLOCK) + (int)threadIdx.x)]) * input[(((int)blockIdx.x * NUM_BLOCK) + (int)threadIdx.x)]))) / (1 + ((a * input[(((int)blockIdx.x * NUM_BLOCK) + (int)threadIdx.x)]) * input[(((int)blockIdx.x * NUM_BLOCK) + (int)threadIdx.x)])))))));
        }
}

template <typename scalar_t>
__global__ void  kernel1(const scalar_t* input, scalar_t* output, const int elements) {
    if (((int)threadIdx.x < (min(((-NUM_BLOCK * (int)blockIdx.x) + (elements-1)), NUM_BLOCK) + 1))) {
      float x2;
      float ax2;
      float b;
      if ((input[(((int)blockIdx.x * NUM_BLOCK) + (int)threadIdx.x)] >= 0)) {
        b = 1;
      }
      else {
        b = -1;
      }
      x2 = (input[(((int)blockIdx.x * NUM_BLOCK) + (int)threadIdx.x)] * input[(((int)blockIdx.x * NUM_BLOCK) + (int)threadIdx.x)]);
      ax2 = (x2 * 0.147);
      output[(((int)blockIdx.x * NUM_BLOCK) + (int)threadIdx.x)] = (b * runtime_sqrt((1 - runtime_exp((((0 - x2) * (1.273239544735 + ax2)) / (1 + ax2))))));
    }
}

std::vector<torch::Tensor> erf_cuda_forward(
    torch::Tensor input)
{
    const int num_rows = input.size(0);

    auto output = torch::zeros({num_rows}, torch::TensorOptions().device(torch::kCUDA));

    
    const dim3 block(NUM_BLOCK, 1, 1);
    const dim3 grid((num_rows - 1) / NUM_BLOCK + 1, 1, 1);

    AT_DISPATCH_FLOATING_TYPES(input.type(), "erf_cuda_forward", ([&] {
        kernel1<scalar_t><<<grid, block>>>(
            input.data<scalar_t>(),
            output.data<scalar_t>(),
            num_rows);
        }));
    
    return {output};
}

