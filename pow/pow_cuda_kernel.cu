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


inline __host__ __device__ float runtime_pow(float x, float exp) { return powf(x, exp); }
inline __host__ __device__ double runtime_pow(double x, double exp) { return pow(x, exp); }

template <typename scalar_t>
__global__ void  kernel0(const scalar_t* input, scalar_t* output, const double exp, const int elements) {
   if (((int)threadIdx.x < (min(((-NUM_BLOCK * (int)blockIdx.x) + (elements-1)), (NUM_BLOCK-1)) + 1))) {
     output[(((int)blockIdx.x * NUM_BLOCK) + (int)threadIdx.x)] = runtime_pow(input[(((int)blockIdx.x * NUM_BLOCK) + (int)threadIdx.x)], exp);
    }
}

std::vector<torch::Tensor> pow_cuda_forward(
    torch::Tensor input, double exp)
{
    const int num_rows = input.size(0);

    auto output = torch::zeros({num_rows}, torch::TensorOptions().device(torch::kCUDA));

    
    const dim3 block(NUM_BLOCK, 1, 1);
    const dim3 grid((num_rows - 1) / NUM_BLOCK + 1, 1, 1);

    AT_DISPATCH_FLOATING_TYPES(input.type(), "pow_cuda_forward", ([&] {
        kernel0<scalar_t><<<grid, block>>>(
            input.data<scalar_t>(),
            output.data<scalar_t>(),
            exp,
            num_rows);
        }));
    
    return {output};
}

