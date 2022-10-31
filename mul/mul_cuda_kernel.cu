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

#include <assert.h>
#include <vector>

#define NUM_BLOCK 512


template <typename scalar_t>
__global__ void __launch_bounds__(NUM_BLOCK * 1 * 1) kernel0(const scalar_t* input1, const scalar_t* input2, scalar_t* output, const int elements) {
    if (((int)threadIdx.x < (min(((-NUM_BLOCK * (int)blockIdx.x) + elements), (NUM_BLOCK-1)) + 1))) {
        output[(((int)blockIdx.x * NUM_BLOCK) + (int)threadIdx.x)] = (input1[(((int)blockIdx.x * NUM_BLOCK) + (int)threadIdx.x)] * input2[(((int)blockIdx.x * NUM_BLOCK) + (int)threadIdx.x)]);
      }
}

template <typename scalar_t>
__global__ void __launch_bounds__(NUM_BLOCK * 1 * 1) kernel1(const scalar_t* input1, const scalar_t* input2, scalar_t* output, const int elements) {
    if (((int)threadIdx.x < (min(((-NUM_BLOCK * (int)blockIdx.x) + elements), (NUM_BLOCK-1)) + 1))) {
        output[(((int)blockIdx.x * NUM_BLOCK) + (int)threadIdx.x)] = (input1[(((int)blockIdx.x * NUM_BLOCK) + (int)threadIdx.x)] * input2[0]);
      }
}

std::vector<torch::Tensor> mul_cuda_forward(
    torch::Tensor input1, torch::Tensor input2)
{
    const int num_rows = input1.size(0);
    const int other_size = input2.size(0);

    auto output = torch::zeros({num_rows}, torch::TensorOptions().device(torch::kCUDA));

    
    const dim3 block(NUM_BLOCK, 1, 1);
    const dim3 grid((num_rows - 1) / NUM_BLOCK + 1, 1, 1);

    if(other_size == num_rows) {
        AT_DISPATCH_FLOATING_TYPES(input1.type(), "mul_cuda_forward", ([&] {
            kernel0<scalar_t><<<grid, block>>>(
                input1.data<scalar_t>(),
                input2.data<scalar_t>(),
                output.data<scalar_t>(),
                num_rows);
            }));
    } else if (other_size == 1) {
        AT_DISPATCH_FLOATING_TYPES(input1.type(), "mul_cuda_forward", ([&] {
            kernel1<scalar_t><<<grid, block>>>(
                input1.data<scalar_t>(),
                input2.data<scalar_t>(),
                output.data<scalar_t>(),
                num_rows);
            }));
    } else {
        assert(0); // "maybe GEMM or shape not match"
    }
    
    
    return {output};
}

