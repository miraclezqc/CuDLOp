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

template <class T> __host__ __device__ T floorDiv(T a, T b) {
    T res = a / b, rem = a % b;
    return res - (rem != 0 && ((rem < 0) != (b < 0)));
}
template <class T> __host__ __device__ T ceilDiv(T a, T b) {
    T res = a / b, rem = a % b;
    return res + (rem != 0 && ((rem < 0) == (b < 0)));
}
template <class T> __host__ __device__ T runtime_mod(T a, T b) {
    T m = a % b;
    if (m < 0) {
        // m += (b < 0) ? -b : b; // avoid this form: it is UB when b == INT_MIN
        m = (b < 0) ? m - b : m + b;
    }
    return m;
}

template <typename scalar_t>
__global__ void  kernel0(const scalar_t* input, scalar_t* output, const int32_t output_elements, const int32_t *shape, const int32_t *stride, const int32_t *expand_stride, const int32_t dim) {
    if (((int)threadIdx.x < (min(((-NUM_BLOCK * (int)blockIdx.x) + (output_elements-1)), (NUM_BLOCK-1)) + 1))) {
        int32_t idx = 0;
        for (int j = 0; j < dim; j++) {
          idx += (runtime_mod(floorDiv<int32_t>((((int)blockIdx.x * NUM_BLOCK) + (int)threadIdx.x), expand_stride[j]), shape[j]) * stride[j]);
        }
        output[(((int)blockIdx.x * NUM_BLOCK) + (int)threadIdx.x)] = input[idx];
      }
}


std::vector<torch::Tensor> expand_cuda_forward(
    torch::Tensor input, torch::Tensor shape, torch::Tensor stride, torch::Tensor expand_stride, int output_elements)
{
    const int num_rows = output_elements;
    const int dim = shape.size(0);

    auto output = torch::zeros(num_rows, torch::TensorOptions().device(torch::kCUDA));

    
    const dim3 block(NUM_BLOCK, 1, 1);
    const dim3 grid((num_rows - 1) / NUM_BLOCK + 1, 1, 1);

    AT_DISPATCH_FLOATING_TYPES(input.type(), "expand_cuda_forward", ([&] {
        kernel0<scalar_t><<<grid, block>>>(
            input.data<scalar_t>(),
            output.data<scalar_t>(),
            output_elements, 
            shape.data<int32_t>(),
            stride.data<int32_t>(),
            expand_stride.data<int32_t>(),
            dim);
        }));
    
    return {output};
}

