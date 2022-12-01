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
#include <cmath>

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
__global__ void kernel0(const scalar_t* input, scalar_t* output, const int elements, const int32_t ( in_h), const int32_t ( in_w), const float ( scale_h), const float ( scale_w), const int32_t ( out_h), const int32_t ( out_w)) {
    if (((int)threadIdx.x < (min(((-NUM_BLOCK * (int)blockIdx.x) + (elements-1)), (NUM_BLOCK-1)) + 1))) {
        int32_t h_idx;
        int32_t w_idx;
        h_idx = min(int32_t((runtime_mod(floorDiv<int32_t>((((int)blockIdx.x * NUM_BLOCK) + (int)threadIdx.x), out_w), out_h) / scale_h)), (in_h + -1));
        w_idx = min(int32_t((runtime_mod((((int)blockIdx.x * NUM_BLOCK) + (int)threadIdx.x), out_w) / scale_w)), (in_w + -1));
        output[(((int)blockIdx.x * NUM_BLOCK) + (int)threadIdx.x)] = input[(((floorDiv<int32_t>((((int)blockIdx.x * NUM_BLOCK) + (int)threadIdx.x), (out_h * out_w)) * (in_h * in_w)) + (h_idx * in_w)) + w_idx)];
      }
}


std::vector<torch::Tensor> nearest_cuda_forward(
    torch::Tensor input, int in_h, int in_w, float scale_h, float scale_w)
{
    const int num_rows = input.size(0);

    int out_h = int(floor(in_h * scale_h));
    int out_w = int(floor(in_w * scale_w));
    int elements = int(num_rows/(in_h * in_w) * (out_h * out_w));

    auto output = torch::zeros({elements}, torch::TensorOptions().device(torch::kCUDA));

    
    const dim3 block(NUM_BLOCK, 1, 1);
    const dim3 grid((elements - 1) / NUM_BLOCK + 1, 1, 1);


    AT_DISPATCH_FLOATING_TYPES(input.type(), "nearest_cuda_forward", ([&] {
        kernel0<scalar_t><<<grid, block>>>(
            input.data<scalar_t>(),
            output.data<scalar_t>(),
            elements,
            in_h,
            in_w,
            scale_h,
            scale_w,
            out_h,
            out_w);
        }));

    
    
    return {output};
}

