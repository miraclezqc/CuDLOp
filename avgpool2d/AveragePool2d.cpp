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


#include <iostream>
#include <vector>


// C++ interface

#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)


// CUDA funciton declearition
void avg_pool2d_out_cuda
(const torch::Tensor& input_,
 int64_t kH_,
 int64_t kW_,
 int64_t dH_,
 int64_t dW_,
 int64_t padH_,
 int64_t padW_,
 bool ceil_mode,
 bool count_include_pad,
 c10::optional<int64_t> divisor_override,
 const torch::Tensor& output);

template <typename T>
static inline T div_rtn(T x, T y) {
  int q = x / y;
  int r = x % y;
  if ((r != 0) && ((r < 0) != (y < 0)))
    --q;
  return q;
}


template<typename T>
static inline T pooling_output_shape_pad_lr(
        T inputSize, T kernelSize, T pad_l, T pad_r, T stride, T dilation,
        bool ceil_mode) {
    T outputSize = div_rtn<T>(
        inputSize + pad_l + pad_r - dilation * (kernelSize - 1) - 1 +
        (ceil_mode ? stride - 1 : 0), stride) + 1;
    if (ceil_mode) {
        // ensure that the last pooling starts inside the image
        // needed to avoid problems in ceil mode
        if ((outputSize - 1) * stride >= inputSize + pad_l) {
          --outputSize;
        }
    }
    return outputSize;
}

      
template<typename T>
static inline T pooling_output_shape(
      T inputSize, T kernelSize, T pad, T stride, T dilation, bool ceil_mode) {
    TORCH_CHECK(stride != 0, "stride should not be zero");
    TORCH_CHECK(pad >= 0,
                "pad must be non-negative, but got pad: ", pad);
    TORCH_CHECK(pad <= kernelSize / 2,
                "pad should be at most half of kernel size, but got pad=",
                pad, " and kernel_size=", kernelSize)
    return pooling_output_shape_pad_lr(
        inputSize, kernelSize, pad, pad, stride, dilation, ceil_mode);
}


torch::Tensor avgpool2d_forward(
    torch::Tensor& input,
    std::vector<int64_t>& kernel_size,
    std::vector<int64_t>& stride,
    std::vector<int64_t>& padding,
    bool ceil_mode,
    bool count_include_pad,
    c10::optional<int64_t> divisor_override)
{
    
    CHECK_INPUT(input);
    // #20866, #22032: Guarantee this for the official C++ API?
    TORCH_CHECK(kernel_size.size() == 1 || kernel_size.size() == 2,
        "avg_pool2d: kernel_size must either be a single int, or a tuple of two ints");
    const int64_t kH = kernel_size[0];
    const int64_t kW = kernel_size.size() == 1 ? kH : kernel_size[1];

    TORCH_CHECK(stride.empty() || stride.size() == 1 || stride.size() == 2,
        "avg_pool2d: stride must either be omitted, a single int, or a tuple of two ints");
    const int64_t dH = stride.empty() ? kH : stride[0];
    const int64_t dW = stride.empty() ? kW : stride.size() == 1 ? dH : stride[1];

    TORCH_CHECK(padding.size() == 1 || padding.size() == 2,
        "avg_pool2d: padding must either be a single int, or a tuple of two ints");
    const int64_t padH = padding[0];
    const int64_t padW = padding.size() == 1 ? padH : padding[1];

    TORCH_CHECK(!divisor_override.has_value() || divisor_override.value() != 0,
        "divisor must be not zero");

    const int64_t nbatch = input.ndimension() == 4 ? input.size(-4) : 1;
    const int64_t nInputPlane = input.size(-3);
    const int64_t inputHeight = input.size(-2);
    const int64_t inputWidth = input.size(-1);

    const int64_t outputHeight = pooling_output_shape<int64_t>(
        inputHeight, kH, padH, dH, 1, ceil_mode);
    const int64_t outputWidth =
        pooling_output_shape<int64_t>(inputWidth, kW, padW, dW, 1, ceil_mode);

    if (input.dim() == 4){
        torch::Tensor output = torch::zeros({input.size(0), input.size(1), outputHeight, outputWidth},  input.options());
        avg_pool2d_out_cuda(input, kH, kW, dH, dW, padH, padW, ceil_mode, count_include_pad, divisor_override, output);
        return output;
    }
    else if (input.dim() == 3){
        torch::Tensor output = torch::zeros({input.size(0),  outputHeight, outputWidth},  input.options());
        avg_pool2d_out_cuda(input, kH, kW, dH, dW, padH, padW, ceil_mode, count_include_pad, divisor_override, output);
        return output;
    }

  
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &avgpool2d_forward, "avgpool2d forward (CUDA)",
  py::arg("input"), py::arg("kernel_size"),  py::arg("stride")= std::vector<int64_t>{}, 
  py::arg("padding") = std::vector<int64_t>{0, 0},   py::arg("ceil_mode") = false,
   py::arg("count_include_pad") = true,  py::arg("divisor_override") = py::none());
}



