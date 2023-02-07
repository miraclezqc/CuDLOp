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

// CUDA funciton declearition
void nll_loss_forward_out_cuda(
    torch::Tensor& output,
    torch::Tensor& total_weight,
    const torch::Tensor& input_,
    const torch::Tensor& target_,
    const torch::Tensor& weight,
    std::string reduction,
    int64_t ignore_index);

// C++ interface

#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor NLLLoss_forward(
    torch::Tensor input,
    torch::Tensor target,
    torch::Tensor weight,
    bool size_average = true,
    int64_t ignore_index = -100,
    bool reduce = true,
    std::string reduction = "mean") 
{
    
    CHECK_INPUT(input);
    CHECK_INPUT(target);
    CHECK_INPUT(weight);

    TORCH_CHECK(
        input.dim() > 0 && input.dim() <= 2, "input tensor should be 1D or 2D");
    TORCH_CHECK(
        target.dim() <= 1,
        "0D or 1D target tensor expected, multi-target not supported");

    auto no_batch_dim = input.dim() == 1  && target.dim() == 0;
    TORCH_CHECK(
        no_batch_dim || (input.size(0) == target.size(0)),
        "size mismatch (got input: ",
        input.sizes(),
        ", target: ",
        target.sizes(),
        ")")

    const auto n_classes = input.size(-1);

    TORCH_CHECK(
        !weight.defined() || (weight.dim() == 1 && weight.numel() == n_classes),
        "weight tensor should be defined either for all ",
        n_classes,
        " classes or no classes"
        " but got weight tensor of shape: ",
        weight.sizes());

    TORCH_CHECK(reduction=="mean" || reduction=="none" || reduction=="sum",
            "Please specify the reduction to apply to the output: 'none' | 'mean' | 'sum'");

    const auto n_dims = input.dim();
    const auto batch_size = input.size(0);

    torch::Tensor total_weight = torch::zeros({input.size(-1)}, torch::TensorOptions().device(torch::kCUDA));
    if (n_dims == 1){
        torch::Tensor output = torch::zeros({1}, torch::TensorOptions().device(torch::kCUDA));
        nll_loss_forward_out_cuda(
            output, total_weight, input, target, weight, reduction, ignore_index);
         return output;
    }
    if (reduction == "none" && n_dims == 2) {
        torch::Tensor output = torch::zeros({input.size(0)}, torch::TensorOptions().device(torch::kCUDA));
        nll_loss_forward_out_cuda(
            output, total_weight, input, target, weight, reduction, ignore_index);
        return output;
    }
    else if (n_dims == 2){
        torch::Tensor output = torch::zeros({1}, torch::TensorOptions().device(torch::kCUDA));
        nll_loss_forward_out_cuda(
            output, total_weight, input, target, weight, reduction, ignore_index);
        return output;
    }
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &NLLLoss_forward, "NLLLoss forward (CUDA)",
  py::arg("input"), py::arg("target"),  py::arg("weight"), 
  py::arg("size_average") = py::none(),   py::arg("ignore_index") = -100,
   py::arg("reduce") = py::none(),  py::arg("reduction") = "mean");
}
