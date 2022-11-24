#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>

#include <iostream>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>

// C++ interface

#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Caculate the output size 
inline std::vector<int64_t> compute_output_size(
    int64_t dim,
    std::tuple<
        torch::Tensor, 
        std::vector<int64_t>,
        std::vector<double>,
        bool > closed_over_args) {
    torch::Tensor input;
    std::vector<int64_t> size;
    std::vector<double> scale_factor;
    bool recompute_scale_factor;

    std::tie(input, size, scale_factor, recompute_scale_factor) =
      closed_over_args;


    
    if (!size.empty()) {
        return size;
    } 

    
  TORCH_INTERNAL_ASSERT(!scale_factor.empty());
  auto scale_factors = scale_factor;

  if (recompute_scale_factor == false) {
    // only warn when the scales have floating values since
    // the result for ints is the same with/without recompute_scale_factor
    bool is_float_scale_factor = false;
    for (double scale : scale_factors) {
      is_float_scale_factor = floor(scale) != scale;
      if (is_float_scale_factor) {
        break;
      }
    }
    if (is_float_scale_factor) {
      TORCH_WARN(
          "Please set recompute_scale_factor=True. ");
    }
  }

  std::vector<int64_t> ret;
  for (const auto i : c10::irange(dim)) {
    ret.emplace_back(static_cast<int64_t>(
        floor(static_cast<double>(input.size(i + 2)) * scale_factors[i])));
  }
  return ret;
}

// int ceil_div(int numerator, int denominator) {
//   std::div_t res = std::div(numerator, denominator);
//   return res.rem ? (res.quot + 1) : res.quot;
// }

template <typename scalar_t, typename accscalar_t>
__global__ void upsample_bicubic2d_out_frame(
    const int num_elements,
    const accscalar_t height_scale,
    const accscalar_t width_scale,
    const bool align_corners,
    const torch::Tensor& input,
    torch::Tensor& output,
    const scalar_t* idata,
    scalar_t* odata,
    const int  batchsize,
    const int channels,
    const int input_height,
    const int input_width,
    const int output_height,
    const int  output_width);


void upsample_bicubic2d_out_cuda_template(
    torch::Tensor& output,
    const torch::Tensor& input,
    std::vector<int64_t> output_size,
    bool align_corners,
    float scales_h,
    float scales_w);

torch::Tensor upsample_cuda_bicubic2d(
    const torch::Tensor& input,
    std::vector<int64_t> output_size,
    bool align_corners,
    double scale_h,
    double scale_w) {
    // Initialize output
    // may cause problem, should force output to be on the same device as input
    torch::Tensor output = torch::zeros({input.size(0), input.size(1), output_size[0], output_size[1]}, torch::TensorOptions().device(torch::kCUDA));
    upsample_bicubic2d_out_cuda_template(output, input,output_size, align_corners, scale_h, scale_w);
    return output;
}



inline torch::Tensor interpolate(
    torch::Tensor& input,
    std::vector<int64_t>& size,
    std::vector<double>& scale_factor,
    std::string mode = "nearest",
    bool align_corners = false,
    bool recompute_scale_factor = false) {
    
    // need to check mode

    auto scale_factor_len = input.dim() - 2;
    
    
    
    
    // argument check
    if (size.empty() && scale_factor.empty()) {
        TORCH_CHECK(false, "either size or scale_factor should be defined");
    }
    if (!size.empty() && !scale_factor.empty()) {
        TORCH_CHECK(false, "only one of size or scale_factor should be defined");
    }

    if (!scale_factor.empty()) {
        if (static_cast<int64_t>(scale_factor.size()) != scale_factor_len ) {
        TORCH_CHECK(
            false,
            "scale_factor shape must match input shape -2. ",
            "Input is ",
            input.dim(),
            "D, scale_factor size is ",
            scale_factor.size());
        }
    }
    // compute scale_factor_list
    std::vector<double> scale_factor_list;
    if (!scale_factor.empty() && recompute_scale_factor) {
        for (auto i=0; i<scale_factor_len; i++){
            scale_factor_list.emplace_back(
                floor(static_cast<double>(input.size(i + 2)) * scale_factor[i])/input.size(i + 2));
        }
    }
    if (!scale_factor.empty() && !recompute_scale_factor) {
        for (auto i=0; i<scale_factor_len; i++){
            scale_factor_list.emplace_back(scale_factor[i]);
        }
    }
    if (!size.empty()) {
        for (auto i=0; i<scale_factor_len; i++){
            scale_factor_list.emplace_back(size.at(i)/input.size(i + 2));
        }
    }

    auto closed_over_args =
      std::make_tuple(input, size, scale_factor, recompute_scale_factor);
    std::vector<int64_t> output_size = compute_output_size(input.dim()-2, closed_over_args);
    

    // select upsample kernel according to the size and mode 
    if (input.dim() == 4 && mode == "bicubic"){
        return upsample_cuda_bicubic2d(
            input,
            output_size,
            align_corners,
            scale_factor_list.at(0),
            scale_factor_list.at(1)
        );
    }

    
}


// CUDA funciton declearition
torch::Tensor upsample_cuda_forward(
    torch::Tensor& input,
    std::vector<int64_t>& size,
    std::vector<double>& scale_factor,
    std::string mode = "nearest",
    bool align_corners = false,
    bool recompute_scale_factor = false){
        CHECK_INPUT(input);
        return interpolate(input,size,scale_factor,mode,
     align_corners, recompute_scale_factor);
    }

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &upsample_cuda_forward, "upsample forward (CUDA)", py::arg("input"),
        py::arg("size") = std::vector<int64_t>(0) , py::arg("scale_factor")= std::vector<double>(0), py::arg("mode")="nearest",
        py::arg("align_corners")=false, py::arg("recompute_scale_factor")=false);
}

