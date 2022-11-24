#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <ATen/AccumulateType.h>

#include <assert.h>
#include <vector>

#include"UpSample.cuh"


#define NUM_BLOCK 512

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
    const int  output_width) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;



  if (index >= num_elements) {
    return;
  }

  // Special case: input and output are the same size, just copy
  const int output_x = index % output_width;
  const int output_y = index / output_width;

  

  if (input_height == output_height && input_width == output_width) {

    for (int n = 0; n < batchsize; n++) {
      for (int c = 0; c < channels; c++) {
        odata[(n*channels+c) * num_elements + index] = idata[(n*channels+c) * num_elements + index];
      }
    }
    return;
  }

  // Interpolation kernel
  accscalar_t real_x = area_pixel_compute_source_index(
      width_scale, output_x, align_corners, /*cubic=*/true);
  int in_x = floorf(real_x);
  accscalar_t t_x = real_x - in_x;

  accscalar_t real_y = area_pixel_compute_source_index(
      height_scale, output_y, align_corners, /*cubic=*/true);
  int in_y = floorf(real_y);
  accscalar_t t_y = real_y - in_y;

  for (int n = 0; n < batchsize; n++) {
    for (int c = 0; c < channels; c++) {
      accscalar_t coefficients[4];

      for (int k = 0; k < 4; k++) {
        coefficients[k] = cubic_interp1d(
            upsample_get_value_bounded<scalar_t>(
                idata, n, c, input_height, input_width, in_y - 1 + k, in_x - 1, (n*channels+c) * input_height*input_width),
            upsample_get_value_bounded<scalar_t>(
                idata, n, c, input_height, input_width, in_y - 1 + k, in_x + 0, (n*channels+c) * input_height*input_width),
            upsample_get_value_bounded<scalar_t>(
                idata, n, c, input_height, input_width, in_y - 1 + k, in_x + 1, (n*channels+c) * input_height*input_width),
            upsample_get_value_bounded<scalar_t>(
                idata, n, c, input_height, input_width, in_y - 1 + k, in_x + 2, (n*channels+c) * input_height*input_width),
            t_x);
      }

      odata[(n*channels+c) * num_elements + index]= static_cast<scalar_t>(cubic_interp1d(
          coefficients[0],
          coefficients[1],
          coefficients[2],
          coefficients[3],
          t_y));

    }
  }
}


void upsample_bicubic2d_out_cuda_template(
    torch::Tensor& output,
    const torch::Tensor& input,
    std::vector<int64_t> output_size,
    bool align_corners,
    float scales_h,
    float scales_w) {
  // may cause problem, min(num_rows, max thread per bleck)
  const int num_rows = output_size[0];
  const dim3 block(NUM_BLOCK, 1, 1);
  const dim3 grid((num_rows - 1) / NUM_BLOCK + 1, 1, 1);

 
  const int output_height = output.size(2);
  const int output_width = output.size(3);

  const int input_height = input.size(2);
  const int input_width = input.size(3);
  const int num_output_elements = output_height * output_width;
  

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "upsample_bicubic2d_out_frame", ([&] {
    using accscalar_t = at::acc_type<scalar_t, true>;
    const scalar_t rheight = area_pixel_compute_scale<scalar_t>(
            input_height, output_height, align_corners, scales_h);
    const scalar_t rwidth = area_pixel_compute_scale<scalar_t>(
        input_width, output_width, align_corners, scales_w);
    upsample_bicubic2d_out_frame<scalar_t, accscalar_t>
    <<<grid, block>>>
   (num_output_elements,rheight,rwidth,align_corners,
    input, output, input.data<scalar_t>(), output.data<scalar_t>(),
    input.size(0), input.size(1), input.size(2), input.size(3), output.size(2), output.size(3));
    }));


}

