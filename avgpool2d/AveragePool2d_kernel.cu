#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <ATen/AccumulateType.h>

#include <assert.h>


#define NUM_TREADS 512
#define NUM_BLOCKS 128

#define CUDA_KERNEL_LOOP_TYPE(i, n, index_type)                         \
  int64_t _i_n_d_e_x = blockIdx.x * blockDim.x + threadIdx.x;           \
  for (index_type i=_i_n_d_e_x; _i_n_d_e_x < (n); _i_n_d_e_x+=blockDim.x * gridDim.x, i=_i_n_d_e_x)

#define CUDA_KERNEL_LOOP(i, n) CUDA_KERNEL_LOOP_TYPE(i, n, int)



__device__ inline int kernel_min(int a, int b) {
  return a <= b ? a : b;
}

__device__ inline int kelnel_max(int a, int b) {
  return a >= b ? a : b;
}

template <typename scalar_t, typename accscalar_t>
__global__ void avg_pool2d_out_cuda_frame(const int nthreads,
    const scalar_t* const bottom_data, const int64_t channels,
    const int64_t height, const int64_t width, const int64_t pooled_height,
    const int64_t pooled_width, const int64_t kernel_h, const int64_t kernel_w,
    const int64_t stride_h, const int64_t stride_w, const int64_t pad_h, const int64_t pad_w,
    scalar_t* const top_data, const int divisor_override,
    const bool count_include_pad, const bool use_divisor) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int pw = index % pooled_width;
    const int ph = (index / pooled_width) % pooled_height;
    const int c = (index / pooled_width / pooled_height) % channels;
    const int n = index / pooled_width / pooled_height / channels;
    int hstart = ph * stride_h - pad_h;
    int wstart = pw * stride_w - pad_w;
    int hend = kernel_min(hstart + kernel_h, height + pad_h);
    int wend = kernel_min(wstart + kernel_w, width + pad_w);
    const int pool_size = (hend - hstart) * (wend - wstart);
    hstart = kelnel_max(hstart, 0);
    wstart = kelnel_max(wstart, 0);
    hend = kernel_min(hend, height);
    wend = kernel_min(wend, width);

    if (hstart >= hend || wstart >= wend) {
      top_data[index] = scalar_t(0);
      continue;
    }

    accscalar_t aveval = accscalar_t(0);
    const scalar_t* const bottom_slice = bottom_data + (n * channels + c) * height * width;
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        aveval += bottom_slice[h * width + w];
      }
    }
    int divide_factor;
    if (use_divisor) {
      divide_factor = divisor_override;
    } else {
      if(count_include_pad) {
        divide_factor = pool_size;
      } else {
        divide_factor = (hend - hstart) * (wend - wstart);
      }
    }
    top_data[index] = static_cast<scalar_t>(aveval / divide_factor);
  }
}



void avg_pool2d_out_cuda(
    const torch::Tensor& input,
    int64_t kH,
    int64_t kW,
    int64_t dH,
    int64_t dW,
    int64_t padH,
    int64_t padW,
    bool ceil_mode,
    bool count_include_pad,
    c10::optional<int64_t> divisor_override,
    const torch::Tensor& output){


    /* sizes */
    const int64_t n_channels = input.size(-3);
    const int64_t inputHeight = input.size(-2);
    const int64_t inputWidth = input.size(-1);

    int64_t outputWidth = output.size(-1);
    int64_t outputHeight = output.size(-2);
    // const auto memory_format = input_.suggest_memory_format();

    // Tensor input = input_.contiguous(memory_format);

    const int64_t count = output.numel();
    const int num_threads = NUM_TREADS;
    const int num_blocks = std::min((int)(count+ num_threads-1)/num_threads, NUM_BLOCKS);

    bool use_divisor = divisor_override.has_value();
    const auto divisor_override_value = use_divisor ? divisor_override.value() : 0;

    if (count != 0) {
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(
          input.scalar_type(),
        "avg_pool2d_out_cuda_frame",
        [&] {
            using accscalar_t = at::acc_type<scalar_t, true>;

            scalar_t *output_data = output.data_ptr<scalar_t>();
            scalar_t *input_data = input.data_ptr<scalar_t>();

            avg_pool2d_out_cuda_frame<scalar_t, accscalar_t>
            <<<num_blocks,
                num_threads>>>(
                count,
                input_data,
                n_channels,
                inputHeight,
                inputWidth,
                outputHeight,
                outputWidth,
                kH,
                kW,
                dH,
                dW,
                padH,
                padW,
                output_data,
                divisor_override_value,
                count_include_pad,
                use_divisor);
            
        }
        );
    }
 }