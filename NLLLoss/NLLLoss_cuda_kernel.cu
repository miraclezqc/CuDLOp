#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>

#include <iostream>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>

#define NUM_TREADS 512
#define NUM_BLOCKS 128


template <typename scalar_t, typename index_t>
__global__ void nll_loss_forward_no_reduce_cuda_kernel(
    int64_t batch_size,
    const scalar_t* input,
    index_t* target,
    scalar_t* output,
    scalar_t* weights,
    int64_t n_classes,
    int64_t ignore_index) {
  

    int index = threadIdx.x;
    if (index >= batch_size) return;
    for (int i = index; i< batch_size; i+=blockDim.x){
      const auto cur_target = target[i];
      if (cur_target == ignore_index) {
                output[i] = 0;
                continue;
              }
      auto cur_weight =
          weights != nullptr ? weights[cur_target] : static_cast<scalar_t>(1);
      // printf("i %d, cur_target %d, cur_input %f, cur_weight %f \n", i, cur_target, input[i*n_classes + cur_target], cur_weight);
      output[i] =  -input[i*n_classes + cur_target] * cur_weight;
    }


}


void nll_loss_forward_out_cuda_template(
    torch::Tensor& output,
    const torch::Tensor& total_weight,
    const torch::Tensor& input,
    const torch::Tensor& target,
    const torch::Tensor& weight,
    std::string reduction,
    int64_t ignore_index) {
  // auto input = *input_.expect_contiguous();
  // auto target = *target_.expect_contiguous();

  int64_t n_classes = input.size(-1);
  int64_t n_dims = input.dim();
  int64_t batch_size = n_dims == 1 ? 1 : input.size(0);

  auto weight_ = weight.defined() ? weight.contiguous() : weight;


    printf("Entering nll_loss_forward_no_reduce_cuda_kernel");
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
          input.scalar_type(),
          "nll_loss_forward_no_reduce_cuda_kernel",
          [&] {
            using index_t = int64_t;
            nll_loss_forward_no_reduce_cuda_kernel<scalar_t, index_t>
                <<<1, std::min((int64_t)NUM_TREADS, input.size(0))>>>(
                    batch_size,
                    input.data<scalar_t>(),
                    target.data<index_t>(),
                    output.data<scalar_t>(),
                    weight_.defined() ? weight_.data<scalar_t>()
                                      : nullptr,
                    n_classes,
                    ignore_index);
                });

    return;
  }

