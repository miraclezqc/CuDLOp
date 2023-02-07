
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>
#include <iostream>
// #include <ATen/ATen.h>
#include <ATen/AccumulateType.h>


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

template <typename scalar_t, typename index_t>
__global__ void nll_loss_forward_reduce_cuda_kernel_1d(
    scalar_t* output,
    scalar_t* total_weight,
    scalar_t* input,
    index_t* target,
    scalar_t* weights,
    bool size_average,
    int64_t n_classes,
    int64_t ignore_index) {
  
  if (threadIdx.x == 0) return;
  const index_t t = *target;
    if (t != ignore_index) {
      CUDA_KERNEL_ASSERT(t >= 0 && t < n_classes);
      const auto cur_weight = weights != nullptr ? weights[t] : scalar_t{1};
      *total_weight = cur_weight;

      if (size_average) {
        // If we try to normalize a zero then we return a NaN
        if (cur_weight == 0) {
          *output = std::numeric_limits<scalar_t>::quiet_NaN();
        } else {
          *output = -input[t];
        }
      } else {
        *output = -cur_weight * input[t];
      }
    } else {
      // If the only element was omited, we get 0. See the discussion in
      // https://github.com/pytorch/pytorch/pull/64572#issuecomment-926504162
      *output = scalar_t{0};
      *total_weight = scalar_t{0};
  }
}

template <typename scalar_t,  typename index_t>
__global__ void nll_loss_forward_reduce_cuda_kernel_2d(
    scalar_t* output,
    scalar_t* total_weight,
    scalar_t* input,
    index_t* target,
    scalar_t* weights,
    bool size_average,
    int64_t nframe, // batch_size
    int64_t ndim,
    int64_t n_classes,
    int64_t ignore_index) {
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  __shared__ scalar_t sh_inputs[NUM_TREADS] ;
  __shared__ scalar_t  acc_weight[NUM_TREADS];

  sh_inputs[threadIdx.x] = static_cast<scalar_t>(0);
  acc_weight[threadIdx.x] = static_cast<scalar_t>(0);
  for (int i = threadIdx.x; i < nframe; i += blockDim.x) {
    index_t t = target[i];
    if (t != ignore_index) {
      CUDA_KERNEL_ASSERT(t >= 0 && t < n_classes);
      scalar_t cur_weight =
          weights != nullptr ? weights[t] : static_cast<scalar_t>(1);
      sh_inputs[threadIdx.x] -= input[i * ndim + t] * cur_weight;
      acc_weight[threadIdx.x] += cur_weight;
    }
  }

  __syncthreads();

  if (threadIdx.x == 0) {
    scalar_t output_acc = 0;
    scalar_t total_weightacc = 0;
    for (int i = 0; i < blockDim.x; ++i) {
      output_acc += sh_inputs[i];
      total_weightacc += acc_weight[i];
    }
    *total_weight = static_cast<scalar_t>(total_weightacc);
    if (size_average) {
      *output = static_cast<scalar_t>(output_acc / total_weightacc);
    } else {
      *output = static_cast<scalar_t>(output_acc);
    }
  }
}

void nll_loss_forward_out_cuda(
    torch::Tensor& output,
    torch::Tensor& total_weight,
    const torch::Tensor& input,
    const torch::Tensor& target,
    const torch::Tensor& weight,
    std::string reduction,
    int64_t ignore_index) {
 
  int64_t n_classes = input.size(-1);
  int64_t n_dims = input.dim();
  int64_t batch_size = n_dims == 1 ? 1 : input.size(0);

  if (n_dims == 1){
     AT_DISPATCH_FLOATING_TYPES_AND_HALF(
          input.scalar_type(),
          "nll_loss_forward_reduce_cuda_kernel_1d",
          [&] {
            bool size_average = (reduction == "mean");
            using index_t = int64_t;
            nll_loss_forward_reduce_cuda_kernel_1d<scalar_t, index_t>
                <<<1, std::min((int64_t)NUM_TREADS, input.size(0)), 0>>>(
                    output.data<scalar_t>(),
                    total_weight.data<scalar_t>(),
                    input.data<scalar_t>(),
                    target.data<index_t>(),
                    weight.defined() ? weight.data<scalar_t>()
                                      : nullptr,
                    size_average,
                    n_classes,
                    ignore_index);
                });
    return;
  }
  if (reduction =="none" && n_dims == 2) {
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
                    weight.defined() ? weight.data<scalar_t>()
                                      : nullptr,
                    n_classes,
                    ignore_index);
                });
      return;
  }
    else if (n_dims == 2) {
         AT_DISPATCH_FLOATING_TYPES_AND_HALF(
             input.scalar_type(),              
              "nll_loss_forward_reduce_cuda_kernel_2d",
              [&] {
                using index_t = int64_t;
                // using scalar_t = at::acc_type<scalar_t, /*is_cuda*/true>;
                nll_loss_forward_reduce_cuda_kernel_2d<scalar_t, index_t>
                    <<<1, std::min((int64_t)NUM_TREADS, input.size(0)) >>>(
                        output.data_ptr<scalar_t>(),
                        total_weight.data_ptr<scalar_t>(),
                        input.data_ptr<scalar_t>(),
                        target.data_ptr<index_t>(),
                        weight.defined() ? weight.data_ptr<scalar_t>()
                                          : nullptr,
                        reduction == "mean",
                        input.size(0),
                        input.size(1),
                        n_classes,
                        ignore_index);
                    });
              return;
            }
}

