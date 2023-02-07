#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <ATen/AccumulateType.h>

#include <assert.h>
#include <vector>

#include"UpSample.cuh"


#define NUM_TREADS 512
#define NUM_BLOCKS 128

template <typename scalar_t, typename accscalar_t>
__global__ void upsample_trilinear3d_out_frame(
    const int num_elements,
    const accscalar_t rdepth,
    const accscalar_t rheight,
    const accscalar_t rwidth,
    const bool align_corners,
    const scalar_t* idata,
    scalar_t* odata,
    const int  batchsize,
    const int channels,
    const int input_depth,
    const int input_height,
    const int input_width,
    const int output_depth,
    const int output_height,
    const int  output_width) {

    int index = threadIdx.x + blockIdx.x * blockDim.x;

    if (index >=  batchsize * channels * num_elements) return;

    // Special case: input and output are the same size, just copy
    if (input_depth == output_depth && input_height == output_height && input_width == output_width) {
      for (int oidx = index; oidx < batchsize * channels * num_elements; oidx += blockDim.x * gridDim.x){
          odata[oidx] = idata[oidx];
          // __syncthreads();
      }
      return;
    }
  

    for (int oidx = index; oidx < batchsize * channels * num_elements; oidx += blockDim.x * gridDim.x){
        const int n = oidx / (channels*num_elements);
        const int c = (oidx - n*channels*num_elements) / num_elements;
        const int t2 =  (oidx - (n*channels+c)*num_elements) / (output_height * output_width);
        const int h2 =  (oidx - (n*channels+c)*num_elements - t2 * (output_height * output_width)) /  output_width;
        const int w2 = (oidx - (n*channels+c)*num_elements - t2 * (output_height * output_width)) %  output_width;
        // printf("oidx:%d, n:%d, c:%d, d:%d, h:%d, w:%d \n", oidx, n,c,t2,h2,w2);

        const accscalar_t t1r = area_pixel_compute_source_index<accscalar_t>(
            rdepth, t2, align_corners, /*cubic=*/false);
        const int t1 = t1r;
        const int t1p = (t1 < input_depth - 1) ? 1 : 0;
        const accscalar_t t1lambda = t1r - t1;
        const accscalar_t t0lambda = static_cast<accscalar_t>(1) - t1lambda;
        //
        const accscalar_t h1r = area_pixel_compute_source_index<accscalar_t>(
            rheight, h2, align_corners, /*cubic=*/false);
        const int h1 = h1r;
        const int h1p = (h1 < input_height - 1) ? 1 : 0;
        const accscalar_t h1lambda = h1r - h1;
        const accscalar_t h0lambda = static_cast<accscalar_t>(1) - h1lambda;
        //
        const accscalar_t w1r = area_pixel_compute_source_index<accscalar_t>(
            rwidth, w2, align_corners, /*cubic=*/false);
        const int w1 = w1r;
        const int w1p = (w1 < input_width - 1) ? 1 : 0;
        const accscalar_t w1lambda = w1r - w1;
        const accscalar_t w0lambda = static_cast<accscalar_t>(1) - w1lambda;

        const accscalar_t val = t0lambda *
                (h0lambda *
                    (w0lambda * idata[(n*channels+c)*input_depth*input_height*input_width + t1*input_height*input_width + h1*input_width + w1] +
                    w1lambda * idata[(n*channels+c)*input_depth*input_height*input_width + t1*input_height*input_width + h1*input_width + w1 + w1p]) +
                h1lambda *
                    (w0lambda * idata[(n*channels+c)*input_depth*input_height*input_width + t1*input_height*input_width + (h1+h1p)*input_width + w1] +
                    w1lambda * idata[(n*channels+c)*input_depth*input_height*input_width + t1*input_height*input_width + (h1+h1p)*input_width + w1+w1p])) +
            t1lambda *
                (h0lambda *
                    (w0lambda * idata[(n*channels+c)*input_depth*input_height*input_width + (t1+t1p)*input_height*input_width + h1*input_width + w1] +
                    w1lambda * idata[(n*channels+c)*input_depth*input_height*input_width + (t1+t1p)*input_height*input_width + h1*input_width + w1+w1p]) +
                h1lambda *
                    (w0lambda * idata[(n*channels+c)*input_depth*input_height*input_width + (t1+t1p)*input_height*input_width + (h1+h1p)*input_width + w1] +
                    w1lambda * idata[(n*channels+c)*input_depth*input_height*input_width + (t1+t1p)*input_height*input_width + (h1+h1p)*input_width + w1+w1p]));
        odata[oidx] = static_cast<scalar_t>(val);
   
    }
}




void upsample_trilinear3d_out_cuda_template(
    torch::Tensor& output,
    const torch::Tensor& input,
    std::vector<int64_t> output_size,
    bool align_corners,
    float scales_d,
    float scales_h,
    float scales_w) {
       
    int output_depth = output_size[0];
    int output_height = output_size[1];
    int output_width = output_size[2];

    int input_depth = input.size(2);
    int input_height = input.size(3);
    int input_width = input.size(4);

    const int num_kernels = output_depth * output_height * output_width;
    const int num_threads = std::min(
        (int)NUM_TREADS, num_kernels);
    const int blocks = std::min(static_cast<int>(input.size(0) * input.size(1)), (int)NUM_BLOCKS);
    
    // cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        input.scalar_type(), "upsample_trilinear3d_out_frame", [&] {
            using accscalar_t = at::acc_type<scalar_t, true>;

            //auto idata = input.packed_accessor64<scalar_t, 5>();
            //auto odata = output.packed_accessor64<scalar_t, 5>();

            const accscalar_t rdepth = area_pixel_compute_scale<accscalar_t>(
                input_depth, output_depth, align_corners, scales_d);
            const accscalar_t rheight = area_pixel_compute_scale<accscalar_t>(
                input_height, output_height, align_corners, scales_h);
            const accscalar_t rwidth = area_pixel_compute_scale<accscalar_t>(
                input_width, output_width, align_corners, scales_w);

            upsample_trilinear3d_out_frame<scalar_t, accscalar_t>
                <<<blocks, num_threads, 0, 0>>>(
                    num_kernels,
                    rdepth,
                    rheight,
                    rwidth,
                    align_corners,
                    input.data<scalar_t>(), output.data<scalar_t>(),
                    input.size(0), input.size(1),
                    input_depth, input_height, input_width,
                    output_depth, output_height, output_width);

        });
}
