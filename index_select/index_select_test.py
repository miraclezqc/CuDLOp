
import argparse
import torch
import torch.nn as nn
import numpy as np
import time
import index_select_cuda

class index_selectFunction(torch.autograd.Function):
    # Note that both forward and backward are @staticmethods
    @staticmethod
    def forward(ctx, input, indices, prestride, poststride, output_elements):
        ctx.save_for_backward(input, indices, prestride, poststride, output_elements)
        output = index_select_cuda.forward(input, indices, prestride, poststride, output_elements)

        return output[0]
        

class index_select(nn.Module):
    def __init__(self):
        super(index_select, self).__init__()
    
    def forward(self, input, indices, prestride, poststride, output_elements):
        return index_selectFunction.apply(input, indices, prestride, poststride, output_elements)

def get_output_shape(shape, dim, indices):
    output_shape = shape.copy()
    output_shape[dim] = len(indices)
    return output_shape

def get_prestride(shape, dim):
    ret = 1
    for i in range(dim+1):
        ret *= shape[i]
    return np.product(shape)//ret

def get_poststride(shape, dim):
    ret = 1
    if dim == 0:
        pass
    else:
        for i in range(dim):
            ret *= shape[i]
    return np.product(shape)//ret


def verify(device):    
    shape = [40,50,60]
    dim = 0
    indices = [0, 3,20,26]
    
    output_shape = get_output_shape(shape, dim, indices)

    pre_stride = get_prestride(shape, dim)
    post_stride = get_poststride(shape, dim)
    print("pre_stride",pre_stride)
    print("post_stride",post_stride)
    output_elements = 1 * np.product(output_shape)
    print("output_elements",output_elements)

    
    # correctness
    in_t = torch.rand(shape).to(device)
    indices_t =  torch.from_numpy(np.array(indices, dtype="int32")).to(device)
    out_torch =  torch.index_select(in_t, dim, indices_t).detach()

  
    my_index_select = index_select().to(device)
    in_t = torch.flatten(in_t)
    out_my = my_index_select(in_t, indices_t, pre_stride, post_stride, output_elements).detach()
    out_my = torch.reshape(out_my, output_shape)


    np.testing.assert_allclose(out_my.cpu().numpy(), out_torch.cpu().numpy(), 1e-6)


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    verify(device)

if __name__ == '__main__':
    main()
