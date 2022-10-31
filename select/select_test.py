# This file has been changed for education and teaching purpose

import argparse
import torch
import torch.nn as nn
import numpy as np
import time
import select_cuda

class selectFunction(torch.autograd.Function):
    # Note that both forward and backward are @staticmethods
    @staticmethod
    def forward(ctx, input, index, prestride, poststride, output_elements):
        ctx.save_for_backward(input, index, prestride, poststride, output_elements)
        output = select_cuda.forward(input, index, prestride, poststride, output_elements)

        return output[0]
        

class select(nn.Module):
    def __init__(self):
        super(select, self).__init__()
    
    def forward(self, input, index, prestride, poststride, output_elements):
        return selectFunction.apply(input, index, prestride, poststride, output_elements)

def get_output_shape(shape, dim):
    output_shape = []
    for i in range(len(shape)):
        if i != dim:
            output_shape.append(shape[i])
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
    index = 11

    output_shape = get_output_shape(shape, dim)
    print("output_shape ", output_shape)

    pre_stride = get_prestride(shape, dim)
    post_stride = get_poststride(shape, dim)
    print("pre_stride",pre_stride)
    print("post_stride",post_stride)
    output_elements = 1 * np.product(output_shape)
    print("output_elements",output_elements)

    
    # correctness
    in_t = torch.rand(shape).to(device)
    out_torch = in_t.select(dim, index)
  
    my_select = select().to(device)
    in_t = torch.flatten(in_t)
    out_my = my_select(in_t, index, pre_stride, post_stride, output_elements).detach()
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
