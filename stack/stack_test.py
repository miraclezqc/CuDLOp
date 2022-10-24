# This file has been changed for education and teaching purpose

import argparse
import torch
import torch.nn as nn
import numpy as np
import time
import stack_cuda

class stackFunction(torch.autograd.Function):
    # Note that both forward and backward are @staticmethods
    @staticmethod
    def forward(ctx, input1, input2, stride):
        ctx.save_for_backward(input1, input2, stride)
        output = stack_cuda.forward(input1, input2, stride)

        return output[0]
        

class stack(nn.Module):
    def __init__(self):
        super(stack, self).__init__()
    
    def forward(self, input1, input2, stride):
        return stackFunction.apply(input1, input2, stride)

def get_stride(shape, dim):
    ret = 1
    for i in range(dim):
        ret *= shape[i]
    return np.product(shape)//ret
    

def verify(device):    
    shape = (40,50,60)
    dim = 1

    stride = get_stride(shape, dim)
    print(stride)

    my_stack = stack().to(device)

    # correctness
    in_t = torch.rand(shape).to(device)
    in_t1 = torch.rand(shape).to(device)
    out_torch =  torch.stack((in_t,in_t1), dim=dim).detach()

    in_t = torch.flatten(in_t)
    in_t1 = torch.flatten(in_t1)
    out_my = my_stack(in_t, in_t1, stride).detach()
    
    out_torch =  torch.flatten(out_torch)
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
