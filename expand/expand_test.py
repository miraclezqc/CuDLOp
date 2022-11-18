
import argparse
import torch
import torch.nn as nn
import numpy as np
import time
import expand_cuda

class expandFunction(torch.autograd.Function):
    # Note that both forward and backward are @staticmethods
    @staticmethod
    def forward(ctx, input, shape, stride, expand_stride, output_elements):
        ctx.save_for_backward(input, shape, stride, expand_stride, output_elements)
        output = expand_cuda.forward(input, shape, stride, expand_stride, output_elements)
        return output[0]
        

class expand(nn.Module):
    def __init__(self):
        super(expand, self).__init__()
    
    def forward(self, input, shape, stride, expand_stride, output_elements):
        return expandFunction.apply(input, shape, stride, expand_stride, output_elements)

def get_offset_array(shape, expand_shape):
    assert len(shape) == len(expand_shape)
    stride = []
    expand_stride = []
    acc = np.product(shape)
    expand_acc = np.product(expand_shape)
    for dim, e_dim in zip(shape,expand_shape):
        if dim == e_dim or e_dim == -1:
            acc = acc// dim
            stride.append(acc)
            expand_acc = expand_acc// e_dim
            expand_stride.append(expand_acc)
        else:
            assert dim == 1 and e_dim > 1
            acc = acc// dim
            stride.append(acc)
            expand_acc = expand_acc// e_dim
            expand_stride.append(expand_acc)
    return stride, expand_stride



def verify(device):    
    shape = [50,1,2,1,4,1]
    expand_shape = [50,3,2,2,4,3]

    stride, expand_stride = get_offset_array(shape, expand_shape)

    output_elements = 1 * np.product(expand_shape)

    
    # correctness
    in_t = torch.rand(shape).to(device)
    out_torch =  in_t.expand(expand_shape).detach()
  
    my_expand = expand().to(device)
    in_t = torch.flatten(in_t)
    shape_t =  torch.from_numpy(np.array(shape, dtype="int32")).to(device)
    stride_t =  torch.from_numpy(np.array(stride, dtype="int32")).to(device)
    expand_stride_t =  torch.from_numpy(np.array(expand_stride, dtype="int32")).to(device)
    out_my = my_expand(in_t, shape_t, stride_t, expand_stride_t, output_elements).detach()

    np.testing.assert_allclose(out_my.cpu().numpy(), out_torch.cpu().numpy().flatten(), 1e-6)


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
