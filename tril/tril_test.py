
import argparse
import torch
import torch.nn as nn
import numpy as np
import time
import tril_cuda

class trilFunction(torch.autograd.Function):
    # Note that both forward and backward are @staticmethods
    @staticmethod
    def forward(ctx, input,row,column,diag):
        ctx.save_for_backward(input,row,column,diag)
        output = tril_cuda.forward(input,row,column,diag)

        return output[0]

class tril(nn.Module):
    def __init__(self):
        super(tril, self).__init__()
    
    def forward(self, input,row,column,diag):
        return trilFunction.apply(input,row,column,diag)


def verify(device):    
    batch_size = 77
    row = 100
    column = 88
    diag = 3
    shape = [batch_size, row, column]

    my_tril = tril().to(device)

    # correctness
    in_t = torch.rand(shape).to(device)
    out_torch =  torch.tril(in_t, diag).detach()

    in_t = torch.flatten(in_t)
    out_my = my_tril(in_t, row, column, diag).detach()
    out_my = torch.reshape(out_my, shape)

    np.testing.assert_allclose(out_my.cpu().numpy(), out_torch.cpu().numpy(), 1e-4)

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    # device = torch.device("cpu")

    verify(device)

if __name__ == '__main__':
    main()
