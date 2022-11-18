
import argparse
import torch
import torch.nn as nn
import numpy as np
import time
import erf_cuda

class erfFunction(torch.autograd.Function):
    # Note that both forward and backward are @staticmethods
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        output = erf_cuda.forward(input)

        return output[0]

class erf(nn.Module):
    def __init__(self):
        super(erf, self).__init__()
    
    def forward(self, input):
        return erfFunction.apply(input)


def verify(device):    
    test_cnt = 100
    num_rows = 100000

    my_erf = erf().to(device)

    # correctness
    in_t = torch.rand(num_rows).to(device)
    out_my = my_erf(in_t).detach()
    out_torch =  torch.erf(in_t).detach()
    np.testing.assert_allclose(out_my.cpu().numpy(), out_torch.cpu().numpy(), 1e-3)
    


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
