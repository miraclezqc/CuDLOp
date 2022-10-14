# This file has been changed for education and teaching purpose

import argparse
import torch
import torch.nn as nn
import numpy as np
import time
import pow_cuda

class powFunction(torch.autograd.Function):
    # Note that both forward and backward are @staticmethods
    @staticmethod
    def forward(ctx, input, exp):
        ctx.save_for_backward(input, exp)
        output = pow_cuda.forward(input, exp)

        return output[0]
        

class pow(nn.Module):
    def __init__(self):
        super(pow, self).__init__()
    
    def forward(self, input, exp):
        return powFunction.apply(input, exp)

def verify(device):    
    test_cnt = 100
    num_rows = 10000000
    exp = 2.3

    my_pow = pow().to(device)

    # correctness
    # It is a counter-example that myLinear does not guarantee correctness.
    # But you should guarantee yours:)
        
    in_t = torch.rand(num_rows).to(device)
    out_my = my_pow(in_t, exp).detach()
    out_torch =  torch.pow(in_t, exp).detach()
    np.testing.assert_allclose(out_my.cpu().numpy(), out_torch.cpu().numpy(), 1e-6)

    # time
    my_time = []
    torch_time = []
    
    for _ in range(test_cnt+10):
        in_t = torch.rand(num_rows).to(device)
        # my laeyrnrom
        start_time = time.time()
        out_my = my_pow(in_t, exp)
        torch.cuda.synchronize(device)
        end_time = time.time()
        my_time.append(end_time-start_time)
        
        # torch linear
        start_time = time.time()
        out_torch = my_pow(in_t, exp)
        torch.cuda.synchronize(device)
        end_time = time.time()
        torch_time.append(end_time-start_time)
    print(f'My Linear avg time: {sum(my_time[10:])/test_cnt}s')
    print(f'PyTorch Linear avg time: {sum(torch_time[10:])/test_cnt}s')

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
