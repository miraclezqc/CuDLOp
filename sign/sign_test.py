
import argparse
import torch
import torch.nn as nn
import numpy as np
import time
import sign_cuda

class signFunction(torch.autograd.Function):
    # Note that both forward and backward are @staticmethods
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        output = sign_cuda.forward(input)

        return output[0]
        

class sign(nn.Module):
    def __init__(self):
        super(sign, self).__init__()
    
    def forward(self, input):
        return signFunction.apply(input)

def verify(device):    
    test_cnt = 100
    num_rows = 10000000

    my_sign = sign().to(device)

    # correctness
    in_t = torch.rand(num_rows).to(device)
    out_my = my_sign(in_t).detach()
    out_torch =  torch.sign(in_t).detach()
    np.testing.assert_allclose(out_my.cpu().numpy(), out_torch.cpu().numpy(), 1e-6)

    # time
    my_time = []
    torch_time = []
    
    for _ in range(test_cnt+10):
        in_t = torch.rand(num_rows).to(device)
        start_time = time.time()
        out_my = my_sign(in_t)
        torch.cuda.synchronize(device)
        end_time = time.time()
        my_time.append(end_time-start_time)
        
        start_time = time.time()
        out_torch = my_sign(in_t)
        torch.cuda.synchronize(device)
        end_time = time.time()
        torch_time.append(end_time-start_time)
    print(f'My sign avg time: {sum(my_time[10:])/test_cnt}s')
    print(f'PyTorch sign avg time: {sum(torch_time[10:])/test_cnt}s')

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
