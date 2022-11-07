
import argparse
import torch
import torch.nn as nn
import numpy as np
import time
import mul_cuda

class mulFunction(torch.autograd.Function):
    # Note that both forward and backward are @staticmethods
    @staticmethod
    def forward(ctx, input1, input2):
        ctx.save_for_backward(input1, input2)
        output = mul_cuda.forward(input1, input2)

        return output[0]
        

class mul(nn.Module):
    def __init__(self):
        super(mul, self).__init__()
    
    def forward(self, input1, input2):
        return mulFunction.apply(input1, input2)

def verify(device):    
    test_cnt = 20
    num_rows = 10000000
    scalar = [2.3]

    my_mul = mul().to(device)

    # correctness
    in_t = torch.rand(num_rows).to(device)
    other_t = torch.rand(num_rows).to(device)
    other_scalar =  torch.from_numpy(np.array(scalar, dtype="float32")).to(device)

    # shape(in_t) = shape(other_t) 
    out_my = my_mul(in_t, other_t).detach()
    out_torch =  torch.mul(in_t, other_t).detach()
    np.testing.assert_allclose(out_my.cpu().numpy(), out_torch.cpu().numpy(), 1e-6)

    # shape(other_t) = [1]
    out_my = my_mul(in_t, other_scalar).detach()
    out_torch =  torch.mul(in_t, other_scalar).detach()
    np.testing.assert_allclose(out_my.cpu().numpy(), out_torch.cpu().numpy(), 1e-6)

    # time
    my_time = []
    torch_time = []
    
    for _ in range(test_cnt+10):
        in_t = torch.rand(num_rows).to(device)
        other_t = torch.rand(num_rows).to(device)
        start_time = time.time()
        out_my = my_mul(in_t, other_t)
        torch.cuda.synchronize(device)
        end_time = time.time()
        my_time.append(end_time-start_time)
        
        start_time = time.time()
        out_torch = my_mul(in_t, other_t)
        torch.cuda.synchronize(device)
        end_time = time.time()
        torch_time.append(end_time-start_time)
    print(f'My mul avg time: {sum(my_time[10:])/test_cnt}s')
    print(f'PyTorch mul avg time: {sum(torch_time[10:])/test_cnt}s')

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
