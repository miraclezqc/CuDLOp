
import argparse
import torch
import torch.nn as nn
import numpy as np
import time
import where_cuda

class whereFunction(torch.autograd.Function):
    # Note that both forward and backward are @staticmethods
    @staticmethod
    def forward(ctx, cond, input1, input2):
        ctx.save_for_backward(input)
        output = where_cuda.forward(cond, input1, input2)

        return output[0]
        

class where(nn.Module):
    def __init__(self):
        super(where, self).__init__()
    
    def forward(self, cond, input1, input2):
        return whereFunction.apply(cond, input1, input2)

def verify(device):    
    test_cnt = 50
    num_rows = 10000000

    my_where = where().to(device)

    # correctness
    cond = (torch.randint(1,num_rows,(num_rows,))%2).bool().to(device)
    in1_t = torch.rand(num_rows).to(device)
    in2_t = torch.rand(num_rows).to(device)
    out_torch =  torch.where(cond,in1_t,in2_t).detach()
    out_my = my_where(cond,in1_t,in2_t).detach()
    np.testing.assert_allclose(out_my.cpu().numpy(), out_torch.cpu().numpy(), 1e-19)

    # time
    my_time = []
    torch_time = []
    for _ in range(test_cnt+10):
        in1_t = torch.rand(num_rows).to(device)
        in2_t = torch.rand(num_rows).to(device)
        cond = (torch.randint(1,num_rows,(num_rows,))%2).bool().to(device)
        start_time = time.time()
        out_my = my_where(cond,in1_t,in2_t)
        torch.cuda.synchronize(device)
        end_time = time.time()
        my_time.append(end_time-start_time)
        
        start_time = time.time()
        out_torch = torch.where(cond,in1_t,in2_t)
        torch.cuda.synchronize(device)
        end_time = time.time()
        torch_time.append(end_time-start_time)
    print(f'My where avg time: {sum(my_time[10:])/test_cnt}s')
    print(f'PyTorch where avg time: {sum(torch_time[10:])/test_cnt}s')

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
