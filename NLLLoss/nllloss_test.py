import torch 
import torch.nn as nn
import numpy as np
import functional as F
import argparse

def test(dim, n_channels, n_batch, reduction_):
    if (dim == 1):
        input = n_channels*torch.rand(n_channels,).cuda()
        target = torch.randint(0, n_channels, (n_channels,)).cuda()
        out_my = F.nll_loss(input, target, reduction = reduction_)
        out_torch = nn.NLLLoss(  reduction = reduction_)(input, target)
        np.testing.assert_allclose(out_my.cpu().numpy(), out_torch.cpu().numpy(), 1e-3)
        print(f"NLLLoss passed: input size {input.size()}, n_channels {n_channels}, reduction {reduction_}")
        
    elif (dim == 2):
        input = n_channels*torch.rand(n_batch,n_channels).cuda()
        target = torch.randint(0, n_channels, (n_batch,)).cuda()
        out_my = F.nll_loss(input, target, reduction = reduction_)
        out_torch = nn.NLLLoss(reduction = reduction_)(input, target)
        np.testing.assert_allclose(out_my.cpu().numpy(), out_torch.cpu().numpy(), 1e-3)
        print(f"NLLLoss passed: input size {input.size()}, n_batchs {n_batch}, n_channels {n_channels}, reduction {reduction_}")
        
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test NLLLoss')
    parser.add_argument('--dim', type=int, default=2,
                        help='Dimension of imput')
    parser.add_argument('--n_channels', type=int, default=100,
                        help='Batch number')
    parser.add_argument('--n_batch', type=int, default=100,
                        help='Batch number')
    parser.add_argument('--reduction', type=str, default="none",
                        help='Reduction mode: support none|sum|mean')
    args = parser.parse_args()

    test(args.dim, args.n_channels, args.n_batch, args.reduction)
    