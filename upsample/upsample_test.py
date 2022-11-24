import argparse
import torch
import torch.nn as nn
import numpy as np
import time
import upsample_cuda



def verify(device):    
    num_rows = 200
    num_cols = 200
    # correctness
    in_t = torch.rand((20,20,num_rows, num_cols)).to(device)
    out_my = upsample_cuda.forward(input = in_t, scale_factor=[2,2],mode="bicubic",align_corners=False, recompute_scale_factor = True)
    out_torch =  nn.Upsample(scale_factor=2, mode='bicubic',align_corners=False, recompute_scale_factor = True)(in_t)
    np.testing.assert_allclose(out_my.cpu().numpy(), out_torch.cpu().numpy(), 1e-3)
    # import pdb; pdb.set_trace()
    
def runTest():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    verify(device)
    
if __name__ == "__main__":
    runTest()