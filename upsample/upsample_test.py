import argparse
import torch
import torch.nn as nn
import numpy as np
import time
import upsample_cuda



def verifybicubic(device):    
    num_rows = 200
    num_cols = 200
    # correctness
    in_t = torch.rand((20,20,num_rows, num_cols)).to(device)
    out_my = upsample_cuda.forward(input = in_t, scale_factor=[2,2],mode="bicubic",align_corners=True, recompute_scale_factor = True)
    out_torch =  nn.Upsample(scale_factor=2, mode='bicubic',align_corners=True, recompute_scale_factor = True)(in_t)
    np.testing.assert_allclose(out_my.cpu().numpy(), out_torch.cpu().numpy(), 1e-3)
    print("bicubic testest OK with specified scale factor")
    
    out_my = upsample_cuda.forward(input = in_t, size = [300,300],mode="bicubic",align_corners=True, recompute_scale_factor = False)
    out_torch =  nn.Upsample(size = [300,300], mode='bicubic',align_corners=True, recompute_scale_factor = False)(in_t)
    np.testing.assert_allclose(out_my.cpu().numpy(), out_torch.cpu().numpy(), 1e-3)
    print("bicubic testest OK with specified size")

    
def verifytrilinear(device):    
    n = 2
    c = 2
    num_d = 100
    num_rows = 100
    num_cols = 100
    # correctness
    in_t = torch.rand((n,c,num_d, num_rows, num_cols)).to(device)
    out_my = upsample_cuda.forward(input = in_t, scale_factor=[2,2,2],mode="trilinear",align_corners=True, recompute_scale_factor = True)
    out_torch =  nn.Upsample(scale_factor=2, mode='trilinear',align_corners=True, recompute_scale_factor = True)(in_t)
    np.testing.assert_allclose(out_my.cpu().numpy(), out_torch.cpu().numpy(), 1e-3)
    print("trilinear testest OK with specified scale factor")
    
    out_my = upsample_cuda.forward(input = in_t, size = [200,200,150],mode="trilinear",align_corners=True, recompute_scale_factor = False)
    out_torch =  nn.Upsample(size = [200,200,150], mode='trilinear',align_corners=True, recompute_scale_factor = False)(in_t)
    np.testing.assert_allclose(out_my.cpu().numpy(), out_torch.cpu().numpy(), 1e-3)
    print("trilinear testest OK with specified size")
    
def runTest():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    verifybicubic(device)
    verifytrilinear(device)
    
if __name__ == "__main__":
    runTest()