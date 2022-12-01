
import argparse
import torch
import torch.nn as nn
import numpy as np
import time
import math
import nearest_cuda

class nearestFunction(torch.autograd.Function):
    # Note that both forward and backward are @staticmethods
    @staticmethod
    def forward(ctx, input,in_h, in_w, scale_h, scale_w):
        ctx.save_for_backward(input,in_h, in_w, scale_h, scale_w)
        output = nearest_cuda.forward(input,in_h, in_w, scale_h, scale_w)

        return output[0]
        

class nearest(nn.Module):
    def __init__(self):
        super(nearest, self).__init__()
    
    def forward(self, input,in_h, in_w, scale_h, scale_w):
        return nearestFunction.apply(input,in_h, in_w, scale_h, scale_w)

def verify(device):   

    # 2-D condition
    scale = (1.783,1.533) # scale factor of H W
    in_h = 50
    in_w = 50
    input_shape = (40,30,in_h,in_w) # N C H W
    out_shape = (int(np.floor(in_h * scale[0])), int(np.floor(in_w * scale[1])))

    my_nearest = nearest().to(device)

    # correctness
    in_t = torch.rand(input_shape).to(device) 
    out_t = nn.Upsample(scale_factor=scale, mode='nearest')(in_t)
    out_t = torch.flatten(out_t)

    in_t = torch.flatten(in_t)
    out_my = my_nearest(in_t, in_h, in_w, scale[0], scale[1]).detach()
    
    np.testing.assert_allclose(out_t.cpu().numpy(), out_my.cpu().numpy(), 1e-6)

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
