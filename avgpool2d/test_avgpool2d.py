import torch
import torch.nn as nn

import numpy as np
import argparse

import functional as F

from typing import List, Optional


    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test NLLLoss')

    parser.add_argument('--N', type=int, default=2,
                            help='Batch number')

    parser.add_argument('--C', type=int, default=2,
                        help='Channel number')
    
    parser.add_argument('--H', type=int, default=10,
                        help='Input height')
    parser.add_argument('--W', type=int, default=10,
                        help='Input width')
    parser.add_argument('--kernel_size', type=int, default=3,
                        help='kernel_size')
    parser.add_argument('--stride', type=int, default=1,
                        help='the stride of the window. Default value is :attr:`kernel_size`')
    parser.add_argument('--padding', type=int, default=0,
                        help='implicit zero padding to be added on both sides')
    parser.add_argument('--ceil_mode', type=bool, default=False,
                        help='when True, will use `ceil` instead of `floor` to compute the output shape')
    parser.add_argument('--count_include_pad', type=bool, default=True,
                        help=' when True, will include the zero-padding in the averaging calculation.')
    parser.add_argument('--divisor_override', type=int, default=None,
                        help=' if specified, it will be used as divisor, otherwise size of the pooling region will be used.')
    

    args = parser.parse_args()

    if (args.N == 0):
        input = torch.rand(args.C, args.H, args.W).cuda()
    else:
        input = torch.rand(args.N, args.C, args.H, args.W).cuda()
        
        
    my_func = F.AvgPool2d(args.kernel_size, args.stride, args.padding, args.ceil_mode, args.count_include_pad, args.divisor_override)
    torch_func =  nn.AvgPool2d(args.kernel_size, args.stride, args.padding, args.ceil_mode, args.count_include_pad, args.divisor_override)
    
    my_output = my_func.forward(input)
    torch_output = torch_func(input)
    
    # import pdb; pdb.set_trace()
    np.testing.assert_allclose(my_output.cpu().numpy(), torch_output.cpu().numpy(), 1e-3)
    
    print(f"AvfPool2d passed")
    