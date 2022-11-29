import torch 
import torch.nn as nn
import numpy as np
import functional as F

m = nn.LogSoftmax(dim=1)
input = torch.randn(1000, 100).cuda()
target = torch.randint(0, 99, (1000,)).cuda()
out_my = F.nll_loss(input, target, reduce = False)
out_torch = nn.NLLLoss(reduce = False)(input, target)
np.testing.assert_allclose(out_my.cpu().numpy(), out_torch.cpu().numpy(), 1e-3)
# import pdb; pdb.set_trace()
