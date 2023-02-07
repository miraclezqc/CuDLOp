## Introduction

**CuDLOp**: Accelerating Some **Longtail** Operators in Deep Learning using CUDA and Compare With Torch

## Requirements

- A recent c++ compiler(g++)
- CUDA 
- python3

## Support
- avgpool2d
- erf
- expand
- index_select
- mul
- nll_loss
- pow
- select
- sign
- stack
- tril
- upsample (mode = nearest, bicubic, trilinear)
- where

## Usage

```shell
(Take 'pow' op as an example)
git clone git@github.com:miraclezqc/CuDLOp.git
cd pow
python3 setup.py install --user
python3 pow_test.py
```
