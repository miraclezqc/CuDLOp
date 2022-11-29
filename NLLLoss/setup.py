from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='NLLLoss_cuda',
    ext_modules=[
        CUDAExtension('NLLLoss_cuda', [
            'NLLLoss_cuda.cpp',
            'NLLLoss_cuda_kernel.cu',
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })