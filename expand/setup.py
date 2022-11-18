from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='expand_cuda',
    ext_modules=[
        CUDAExtension('expand_cuda', [
            'expand_cuda.cpp',
            'expand_cuda_kernel.cu',
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })