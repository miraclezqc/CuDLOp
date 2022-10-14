from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='pow_cuda',
    ext_modules=[
        CUDAExtension('pow_cuda', [
            'pow_cuda.cpp',
            'pow_cuda_kernel.cu',
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })