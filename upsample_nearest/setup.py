from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='nearest_cuda',
    ext_modules=[
        CUDAExtension('nearest_cuda', [
            'nearest_cuda.cpp',
            'nearest_cuda_kernel.cu',
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })