from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='erf_cuda',
    ext_modules=[
        CUDAExtension('erf_cuda', [
            'erf_cuda.cpp',
            'erf_cuda_kernel.cu',
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })