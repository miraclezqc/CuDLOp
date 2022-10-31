from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='mul_cuda',
    ext_modules=[
        CUDAExtension('mul_cuda', [
            'mul_cuda.cpp',
            'mul_cuda_kernel.cu',
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })