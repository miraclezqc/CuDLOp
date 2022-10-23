from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='sign_cuda',
    ext_modules=[
        CUDAExtension('sign_cuda', [
            'sign_cuda.cpp',
            'sign_cuda_kernel.cu',
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })