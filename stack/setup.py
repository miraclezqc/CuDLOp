from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='stack_cuda',
    ext_modules=[
        CUDAExtension('stack_cuda', [
            'stack_cuda.cpp',
            'stack_cuda_kernel.cu',
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })