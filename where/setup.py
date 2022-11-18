from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='where_cuda',
    ext_modules=[
        CUDAExtension('where_cuda', [
            'where_cuda.cpp',
            'where_cuda_kernel.cu',
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })