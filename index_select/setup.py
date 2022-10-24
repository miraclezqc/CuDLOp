from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='index_select_cuda',
    ext_modules=[
        CUDAExtension('index_select_cuda', [
            'index_select_cuda.cpp',
            'index_select_cuda_kernel.cu',
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })