from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='select_cuda',
    ext_modules=[
        CUDAExtension('select_cuda', [
            'select_cuda.cpp',
            'select_cuda_kernel.cu',
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })