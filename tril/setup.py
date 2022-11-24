from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='tril_cuda',
    ext_modules=[
        CUDAExtension('tril_cuda', [
            'tril_cuda.cpp',
            'tril_cuda_kernel.cu',
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })