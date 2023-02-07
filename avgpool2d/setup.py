from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='AvgPool2d_cuda',
    ext_modules=[
        CUDAExtension('AvgPool2d_cuda', [
            'AveragePool2d.cpp',
            'AveragePool2d_kernel.cu',
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })

