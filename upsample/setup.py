from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='upsample_cuda',
    ext_modules=[
        CUDAExtension('upsample_cuda', [
            'upsample_cuda.cpp',
            'upsample_bicubic_kernel.cu',
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })