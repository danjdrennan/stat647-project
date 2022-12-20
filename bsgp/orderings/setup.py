from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(
    name='maxmin_ordering_cpp',
    ext_modules=[
        CppExtension('maxmin_cpp', ['maxMin.cpp']),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })