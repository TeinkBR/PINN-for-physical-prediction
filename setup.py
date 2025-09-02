from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(
    name='pinn_loss',
    ext_modules=[
        CppExtension(
            'pinn_loss',
            ['pinn_loss.cpp'],
            extra_compile_args=['-std=c++14', '-stdlib=libc++'],
            extra_link_args=['-stdlib=libc++']
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension.with_options(no_python_abi_suffix=True)
    }
)