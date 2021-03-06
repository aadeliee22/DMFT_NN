from distutils.core import setup, Extension
module_name = '_ed_solver'

include_dirs = ['./pybind11/include/']
libraries    = ['stdc++', 'arpack', 'openblas', 'lapack']

module1 = Extension(module_name,
        define_macros = [('MAJOR_VERSION', '1'), ('MINOR_VERSION', '0'), ('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')],
        include_dirs = include_dirs,
        libraries = libraries,
        sources = ['./src/ed_solver.cpp'],
        language = 'c++',
        extra_compile_args = ['-std=c++11', '-O3'])

setup (name = module_name,
       version = '1.0',
       description = '...',
       author = 'Dongkyu Kim',
       author_email = 'dkkim1005@gist.ac.kr',
       url = 'https://docs.python.org/extending/building',
       long_description = "dmft-ed solver",
       platforms = ['linux'],
       ext_modules = [module1])
