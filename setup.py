from distutils.core import setup, Extension

module = Extension('sobol',
                   include_dirs = ['/usr/local/include'],
                   library_dirs = ['/usr/lib', '/usr/local/lib'],
                   libraries = ['gsl', 'gslcblas'],
                   extra_compile_args = ['-std=c99'],
                   sources = ['sobol.c'],
                  )

setup(name = 'Sobol',
      version = '1.0',
      description = 'Wrapper for GSL sobol',
      author = 'Adam Scriven',
      author_email = 'adam.scriven@gmail.com',
      ext_modules=[module],
     )
