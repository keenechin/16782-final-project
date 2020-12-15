from setuptools import setup
from Cython.Build import cythonize
import numpy

setup(name="5 Bar Planner", include_dirs=[numpy.get_include()],
ext_modules = cythonize("five_bar.pyx"), zip_safe=False, language_level = "3")
