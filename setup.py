from setuptools import setup
from Cython.Build import cythonize

setup(name="5 Bar Planner", ext_modules = cythonize("five_bar.pyx"), zip_safe=False, language_level = "3")
