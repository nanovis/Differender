from setuptools import setup, find_packages
from pathlib import Path
from differender import __version__

with open(Path(__file__).parent / 'README.md', encoding='utf-8') as f:
    long_description = f.read()
setup(
    name="differender",
    description=
    "Differentiable Volume Renderer for PyTorch written in Taichi",
    long_description=long_description,
    long_description_content_type='text/markdown',
    version=__version__,
    license='MIT',
    packages=find_packages(),
    url='https://github.com/nanovis/Differender',
    author='Dominik Engel, Feng Liang',
    install_requires=[
        'torch',
        'torchvision',
        'numpy',
        'taichi',
    ],
)
