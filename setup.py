import os
import versioneer
from setuptools import setup, find_packages


on_rtd = os.environ.get('READTHEDOCS') == 'True'
if on_rtd:
    install_requires = []
else:
    install_requires = [
        'torch',
        'numpy',
        'scipy',
        'h5py',
        'matplotlib',
        'numba',
        'ipython',
        'imageio',
        'colorlog',
        'tqdm',
        'scikit-learn',
        'scikit-image',
        'tensorboardX',
        'torchvision'
    ]

setup(
    name='elektronn3',
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    description='Utilities for 3D CNNs in PyTorch',
    url='https://github.com/ELEKTRONN/elektronn3',
    author='ELEKTRONN team',
    license='MIT',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    packages=find_packages(exclude=['scripts', ]),
    install_requires=install_requires,
)