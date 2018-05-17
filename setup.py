from setuptools import setup, find_packages


setup(
    name='elektronn3',
    version='0.0.0',
    description='Utilities for 3D CNNs in PyTorch',
    url='https://github.com/ELEKTRONN/elektronn3',
    author='ELEKTRONN team',
    license='MIT',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
    ],

    packages=find_packages(exclude=['scripts']),

    install_requires=[
        'torch==0.4.0',
        'numpy',
        'scipy',
        'h5py',
        'matplotlib',
        'numba',
        'ipython',
        'pillow',
        'colorlog',
        'tqdm',
        'tensorflow-tensorboard>=1.7.0',
        'tensorflow>=1.7.0',
    ],

)
