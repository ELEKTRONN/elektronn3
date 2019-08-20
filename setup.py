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
        'Programming Language :: Python :: 3.7',
    ],

    packages=find_packages(exclude=['scripts']),

    install_requires=[
        'torch==1.1.0',
        'numpy',
        'scipy',
        'h5py',
        'matplotlib',
        'numba',
        'tbb',
        'ipython',
        'pillow',
        'colorlog',
        'tqdm',
        'scikit-learn',
        'scikit-image',
        'tensorflow',
        'tensorboard',
        'tensorboardX',
        'torchvision==0.3.0'
    ],

)
