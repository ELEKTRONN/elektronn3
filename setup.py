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
        # 'torch>=0.3.0',  # currently broken, see https://github.com/pytorch/pytorch/issues/566
        'numpy',
        'scipy',
        'h5py',
        'matplotlib',
        'numba',
        'ipython',
        'tqdm',
        'tensorflow-tensorboard>=0.4.0rc3',
        'tensorflow>=1.4.0',
    ],

)
