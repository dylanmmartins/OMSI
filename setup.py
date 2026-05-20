
import setuptools

setuptools.setup(
    name='OMSI',
    packages=setuptools.find_packages(),
    description='Optimized MCMC spike inference',
    author='Dylan Martins',
    version='1.1.0',
    python_requires='>=3.9',
    install_requires=[
        'numpy>=1.24',
        'scipy>=1.13',
        'numba>=0.60',
        'h5py>=3.9',
        'ray>=2.0',
        'tqdm>=4.0',
    ],
)