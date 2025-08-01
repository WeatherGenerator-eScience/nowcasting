import setuptools

setuptools.setup(
    name="nowcasting",
    version="0.0.1",
    author="Mats Robben",
    url="https://github.com/MatsRobben/nowcasting",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "numpy",
        "numba",
        "pandas",
        "matplotlib",
        "torch",
        "torchvision",
        "torchmetrics",
        "einops",
        "lightning",
        "zarr",
        "h5py",
        "netcdf4",
        "xarray",
        "dask",
        "fire",
        "omegaconf",
        "tqdm",
        "opencv-python-headless",
        "pysteps",
        "scipy",
        "codecarbon",
        "scikit-image",
        "tensorboard",
    ],
    python_requires=">=3.11",
)