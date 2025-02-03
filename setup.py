from setuptools import setup, find_packages

setup(
    name="DatasetBenchmark",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "torch",
        "pandas",
        "seaborn",
        "matplotlib",
        "datasets",
        "h5py",
    ],
    extras_require={"dev": ["black", "ipykernel", "ipywdigets"]},
    entry_points={
        "console_scripts": [
            # Add command-line scripts here
        ],
    },
    author="Moritz Hesche",
    author_email="mo.hesche@gmail.com",
    description="A brief description of your project",
    url="",
    classifiers=["Programming Language :: Python :: 3"],
)
