#!/usr/bin/env python
import codecs
import os
import re
from setuptools import find_packages, setup

__version__ = '0.1.0'

# PROJECT SPECIFIC
NAME = "sbipix"
PACKAGES = find_packages(where="src")
META_PATH = os.path.join("src", "sbipix", "__init__.py")
CLASSIFIERS = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Topic :: Scientific/Engineering :: Astronomy",
    "Topic :: Scientific/Engineering :: Physics",
]

# Core dependencies based on your environment
INSTALL_REQUIRES = [
    "numpy>=1.26.0",
    "scipy>=1.12.0",
    "matplotlib>=3.8.0",
    "seaborn>=0.13.0",
    "astropy>=6.0.0",
    "h5py>=3.10.0",
    "torch>=2.2.0",
    "sbi>=0.23.0",
    "scikit-learn>=1.4.0",
    "statsmodels>=0.14.4",
    "tqdm>=4.66.0",
    "pandas>=2.2.0",
    "corner>=2.2.0",
    "photutils>=1.13.0",
    "astroquery>=0.4.6",
    "regions>=0.8",
    "spectres>=2.2.0",
    "dense_basis>=0.1.8"
]

# Optional dependencies for specific features
EXTRAS_REQUIRE = {
    "sampler": [
        "emcee>=3.1.0",
        "dynesty>=2.1.0",
        "zeus-mcmc>=2.5.0",
        "nautilus-sampler>=1.0.0",
    ],
    "ml": [
        "nflows>=0.14",
        "normflows>=1.7.0",
        "pyknos>=0.16.0",
        "zuko>=1.3.0",
    ],
    "sed_fitting": [
        "bagpipes>=1.1.0",
        "astro-prospector>=1.4.0",
        "astro-sedpy>=0.3.0",
        "sedpy>=1.0.0",
        "fsps>=0.4.6",
    ],
    "visualization": [
        "seaborn>=0.13.0",
        "matplotlib-scalebar>=0.8.0",
    ],
    "dev": [
        "pytest>=8.3.0",
        "jupyter>=1.0.0",
        "jupyterlab>=4.1.0",
        "ipython>=8.22.0",
        "nbconvert>=7.16.0",
    ],
    "docs": [
        "sphinx>=4.0",
        "sphinx-rtd-theme>=1.0",
        "nbsphinx>=0.8",
    ],
}

# Combine all extras for complete installation
EXTRAS_REQUIRE["all"] = list(set(sum(EXTRAS_REQUIRE.values(), [])))

# END PROJECT SPECIFIC

HERE = os.path.dirname(os.path.realpath(__file__))

def read(*parts):
    with codecs.open(os.path.join(HERE, *parts), "rb", "utf-8") as f:
        return f.read()

def find_meta(meta, meta_file=read(META_PATH)):
    meta_match = re.search(
        r"^__{meta}__ = ['\"]([^'\"]*)['\"]".format(meta=meta), meta_file, re.M
    )
    if meta_match:
        return meta_match.group(1)
    raise RuntimeError("Unable to find __{meta}__ string.".format(meta=meta))

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

if __name__ == "__main__":
    setup(
        name=NAME,
        version=__version__,
        author=find_meta("author"),
        author_email=find_meta("email"),
        maintainer=find_meta("author"),
        maintainer_email=find_meta("email"),
        url=find_meta("uri"),
        license=find_meta("license"),
        description=find_meta("description"),
        long_description=long_description,
        long_description_content_type="text/markdown",
        packages=PACKAGES,
        package_dir={"": "src"},
        include_package_data=True,
        package_data={
            'sbipix': [
                'data/*.fits',
                'data/*.hdf5',
                'data/*.h5',
                'models/*.pt',
                'models/*.pth',
                'config/*.yaml',
                'config/*.yml',
            ]
        },
        install_requires=INSTALL_REQUIRES,
        extras_require=EXTRAS_REQUIRE,
        classifiers=CLASSIFIERS,
        python_requires=">=3.8",
        zip_safe=False,
        entry_points={
            "console_scripts": [
                "sbipix=sbipix.cli:main",
            ],
        },
    )
