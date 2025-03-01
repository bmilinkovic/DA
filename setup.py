from setuptools import setup, find_packages

setup(
    name="dynamical_autonomy",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scipy",
        "matplotlib",
        "statsmodels",
        "networkx",
        "pyvis",
        "jinja2",
        "seaborn",
        "pytest",
        "jupyter",
        "pandas",
    ],
    author="Dynamical Autonomy Authors",
    author_email="",
    description="A package for analyzing autonomy in dynamical systems",
    keywords="dynamical systems, autonomy, VAR models, causality",
    url="",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
    python_requires=">=3.7",
) 