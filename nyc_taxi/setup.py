import os
from setuptools import setup, find_packages
from typing import List


def get_requirements(file_path: str) -> List[str]:
    """Reads package requirements from a file."""
    with open(file_path, 'r') as file:
        requirements = file.read().splitlines()
    return requirements

# Read the README file for long description
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(os.path.dirname(this_directory), 'README.md'), 'r') as f:
    long_description = f.read()

setup(
    name='nyc_taxi_predictions',
    version='1.0',
    author='Ilia Koiushev',
    author_email='ilya.koyushev1@gmail.com',
    description="A machine learning pipeline for predicting NYC taxi trip durations.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    install_requires=get_requirements(os.path.join(os.path.dirname(this_directory), 'requirements.txt')),
    python_requires='>=3.8',
    classifiers=['Programming Language :: Python :: 3'],
)
