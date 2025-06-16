import setuptools
import subprocess
import os

heros_version = subprocess.run(['git','describe','--tags'], stdout=subprocess.PIPE).stdout.decode("utf-8").strip()
assert "." in heros_version

assert os.path.isfile("heros/version.py")
with open("heros/VERSION", "w", encoding="utf-8") as fh:
    fh.write(f"{heros_version}\n")

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="heros",
    version=heros_version,
    author="Ryan Urbanowicz",
    author_email="ryan.urbanowicz@cshs.org",
    description="The Heuristic Evolutionary Rule Optimization System (HEROS) is a supervised rule-based machine learning algorithm designed to agnostically model diverse 'structured' data problems and yield compact human interpretable solutions. This implementation is scikit-learn compatible.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/UrbsLab/heros",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: Other/Academic Software License",
        "Operating System :: OS Independent", 
    ],
    install_requires=[
        "numpy",
        "pandas",
        "scikit-learn",
        "setuptools",
        "paretoset",
        "scipy",
        "skrebate==0.7",
        "matplotlib",
        "seaborn",
        "collections",
        "random",
        "os",
        "time", 
        "copy", 
        "math",
        "ast",
        "networkx",
        "itertools",
        "pickle",

    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.6"
)